package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

// 推荐配置：根据内存、显存与模型结构（读取 GGUF 文件头）估算各部分占用，
// 选出速度最优且放得下的方案，生成参数与逐项原因说明。

const (
	miB = int64(1) << 20
	giB = int64(1) << 30
)

// RecInput 是推荐配置的硬件条件。
type RecInput struct {
	RAM     int64   // 内存字节
	VRAM    int64   // 显存字节，0 表示无独立显卡
	Unified bool    // 统一内存（Apple 芯片等）：显存来自内存
	Ctx     int     // 每个用户期望的上下文长度，0 表示自动
	Users   int     // 同时使用的用户/客户端数
	Mmproj  int64   // 视觉投影文件大小，0 表示不启用视觉
	MemBW   float64 // 内存带宽 GB/s（理论值），0 表示未知
	MemName string  // 内存规格说明，如 “DDR5-5600 双通道”
}

// RecMem 是占用明细中的一项。
type RecMem struct {
	Label string `json:"label"`
	GPU   int64  `json:"gpu"`
	CPU   int64  `json:"cpu"`
}

// RecReason 是一条参数选择原因。
type RecReason struct {
	Title string `json:"title"`
	Flag  string `json:"flag,omitempty"`
	Text  string `json:"text"`
}

// RecResult 是推荐结果。
type RecResult struct {
	Plan       string      `json:"plan"`     // gpu：全部在 GPU；moe：部分专家在 CPU；partial：部分层在 GPU；cpu：纯 CPU
	Summary    string      `json:"summary"`  // 一句话结论
	Speed      string      `json:"speed"`    // 速度预期
	TPS        float64     `json:"tps"`      // 估算生成速度 token/s，0 表示无法估算
	TPSUpper   bool        `json:"tpsUpper"` // TPS 只算了 CPU 部分，是上限
	CPURead    int64       `json:"cpuRead"`  // 每生成一个 token 需从内存读取的权重字节
	Params     Params      `json:"params"`
	Reasons    []RecReason `json:"reasons"`
	Warnings   []string    `json:"warnings"`
	Mem        []RecMem    `json:"mem"`
	VRAMBudget int64       `json:"vramBudget"` // 可用于模型的显存（已扣除预留）
	RAMBudget  int64       `json:"ramBudget"`
	VRAMUse    int64       `json:"vramUse"`
	RAMUse     int64       `json:"ramUse"`
	VRAMRes    int64       `json:"vramReserve"`
	RAMRes     int64       `json:"ramReserve"`
	Ctx        int         `json:"ctx"`
	CacheType  string      `json:"cacheType"`
}

// kvBytes 返回 ctx 个 token 的 KV 缓存字节；q8_0 每 32 个值占 34 字节。
func kvBytes(a *ModelArch, ctx int, ct string) int64 {
	b := a.KVPerTok * int64(ctx)
	if ct == "q8_0" {
		b = b * 34 / 64
	}
	return b
}

// computeBytes 估算计算缓冲（默认 -ub 512、开启 Flash Attention）：主要是输出 logits 与中间激活。
func computeBytes(a *ModelArch) int64 {
	const ub = 512
	return int64(a.Vocab)*ub*4 + int64(a.Embd)*ub*4*6 + 128*miB
}

func gb(b int64) string {
	if b < giB {
		return fmt.Sprintf("%d MB", b/miB)
	}
	return fmt.Sprintf("%.1f GB", float64(b)/float64(giB))
}

func ctxText(c int) string {
	if c%1024 == 0 {
		return fmt.Sprintf("%d（%dK）", c, c/1024)
	}
	return fmt.Sprint(c)
}

// ctxLadder 返回从 target 起逐级减半、不低于 4096 的候选上下文长度。
func ctxLadder(target int) []int {
	out := []int{target}
	for _, c := range []int{131072, 65536, 32768, 16384, 8192, 4096} {
		if c < target {
			out = append(out, c)
		}
	}
	return out
}

// RecommendHW 根据硬件条件与模型结构生成推荐配置。
func RecommendHW(a *ModelArch, in RecInput) RecResult {
	users := max(in.Users, 1)
	r := RecResult{}
	reason := func(title, flag, format string, args ...any) {
		r.Reasons = append(r.Reasons, RecReason{title, flag, fmt.Sprintf(format, args...)})
	}
	warn := func(format string, args ...any) { r.Warnings = append(r.Warnings, fmt.Sprintf(format, args...)) }

	// ---- 预算：扣除系统与驱动预留 ----
	r.RAMRes = max(2*giB, in.RAM*15/100)
	ramBudget := max(in.RAM-r.RAMRes, 0)
	vramBudget := int64(0)
	if in.VRAM > 0 {
		r.VRAMRes = max(768*miB, in.VRAM*6/100)
		vramBudget = max(in.VRAM-r.VRAMRes, 0)
		if in.Unified {
			// 统一内存：GPU 能用的部分也受总内存约束
			vramBudget = min(vramBudget, ramBudget)
		}
	}
	r.RAMBudget, r.VRAMBudget = ramBudget, vramBudget

	// ---- 上下文目标 ----
	ctxTrain := a.CtxTrain
	if ctxTrain <= 0 {
		ctxTrain = 32768
	}
	per := in.Ctx
	autoCtx := per <= 0
	if autoCtx {
		per = min(ctxTrain, 32768)
	} else if per > ctxTrain {
		warn("期望上下文 %d 超过模型训练长度 %d，已改为 %d；超过训练长度需要 RoPE 缩放，回答质量会下降。", per, ctxTrain, ctxTrain)
		per = ctxTrain
	}
	target := per * users
	ladder := ctxLadder(target)
	minCtx := min(target, 4096)

	comp := computeBytes(a)
	var layers int64
	for _, s := range a.LayerSize {
		layers += s
	}
	other := max(a.Total-layers-a.TokEmbd-a.Output, 0)
	if a.Tied {
		other = max(a.Total-layers-a.Output, 0) // Output 中含另存的一份词嵌入
	}
	gpuFull := layers + a.Output + other // 全部层放 GPU 时的显存权重
	moe := a.Experts > 0 && a.ExpsTotal > 0

	// pick 在 ladder 中寻找 fixed + KV ≤ budget 的最大上下文，f16 与 q8_0 中取上下文更大者（相同时取 f16）
	pick := func(fixed, budget int64, kvFrac float64) (int, string, bool) {
		best, bestCT := 0, ""
		for _, ct := range []string{"f16", "q8_0"} {
			for _, c := range ladder {
				if c < minCtx {
					break
				}
				if fixed+int64(float64(kvBytes(a, c, ct))*kvFrac) <= budget {
					if c > best {
						best, bestCT = c, ct
					}
					break
				}
			}
		}
		return best, bestCT, best > 0
	}

	p := Params{}
	var gpuW, cpuW, kvG, kvC int64
	var cpuExp int64 // 放在 CPU 上的专家权重
	var ctx int
	ct := "f16"
	ngl := 0

	switch {
	case vramBudget > 0:
		mm := in.Mmproj
		if c, t, ok := pick(gpuFull+mm+comp, vramBudget, 1); ok {
			r.Plan, ctx, ct, ngl = "gpu", c, t, a.Layers
			gpuW, cpuW = gpuFull, a.TokEmbd
			break
		}
		if moe {
			nonExp := gpuFull - a.ExpsTotal
			if c, t, ok := pick(nonExp+mm+comp, vramBudget, 1); ok {
				ctx, ct, ngl = c, t, a.Layers
				room := vramBudget - nonExp - mm - comp - kvBytes(a, c, t)
				n, moved := 0, int64(0)
				for n < a.Layers && a.ExpsTotal-moved > room {
					moved += a.LayerExps[n]
					n++
				}
				r.Plan = "moe"
				gpuW, cpuW, cpuExp = gpuFull-moved, a.TokEmbd+moved, moved
				if n >= a.Layers {
					p.CPUMoe = ptr(true)
				} else {
					p.NCPUMoe = ptr(n)
				}
				break
			}
		}
		// 部分层放 GPU：llama.cpp 从最后一层往前放；MoE 模型同时把全部专家放 CPU
		c := min(target, 16384)
		ctx = c
		if kvBytes(a, c, "f16") > giB {
			ct = "q8_0"
		}
		kvLayer := kvBytes(a, c, ct) / int64(max(a.KVLayers, 1))
		avail := vramBudget - mm - comp
		var used int64
		for i := a.Layers - 1; i >= 0; i-- {
			cost := a.LayerSize[i] + kvLayer
			if moe {
				cost -= a.LayerExps[i]
			}
			if used+cost > avail {
				break
			}
			used += cost
			ngl++
		}
		if ngl == 0 {
			r.Plan = "cpu"
			warn("显存 %s 扣除预留与计算缓冲后连一层都放不下，改为纯 CPU 运行。", gb(in.VRAM))
			break
		}
		r.Plan = "partial"
		if moe {
			p.CPUMoe = ptr(true)
			cpuExp = a.ExpsTotal
		}
		kvG = kvLayer * int64(min(ngl, a.KVLayers))
		gpuW = used - kvG
		cpuW = a.Total - gpuW
		kvC = kvBytes(a, c, ct) - kvG
	default:
		r.Plan = "cpu"
	}

	if r.Plan == "cpu" {
		gpuW, kvG, ngl = 0, 0, 0
		cpuW, cpuExp = a.Total, a.ExpsTotal
		c, t, ok := pick(a.Total+comp, ramBudget, 1)
		if !ok {
			c, t = minCtx, "q8_0"
		}
		ctx, ct = c, t
		kvC = kvBytes(a, ctx, ct)
	} else if r.Plan == "gpu" || r.Plan == "moe" {
		kvG = kvBytes(a, ctx, ct)
	}
	r.Ctx, r.CacheType = ctx, ct

	// ---- 占用汇总 ----
	cpuComp := int64(0)
	gpuComp := comp
	if r.Plan == "cpu" {
		cpuComp, gpuComp = comp, 0
	} else if r.Plan != "gpu" {
		cpuComp = comp / 2 // CPU 部分也有计算缓冲
	}
	mmG, mmC := in.Mmproj, int64(0)
	if r.Plan == "cpu" {
		mmG, mmC = 0, in.Mmproj
	}
	r.Mem = []RecMem{
		{"模型权重", gpuW, cpuW},
		{"KV 缓存（上下文）", kvG, kvC},
		{"计算缓冲", gpuComp, cpuComp},
	}
	if in.Mmproj > 0 {
		r.Mem = append(r.Mem, RecMem{"视觉投影 mmproj", mmG, mmC})
	}
	r.VRAMUse = gpuW + kvG + gpuComp + mmG
	r.RAMUse = cpuW + kvC + cpuComp + mmC
	if in.Unified {
		r.RAMUse += r.VRAMUse
	}

	// ---- 参数 ----
	gpu := r.Plan != "cpu"
	if gpu {
		p.Mode = ptr("gpu")
		if ngl >= a.Layers {
			p.NGpuLayers = ptr(FlexStr("999"))
		} else {
			p.NGpuLayers = ptr(FlexStr(fmt.Sprint(ngl)))
		}
		p.FlashAttn = ptr("on")
	} else {
		p.Mode = ptr("cpu")
	}
	p.CtxSize = ptr(ctx)
	if ct != "f16" {
		p.CacheTypeK, p.CacheTypeV = ptr(ct), ptr(ct)
		if !gpu {
			p.FlashAttn = ptr("on")
		}
	}
	if users > 1 {
		p.Parallel = ptr(users)
	}
	r.Params = p

	// ---- 原因说明 ----
	moeText := ""
	if moe {
		moeText = fmt.Sprintf("（MoE：共 %d 个专家，每个 token 只激活 %d 个，专家权重 %s）", a.Experts, a.ExpertsOn, gb(a.ExpsTotal))
	}
	need := gpuFull + in.Mmproj + comp + kvBytes(a, minCtx, "q8_0")
	switch r.Plan {
	case "gpu":
		r.Summary = fmt.Sprintf("整个模型（%s）连同上下文都能放进显存，全部由 GPU 计算，速度最快。", gb(gpuFull))
		r.Speed = "最快：生成速度主要取决于显存带宽。"
		reason("运行方式：GPU", "", "可用显存约 %s（显存 %s − 预留 %s），放得下模型权重 %s + KV 缓存 %s + 计算缓冲 %s。",
			gb(vramBudget), gb(in.VRAM), gb(r.VRAMRes), gb(gpuFull), gb(kvG), gb(comp))
		reason("全部层放显存", "-ngl 999", "模型共 %d 层，999 表示全部（含输出层）放入显存。相比 auto，结果固定、不受当时剩余显存影响。%s", a.Layers, moeText)
	case "moe":
		n := derefI(p.NCPUMoe)
		where := fmt.Sprintf("前 %d 层（共 %d 层）的专家权重", n, a.Layers)
		flag := fmt.Sprintf("-ngl 999 -ncmoe %d", n)
		if p.CPUMoe != nil {
			where, flag = "全部专家权重", "-ngl 999 -cmoe"
		}
		r.Summary = fmt.Sprintf("整个模型需要 %s 显存，放不下；但它是 MoE 模型，把%s放到内存，其余仍全部在 GPU。", gb(need), where)
		r.Speed = fmt.Sprintf("较快：每个 token 只用到 %d/%d 个专家，放在 CPU 的专家计算量小，通常远快于按层切分。", a.ExpertsOn, a.Experts)
		reason("运行方式：GPU + MoE 专家部分在 CPU", flag,
			"注意力、共享专家、输出层等每个 token 都要用的权重共 %s，全部放显存；%s；路由专家只有少数被激活，放在 CPU 对速度影响最小。剩余显存 %s 用来尽量多放专家（%s 在 GPU，%s 在 CPU）。",
			gb(gpuFull-a.ExpsTotal), moeText, gb(vramBudget-(gpuFull-a.ExpsTotal)-in.Mmproj-comp-kvG), gb(gpuW-(gpuFull-a.ExpsTotal)), gb(cpuW-a.TokEmbd))
		if p.CPUMoe == nil {
			reason("为什么是 -ncmoe 而不是 -cmoe", "", "-cmoe 把全部专家放 CPU，会浪费剩下的显存；-ncmoe %d 只移走刚好放不下的那部分，GPU 承担更多计算。若启动时仍报显存不足，把这个数加 1~2。", n)
		}
	case "partial":
		r.Summary = fmt.Sprintf("模型需要约 %s 显存，可用 %s，只能把 %d/%d 层放进显存，其余由 CPU 计算。", gb(need), gb(vramBudget), ngl, a.Layers)
		r.Speed = "中等偏慢：每个 token 都要经过 CPU 上的层，速度主要受内存带宽限制。放进显存的层越多越快。"
		reason("运行方式：GPU + CPU 分层", fmt.Sprintf("-ngl %d", ngl),
			"每层约 %s（含该层的 KV 缓存），按可用显存 %s 计算最多放 %d 层（llama.cpp 从最后一层开始放）。若启动报显存不足就减 1~2，若显存还有富余可以加大。",
			gb((gpuW+kvG)/int64(max(ngl, 1))), gb(vramBudget), ngl)
		if moe {
			reason("MoE 专家全部放 CPU", "-cmoe", "非专家部分优先放显存，能放进的层数比按整层切分多，速度更好。%s", moeText)
		}
		reason("可选：交给 -fit 自动计算", "-ngl auto -fit on", "新版 llama.cpp 也可以用 -ngl auto 配合 -fit on，按启动时的实际剩余显存自动决定层数；这里给出固定值，便于复现。")
	case "cpu":
		if in.VRAM > 0 {
			r.Summary = "显存太小，GPU 帮不上忙，使用纯 CPU 运行。"
		} else {
			r.Summary = "没有可用显存，使用纯 CPU 运行。"
		}
		r.Speed = "慢：完全依赖 CPU 与内存带宽，适合小模型（≤8B）或量化较低的版本。"
		reason("运行方式：CPU", "", "模型权重 %s 全部在内存中，由 CPU 计算。可用内存约 %s（内存 %s − 系统预留 %s）。", gb(a.Total), gb(ramBudget), gb(in.RAM), gb(r.RAMRes))
		reason("线程数保持自动", "", "llama.cpp 默认使用物理核心数，通常就是最快的；超过物理核心数反而会变慢。")
	}

	ctxWhy := ""
	switch {
	case !autoCtx && ctx >= target:
		ctxWhy = "按你填写的期望上下文设置。"
	case ctx >= target && autoCtx && target < ctxTrain:
		ctxWhy = fmt.Sprintf("模型最多支持 %s，日常对话与代码 32K 已足够，再大会占用更多显存而少有用处；需要处理长文档时可在“更多选项”中调大。", ctxText(ctxTrain))
	case ctx >= target:
		ctxWhy = fmt.Sprintf("已达到模型训练长度 %s。", ctxText(ctxTrain))
	case r.Plan == "partial":
		ctxWhy = fmt.Sprintf("期望 %s；部分层在 CPU 时，上下文越大 KV 缓存越占显存、能放进 GPU 的层越少，取 16K 兼顾长度与速度。", ctxText(target))
	default:
		ctxWhy = fmt.Sprintf("期望 %s，但%s只够放下 %s，再大会放不下。", ctxText(target), map[bool]string{true: "显存", false: "内存"}[gpu && r.Plan != "partial"], ctxText(ctx))
	}
	if users > 1 {
		ctxWhy += fmt.Sprintf(" %d 个用户同时使用，每人约 %d。", users, ctx/users)
	}
	reason("上下文长度 "+ctxText(ctx), fmt.Sprintf("-c %d", ctx),
		"上下文是模型一次能“记住”的内容。每 1K token 的 KV 缓存约 %s（%s 精度）。%s", gb(kvBytes(a, 1024, ct)), ct, ctxWhy)
	if a.SWA > 0 {
		reason("滑动窗口注意力", "", "该模型部分层只关注最近 %d 个 token，实际 KV 占用会低于这里的估算，可以适当调大上下文。", a.SWA)
	}
	if ct == "q8_0" {
		gain := "用来换取更长的上下文（f16 只能放下更短的上下文）"
		if r.Plan == "partial" {
			gain = "省下的显存可以多放几层模型"
		} else if r.Plan == "cpu" {
			gain = "减少内存占用"
		}
		reason("KV 缓存量化为 q8_0", "-ctk q8_0 -ctv q8_0", "q8_0 让 KV 缓存占用减半、几乎不影响回答质量，%s。量化 V 缓存需要开启 Flash Attention。", gain)
	} else {
		reason("KV 缓存保持 f16", "", "空间足够，使用默认精度即可，无需量化。")
	}
	if gpu {
		reason("开启 Flash Attention", "-fa on", "降低注意力计算的显存占用、加快长上下文处理；也是量化 KV 缓存的前提。旧版 llama.cpp 不认识 on 时可去掉 on 只写 -fa。")
	}
	if users > 1 {
		reason("并发槽数", fmt.Sprintf("-np %d", users), "允许 %d 个请求同时处理，总上下文在各槽之间分配。", users)
	}
	if in.Mmproj > 0 {
		reason("视觉投影", "--mmproj", "启用视觉需额外加载 mmproj（%s），已计入%s占用。", gb(in.Mmproj), map[bool]string{true: "显存", false: "内存"}[gpu])
	}

	// ---- 内存带宽与速度估算 ----
	// 生成每个 token 都要把参与计算的权重读一遍，速度 ≈ 带宽 ÷ 每 token 读取量。
	// MoE 每个 token 只激活 expertsUsed/experts 的专家；llama.cpp 在 CPU 上实际约能用到理论带宽的 65%，
	// 统一内存的 GPU 约 75%。
	frac := 1.0
	if moe && a.ExpertsOn > 0 {
		frac = float64(a.ExpertsOn) / float64(a.Experts)
	}
	readOf := func(w, exps int64) int64 { return w - exps + int64(float64(exps)*frac) }
	cpuRead := int64(0)
	if cpuW > a.TokEmbd {
		cpuRead = readOf(cpuW-a.TokEmbd, cpuExp) // 词嵌入每个 token 只取一行
	}
	gpuRead := int64(0)
	if gpuW > 0 {
		gpuRead = readOf(gpuW, max(a.ExpsTotal-cpuExp, 0))
	}
	r.CPURead = cpuRead
	if in.MemBW > 0 {
		bw := in.MemBW * float64(giB)
		sec := float64(cpuRead) / (bw * 0.65)
		if in.Unified {
			sec += float64(gpuRead) / (bw * 0.75)
		}
		if sec > 0 {
			r.TPS = 1 / sec
			r.TPSUpper = !in.Unified && gpuRead > 0
		}
		name := in.MemName
		if name == "" {
			name = "内存"
		}
		switch {
		case in.Unified:
			reason("统一内存带宽 "+fmt.Sprintf("%.0f GB/s", in.MemBW), "",
				"%s：GPU 与 CPU 共用同一块内存，生成速度直接由内存带宽决定。每生成一个 token 约读取 %s 权重%s，估算约 %.0f token/s。",
				name, gb(cpuRead+gpuRead), map[bool]string{true: "（MoE 只读取被激活的专家）", false: ""}[moe], r.TPS)
		case r.Plan == "gpu":
			reason("内存规格不影响生成速度", "", "%s（约 %.0f GB/s）：模型全部在显存中，生成速度取决于显卡的显存带宽；内存只在加载模型时用到。", name, in.MemBW)
		default:
			upper := ""
			if r.TPSUpper {
				upper = "（只算了 CPU 部分，加上 GPU 部分的时间，实际会再低一些）"
			}
			tip := "提升方法：让更多权重放进显存（换更低的量化或更大显存的显卡），或换更高带宽的内存。"
			if r.Plan == "cpu" {
				tip = "提升方法：换更高带宽的内存（如 DDR5、更多通道），或选更低的量化版本 / 每次只激活少量参数的 MoE 模型。"
			}
			if strings.Contains(name, "单通道") {
				tip += "当前为单通道，带宽只有双通道的一半：再加一条同规格内存组成双通道，这部分速度几乎翻倍。"
			}
			reason("内存带宽决定 CPU 部分的速度", "", "%s 理论带宽 %.1f GB/s，llama.cpp 实际约能用到 65%%。CPU 上每生成一个 token 要从内存读取约 %s 权重%s，所以约 %.1f token/s%s。%s",
				name, in.MemBW, gb(cpuRead), map[bool]string{true: "（MoE 只读取被激活的专家）", false: ""}[moe && cpuExp > 0], r.TPS, upper, tip)
		}
	}
	if r.TPS > 0 {
		r.Speed += fmt.Sprintf(" 估算约 %.1f token/s%s。", r.TPS, map[bool]string{true: "（上限）", false: ""}[r.TPSUpper])
	} else if in.MemBW <= 0 && r.Plan != "gpu" {
		r.Speed += " 填写内存规格后可估算具体速度。"
	}

	// ---- 风险提示 ----
	if r.TPS > 0 && r.TPS < 3 {
		warn("预计生成速度只有约 %.1f token/s，聊天体验较差。可以换更低的量化版本、参数更少的模型或 MoE 模型（如 30B-A3B 这类每次只激活少量参数的模型），或提升内存带宽。", r.TPS)
	}
	if r.RAMUse > ramBudget {
		if r.RAMUse > in.RAM {
			warn("需要内存约 %s，超过内存总量 %s：系统会反复从硬盘读取模型，速度会非常慢。建议换更低的量化版本（如 Q3_K_M / IQ3_XS）或参数更少的模型。", gb(r.RAMUse), gb(in.RAM))
		} else {
			warn("需要内存约 %s，接近内存总量 %s，运行时请关闭其它占内存的程序。", gb(r.RAMUse), gb(in.RAM))
		}
	}
	if gpu && r.Plan == "partial" && ngl*3 < a.Layers {
		warn("只有 %d/%d 层在显存中，GPU 加速效果有限。建议选择更小的量化版本，让更多层放进显存。", ngl, a.Layers)
	}
	if ctx < 8192 && target > ctx {
		warn("上下文只有 %d，多轮对话或长回复很快会写满；可以换更小的量化版本以腾出空间。", ctx)
	}
	return r
}

// recReq 是推荐配置请求。
type recReq struct {
	Variant string  `json:"variant"`
	RAMGB   float64 `json:"ramGB"`
	VRAMGB  float64 `json:"vramGB"`
	Unified bool    `json:"unified"`
	Ctx     int     `json:"ctx"`
	Users   int     `json:"users"`
	Vision  bool    `json:"vision"`
	MemBW   float64 `json:"memBW"`   // 内存带宽 GB/s
	MemName string  `json:"memName"` // 内存规格说明
}

func (a *App) handleRecommend(w http.ResponseWriter, r *http.Request, g Group) {
	var req recReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, 400, err)
		return
	}
	if req.RAMGB <= 0 {
		writeErr(w, 400, "请填写内存大小")
		return
	}
	model := g.Variants[0]
	for _, v := range g.Variants {
		if v.Path == req.Variant {
			model = v
		}
	}
	arch, err := withTimeout(30*time.Second, func() (*ModelArch, error) { return LoadModelArch(model) })
	if err != nil {
		writeErr(w, 400, "读取模型文件头失败："+err.Error())
		return
	}
	in := RecInput{RAM: int64(req.RAMGB * float64(giB)), VRAM: int64(req.VRAMGB * float64(giB)),
		Unified: req.Unified, Ctx: req.Ctx, Users: req.Users, MemBW: req.MemBW, MemName: req.MemName}
	var mmproj string
	if req.Vision && len(g.Mmprojs) > 0 {
		mmproj = g.Mmprojs[0]
		if st, err := os.Stat(mmproj); err == nil {
			in.Mmproj = st.Size()
		}
	}
	res := RecommendHW(arch, in)

	// 生成命令：内置默认值 + 推荐参数，程序路径按运行方式选择
	pl, perr := a.plan(g, runReq{Variant: model.Path, Mode: res.Params.Mode, Vision: &req.Vision, Mmproj: mmproj, Extra: ptr("")})
	bin := pl.bin
	if perr != nil || bin == "" {
		bin = "llama-server"
	}
	o := pl.opt
	o.Mode = *res.Params.Mode
	o.Port = a.cfg.Get().BasePort
	if *res.Params.Mode == "gpu" {
		o.GPUOK = true // 按填写的硬件生成，不受本机 GPU 检测结果影响
	}
	p := merge(Params{Jinja: ptr(true), Host: ptr("127.0.0.1")}, res.Params)
	args, err := BuildArgs(model, p, o)
	if err != nil {
		writeErr(w, 400, err)
		return
	}
	out := map[string]any{"result": res, "arch": arch, "model": model, "command": quoteCmd(bin, args)}
	if *res.Params.Mode == "gpu" && !pl.opt.GPUOK {
		out["warn"] = "本机 GPU 版 llama-server 未检测到可用 GPU；以上命令适用于有显卡的机器，或检查 GPU 版程序配置。"
	}
	writeJSON(w, out)
}
