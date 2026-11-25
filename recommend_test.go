package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeGGUF 生成一个只含文件头与占位数据的 gguf 文件。
func writeGGUF(t *testing.T, path string, kv [][2]any, tensors [][2]any) {
	t.Helper()
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	ws := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(len(tensors)))
	w(uint64(len(kv)))
	for _, e := range kv {
		ws(e[0].(string))
		switch v := e[1].(type) {
		case string:
			w(uint32(ggufString))
			ws(v)
		case uint32:
			w(uint32(ggufU32))
			w(v)
		case float32:
			w(uint32(ggufF32))
			w(v)
		case []string:
			w(uint32(ggufArray))
			w(uint32(ggufString))
			w(uint64(len(v)))
			for _, s := range v {
				ws(s)
			}
		case []int32:
			w(uint32(ggufArray))
			w(uint32(ggufI32))
			w(uint64(len(v)))
			for _, x := range v {
				w(x)
			}
		}
	}
	var off uint64
	for _, e := range tensors {
		ws(e[0].(string))
		w(uint32(1))
		w(uint64(e[1].(int)))
		w(uint32(0))
		w(off)
		off += uint64(e[1].(int))
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(make([]byte, off))
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadModelArch(t *testing.T) {
	dir := t.TempDir()
	kv := [][2]any{
		{"general.architecture", "qwen3moe"},
		{"general.name", "Test"},
		{"qwen3moe.block_count", uint32(4)},
		{"qwen3moe.context_length", uint32(40960)},
		{"qwen3moe.embedding_length", uint32(64)},
		{"qwen3moe.attention.head_count", uint32(8)},
		{"qwen3moe.attention.head_count_kv", []int32{2, 0, 2, 2}}, // 第 2 层无 KV
		{"qwen3moe.attention.key_length", uint32(16)},
		{"qwen3moe.expert_count", uint32(8)},
		{"qwen3moe.expert_used_count", uint32(2)},
		{"qwen3moe.rope.freq_base", float32(1e6)},
		{"tokenizer.ggml.tokens", []string{"a", "b", "c"}},
	}
	var ts [][2]any
	ts = append(ts, [2]any{"token_embd.weight", 1000})
	for i := 0; i < 4; i++ {
		ts = append(ts, [2]any{fmt.Sprintf("blk.%d.attn_q.weight", i), 200}, [2]any{fmt.Sprintf("blk.%d.ffn_up_exps.weight", i), 3000})
	}
	ts = append(ts, [2]any{"output.weight", 500}, [2]any{"output_norm.weight", 12})
	// 分片：第二个分片只有输出层
	writeGGUF(t, filepath.Join(dir, "m-00001-of-00002.gguf"), kv, ts[:9])
	writeGGUF(t, filepath.Join(dir, "m-00002-of-00002.gguf"), nil, ts[9:])
	a, err := LoadModelArch(Model{Path: filepath.Join(dir, "m-00001-of-00002.gguf"), Parts: 2})
	if err != nil {
		t.Fatal(err)
	}
	if a.Layers != 4 || a.CtxTrain != 40960 || a.Vocab != 3 || a.Experts != 8 || a.ExpertsOn != 2 {
		t.Fatalf("元数据错误：%+v", a)
	}
	if a.KVLayers != 3 || a.KVPerTok != 3*2*(16+16)*2 {
		t.Fatalf("KV 估算错误：layers=%d perTok=%d", a.KVLayers, a.KVPerTok)
	}
	if a.TokEmbd != 1000 || a.ExpsTotal != 12000 || a.LayerSize[0] != 3200 || a.LayerExps[3] != 3000 || a.Tied {
		t.Fatalf("张量统计错误：%+v", a)
	}
	// output_norm 是最后一个张量，大小由文件长度推算（含末尾对齐）
	if a.Output != 512 || a.Total != 1000+4*3200+512 {
		t.Fatalf("输出层/总量错误：output=%d total=%d", a.Output, a.Total)
	}
}

// fakeArch 构造一个均匀的模型：layers 层，每层 layer 字节（其中 exps 为专家），KV 每 token kvTok 字节。
func fakeArch(layers int, layer, exps, kvTok int64) *ModelArch {
	a := &ModelArch{Arch: "test", Layers: layers, CtxTrain: 131072, Embd: 4096, Vocab: 150000,
		KVPerTok: kvTok, KVLayers: layers, TokEmbd: 600 * miB, Output: 600 * miB}
	for i := 0; i < layers; i++ {
		a.LayerSize = append(a.LayerSize, layer)
		a.LayerExps = append(a.LayerExps, exps)
		a.ExpsTotal += exps
	}
	if exps > 0 {
		a.Experts, a.ExpertsOn = 128, 8
	}
	a.Total = int64(layers)*layer + a.TokEmbd + a.Output
	return a
}

func TestRecommendHW(t *testing.T) {
	// 8B 级 Q4（约 4.7GB），每 token KV 144KB
	small := fakeArch(36, 110*miB, 0, 144*1024)
	r := RecommendHW(small, RecInput{RAM: 32 * giB, VRAM: 24 * giB})
	if r.Plan != "gpu" || r.Ctx != 32768 || r.CacheType != "f16" || string(*r.Params.NGpuLayers) != "999" || *r.Params.FlashAttn != "on" {
		t.Fatalf("24GB 跑 8B 应全部放 GPU、32K、f16：%+v", r)
	}
	// 8GB 显存：放下权重但 32K f16 KV（4.5GB）放不下，改用 q8_0 或缩短上下文
	r = RecommendHW(small, RecInput{RAM: 32 * giB, VRAM: 8 * giB})
	if r.Plan != "gpu" || r.VRAMUse > r.VRAMBudget {
		t.Fatalf("8GB 跑 8B 应全部放 GPU 且不超预算：%+v", r)
	}
	if r.CacheType == "q8_0" && (r.Params.CacheTypeK == nil || *r.Params.CacheTypeV != "q8_0") {
		t.Fatal("q8_0 时应同时设置 -ctk/-ctv")
	}

	// MoE 30B-A3B（约 17GB，专家约 15GB）在 12GB 显存上：-ncmoe 部分专家放 CPU
	moe := fakeArch(48, 360*miB, 320*miB, 96*1024)
	r = RecommendHW(moe, RecInput{RAM: 64 * giB, VRAM: 12 * giB})
	if r.Plan != "moe" || r.Params.NCPUMoe == nil || *r.Params.NCPUMoe <= 0 || *r.Params.NCPUMoe >= 48 || r.VRAMUse > r.VRAMBudget {
		t.Fatalf("MoE 应部分专家放 CPU：%+v", r)
	}
	// 显存很小时所有专家放 CPU
	r = RecommendHW(moe, RecInput{RAM: 64 * giB, VRAM: 4 * giB})
	if r.Plan != "moe" || r.Params.CPUMoe == nil {
		t.Fatalf("4GB 显存应 -cmoe：%+v", r)
	}

	// 70B 稠密（约 40GB）在 24GB 显存上：部分层放 GPU
	big := fakeArch(80, 500*miB, 0, 320*1024)
	r = RecommendHW(big, RecInput{RAM: 64 * giB, VRAM: 24 * giB})
	n := 0
	fmt.Sscan(string(*r.Params.NGpuLayers), &n)
	if r.Plan != "partial" || n <= 20 || n >= 80 || r.VRAMUse > r.VRAMBudget {
		t.Fatalf("70B 应部分层放 GPU：ngl=%d %+v", n, r)
	}

	// 无显卡：纯 CPU；内存不足时给出提示
	r = RecommendHW(small, RecInput{RAM: 16 * giB})
	if r.Plan != "cpu" || *r.Params.Mode != "cpu" || r.Params.NGpuLayers != nil {
		t.Fatalf("无显存应为 CPU：%+v", r)
	}
	r = RecommendHW(big, RecInput{RAM: 16 * giB})
	if len(r.Warnings) == 0 || !strings.Contains(strings.Join(r.Warnings, ""), "超过内存总量") {
		t.Fatalf("内存不足应提示：%v", r.Warnings)
	}

	// 指定上下文与并发：总上下文 = 每人 × 人数
	r = RecommendHW(small, RecInput{RAM: 32 * giB, VRAM: 24 * giB, Ctx: 8192, Users: 4})
	if r.Ctx != 32768 || r.Params.Parallel == nil || *r.Params.Parallel != 4 {
		t.Fatalf("并发上下文错误：ctx=%d", r.Ctx)
	}
	// 超过训练长度时截断
	r = RecommendHW(small, RecInput{RAM: 32 * giB, VRAM: 80 * giB, Ctx: 1 << 20})
	if r.Ctx != 131072 || len(r.Warnings) == 0 {
		t.Fatalf("应截断到训练长度：ctx=%d", r.Ctx)
	}
}

func TestMemSpecParse(t *testing.T) {
	m := parseWinMem("26,3200,3200\r\n26,2933,3200\r\n")
	if m.Type != "ddr4" || m.Speed != 2933 || m.Modules != 2 {
		t.Fatalf("Windows 解析错误：%+v", m)
	}
	// 报告时钟频率（MHz）的机器：DDR5 2400 实为 4800 MT/s
	if m := parseWinMem("34,2400,4800\r\n34,2400,4800\r\n"); m.Speed != 4800 {
		t.Fatalf("应按传输率换算：%+v", m)
	}
	if m := parseWinMem("34,2400,0\r\n"); m.Speed != 4800 {
		t.Fatalf("低于 DDR5 最低速率时应乘 2：%+v", m)
	}
	if m := parseWinMem("26,2400,3200\r\n"); m.Speed != 2400 {
		t.Fatalf("DDR4 降频运行在 2400 是正常值，不应换算：%+v", m)
	}
	if m := parseWinMem("34,0,5600\n"); m.Type != "ddr5" || m.Speed != 5600 {
		t.Fatalf("配置频率为 0 时应取标称频率：%+v", m)
	}
	dmi := `Handle 0x0040, DMI type 17, 92 bytes
Memory Device
	Size: 16 GB
	Type: DDR5
	Speed: 5600 MT/s
	Configured Memory Speed: 5200 MT/s
Handle 0x0041, DMI type 17, 92 bytes
Memory Device
	Size: No Module Installed
	Type: Unknown
Handle 0x0042, DMI type 17, 92 bytes
Memory Device
	Size: 16 GB
	Type: DDR5
	Speed: 5600 MT/s
	Configured Memory Speed: 5200 MT/s
`
	if m := parseDMIMem(dmi); m.Type != "ddr5" || m.Speed != 5200 || m.Modules != 2 {
		t.Fatalf("dmidecode 解析错误：%+v", m)
	}
	if AppleBandwidth("Apple M2 Pro") != 200 || AppleBandwidth("Apple M4 Max") != 546 || AppleBandwidth("Apple M3") != 100 {
		t.Fatal("Apple 芯片带宽匹配错误")
	}
}

func TestRecommendSpeed(t *testing.T) {
	small := fakeArch(36, 110*miB, 0, 144*1024)
	// 纯 CPU：DDR4-3200 双通道 51.2 GB/s，读取约 4.5GB → 约 7 token/s
	r := RecommendHW(small, RecInput{RAM: 32 * giB, MemBW: 51.2, MemName: "DDR4-3200 双通道"})
	if r.TPS < 5 || r.TPS > 10 || r.TPSUpper {
		t.Fatalf("CPU 速度估算错误：%.2f", r.TPS)
	}
	// DDR5 更快
	r5 := RecommendHW(small, RecInput{RAM: 32 * giB, MemBW: 89.6})
	if r5.TPS <= r.TPS {
		t.Fatal("带宽更高速度应更快")
	}
	// 全部在独立显卡：内存不参与，不估算
	if r := RecommendHW(small, RecInput{RAM: 32 * giB, VRAM: 24 * giB, MemBW: 51.2}); r.TPS != 0 {
		t.Fatalf("独立显卡全放 GPU 不应按内存估算：%.2f", r.TPS)
	}
	// MoE 专家放 CPU：只读取激活的专家，速度远高于同体积稠密模型；结果为上限
	moe := fakeArch(48, 360*miB, 320*miB, 96*1024)
	r = RecommendHW(moe, RecInput{RAM: 64 * giB, VRAM: 12 * giB, MemBW: 51.2})
	if !r.TPSUpper || r.TPS < 20 {
		t.Fatalf("MoE 速度估算错误：%.2f upper=%v", r.TPS, r.TPSUpper)
	}
	// 70B 部分放 GPU + DDR4：速度很低，应提示
	big := fakeArch(80, 500*miB, 0, 320*1024)
	r = RecommendHW(big, RecInput{RAM: 64 * giB, VRAM: 24 * giB, MemBW: 51.2})
	if r.TPS >= 3 || !strings.Contains(strings.Join(r.Warnings, ""), "token/s") {
		t.Fatalf("低速应提示：%.2f %v", r.TPS, r.Warnings)
	}
	// 统一内存：按总读取量估算，不是上限
	r = RecommendHW(small, RecInput{RAM: 32 * giB, VRAM: 21 * giB, Unified: true, MemBW: 200})
	if r.Plan != "gpu" || r.TPS < 20 || r.TPSUpper {
		t.Fatalf("统一内存速度估算错误：%s %.2f", r.Plan, r.TPS)
	}
}
