package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// GGUF 文件头解析：只读取元数据与张量信息（不读取权重），用于估算显存/内存占用。
// 格式见 https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

// GGUFTensor 是一个张量的名称与在文件中占用的字节数。
type GGUFTensor struct {
	Name  string
	Bytes int64
}

// GGUFFile 是一个 gguf 文件（或分片）的头信息。
type GGUFFile struct {
	Meta    map[string]any // 标量值；数组只记录长度（键 + ".len"），整数数组额外保存在 Ints 中
	Ints    map[string][]int64
	Tensors []GGUFTensor
}

const (
	ggufU8 = iota
	ggufI8
	ggufU16
	ggufI16
	ggufU32
	ggufI32
	ggufF32
	ggufBool
	ggufString
	ggufArray
	ggufU64
	ggufI64
	ggufF64
)

type ggufReader struct {
	r   *bufio.Reader
	pos int64
	err error
}

func (g *ggufReader) read(n int) []byte {
	if g.err != nil {
		return make([]byte, n)
	}
	b := make([]byte, n)
	_, g.err = io.ReadFull(g.r, b)
	g.pos += int64(n)
	return b
}

func (g *ggufReader) skip(n int64) {
	if g.err != nil {
		return
	}
	for n > 0 {
		k := n
		if k > 1<<30 {
			k = 1 << 30
		}
		d, err := g.r.Discard(int(k))
		g.pos += int64(d)
		n -= int64(d)
		if err != nil {
			g.err = err
			return
		}
	}
}

func (g *ggufReader) u32() uint32 { return binary.LittleEndian.Uint32(g.read(4)) }
func (g *ggufReader) u64() uint64 { return binary.LittleEndian.Uint64(g.read(8)) }

func (g *ggufReader) str() string {
	n := g.u64()
	if n > 64<<20 {
		g.err = fmt.Errorf("字符串长度异常：%d", n)
		return ""
	}
	return string(g.read(int(n)))
}

// skipStr 跳过字符串（词表数组中的大量字符串不需要保留）。
func (g *ggufReader) skipStr() {
	n := g.u64()
	if n > 64<<20 {
		g.err = fmt.Errorf("字符串长度异常：%d", n)
		return
	}
	g.skip(int64(n))
}

var ggufSize = map[uint32]int{ggufU8: 1, ggufI8: 1, ggufU16: 2, ggufI16: 2, ggufU32: 4, ggufI32: 4, ggufF32: 4, ggufBool: 1, ggufU64: 8, ggufI64: 8, ggufF64: 8}

// scalar 读取一个标量值，整数统一为 int64，浮点为 float64。
func (g *ggufReader) scalar(t uint32) any {
	switch t {
	case ggufU8:
		return int64(g.read(1)[0])
	case ggufI8:
		return int64(int8(g.read(1)[0]))
	case ggufU16:
		return int64(binary.LittleEndian.Uint16(g.read(2)))
	case ggufI16:
		return int64(int16(binary.LittleEndian.Uint16(g.read(2))))
	case ggufU32:
		return int64(g.u32())
	case ggufI32:
		return int64(int32(g.u32()))
	case ggufU64:
		return int64(g.u64())
	case ggufI64:
		return int64(g.u64())
	case ggufF32:
		return float64(math.Float32frombits(g.u32()))
	case ggufF64:
		return math.Float64frombits(g.u64())
	case ggufBool:
		return g.read(1)[0] != 0
	case ggufString:
		return g.str()
	}
	g.err = fmt.Errorf("未知的元数据类型 %d", t)
	return nil
}

// ReadGGUF 读取 gguf 文件头。
func ReadGGUF(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	g := &ggufReader{r: bufio.NewReaderSize(f, 1<<20)}
	if string(g.read(4)) != "GGUF" {
		return nil, errors.New("不是 GGUF 文件")
	}
	ver := g.u32()
	if g.err == nil && ver < 2 {
		return nil, fmt.Errorf("不支持的 GGUF 版本 %d", ver)
	}
	nTensor, nKV := g.u64(), g.u64()
	if nTensor > 1<<20 || nKV > 1<<20 {
		return nil, errors.New("GGUF 文件头异常")
	}
	out := &GGUFFile{Meta: map[string]any{}, Ints: map[string][]int64{}}
	for i := uint64(0); i < nKV && g.err == nil; i++ {
		key := g.str()
		t := g.u32()
		if t != ggufArray {
			out.Meta[key] = g.scalar(t)
			continue
		}
		et, n := g.u32(), g.u64()
		if n > 1<<28 {
			return nil, errors.New("GGUF 数组长度异常")
		}
		out.Meta[key+".len"] = int64(n)
		switch {
		case et == ggufString:
			for j := uint64(0); j < n && g.err == nil; j++ {
				g.skipStr()
			}
		case ggufSize[et] > 0 && et != ggufF32 && et != ggufF64 && et != ggufBool && n <= 4096:
			vs := make([]int64, n)
			for j := range vs {
				vs[j], _ = g.scalar(et).(int64)
			}
			out.Ints[key] = vs
		case ggufSize[et] > 0:
			g.skip(int64(n) * int64(ggufSize[et]))
		default:
			return nil, fmt.Errorf("不支持的数组元素类型 %d", et)
		}
	}
	type tinfo struct {
		name string
		off  uint64
	}
	ts := make([]tinfo, 0, nTensor)
	for i := uint64(0); i < nTensor && g.err == nil; i++ {
		name := g.str()
		nd := g.u32()
		if nd > 8 {
			return nil, errors.New("GGUF 张量维度异常")
		}
		g.skip(int64(nd) * 8)
		g.u32() // 类型
		ts = append(ts, tinfo{name, g.u64()})
	}
	if g.err != nil {
		return nil, fmt.Errorf("读取 GGUF 文件头失败：%v", g.err)
	}
	align := int64(32)
	if a, ok := out.Meta["general.alignment"].(int64); ok && a > 0 {
		align = a
	}
	dataStart := (g.pos + align - 1) / align * align
	dataLen := st.Size() - dataStart
	// 张量按偏移排序，相邻偏移之差即占用字节（含对齐填充），无需按量化类型计算
	sort.Slice(ts, func(i, j int) bool { return ts[i].off < ts[j].off })
	for i, t := range ts {
		end := dataLen
		if i+1 < len(ts) {
			end = int64(ts[i+1].off)
		}
		out.Tensors = append(out.Tensors, GGUFTensor{Name: t.name, Bytes: max(0, end-int64(t.off))})
	}
	return out, nil
}

// shardPaths 返回分片模型的全部文件（Model.Path 为第一个分片）。
func shardPaths(m Model) []string {
	mm := splitRe.FindStringSubmatchIndex(m.Path)
	if m.Parts <= 1 || mm == nil {
		return []string{m.Path}
	}
	total := m.Path[mm[4]:mm[5]]
	n, _ := strconv.Atoi(total)
	out := make([]string, 0, n)
	for i := 1; i <= n; i++ {
		out = append(out, m.Path[:mm[2]]+fmt.Sprintf("%05d", i)+m.Path[mm[3]:])
	}
	return out
}

// ModelArch 是估算占用所需的模型结构信息。
type ModelArch struct {
	Arch      string `json:"arch"`
	Name      string `json:"name"`
	Layers    int    `json:"layers"`
	CtxTrain  int    `json:"ctxTrain"`
	Embd      int    `json:"embd"`
	Vocab     int    `json:"vocab"`
	Experts   int    `json:"experts"`     // 0 表示非 MoE
	ExpertsOn int    `json:"expertsUsed"` // 每个 token 激活的专家数
	SWA       int    `json:"swa"`         // 滑动窗口大小（0 表示无）
	KVPerTok  int64  `json:"kvPerToken"`  // 每个 token 的 KV 缓存字节（f16，全部层）
	KVLayers  int    `json:"kvLayers"`    // 需要 KV 缓存的层数
	MLA       bool   `json:"mla"`

	Total     int64   `json:"total"`     // 全部权重字节
	TokEmbd   int64   `json:"tokEmbd"`   // 输入词嵌入（始终在 CPU）
	Output    int64   `json:"output"`    // 输出层（全部层放 GPU 时随之放到 GPU）
	LayerSize []int64 `json:"-"`         // 每层权重字节（含专家）
	LayerExps []int64 `json:"-"`         // 每层专家权重字节
	ExpsTotal int64   `json:"expsTotal"` // 全部专家权重
	Tied      bool    `json:"tied"`      // 输出层与词嵌入共用（无 output.weight）
}

var (
	blkRe = regexp.MustCompile(`^blk\.(\d+)\.`)
	// 与 llama.cpp -cmoe/-ncmoe 的匹配规则一致：路由专家的 up/down/gate 权重
	expsRe = regexp.MustCompile(`\.ffn_(up|down|gate|gate_up)_(ch|)exps`)
)

// metaInt 读取 "<arch>.<key>"，值为数组时取最大值，perLayer 返回逐层值。
func metaInt(f *GGUFFile, key string) (v int64, perLayer []int64) {
	if x, ok := f.Meta[key].(int64); ok {
		return x, nil
	}
	if a := f.Ints[key]; len(a) > 0 {
		for _, x := range a {
			v = max(v, x)
		}
		return v, a
	}
	return 0, nil
}

// LoadModelArch 读取模型（含全部分片）的结构信息。
func LoadModelArch(m Model) (*ModelArch, error) {
	var files []*GGUFFile
	for _, p := range shardPaths(m) {
		f, err := ReadGGUF(p)
		if err != nil {
			return nil, fmt.Errorf("%s：%v", filepath.Base(p), err)
		}
		files = append(files, f)
	}
	h := files[0]
	a := &ModelArch{}
	a.Arch, _ = h.Meta["general.architecture"].(string)
	a.Name, _ = h.Meta["general.name"].(string)
	p := a.Arch + "."
	get := func(k string) int64 { v, _ := metaInt(h, p+k); return v }
	a.Layers = int(get("block_count"))
	a.CtxTrain = int(get("context_length"))
	a.Embd = int(get("embedding_length"))
	a.Experts = int(get("expert_count"))
	a.ExpertsOn = int(get("expert_used_count"))
	a.SWA = int(get("attention.sliding_window"))
	a.Vocab = int(get("vocab_size"))
	if a.Vocab == 0 {
		if n, ok := h.Meta["tokenizer.ggml.tokens.len"].(int64); ok {
			a.Vocab = int(n)
		}
	}
	if a.Layers <= 0 {
		return nil, fmt.Errorf("无法识别模型结构（架构 %q 缺少 block_count）", a.Arch)
	}

	// KV 缓存：每层 head_count_kv × (key_length + value_length) 个 f16。
	// head_count_kv 可能是逐层数组（为 0 的层是循环/线性注意力层，不需要 KV）。
	nHead, _ := metaInt(h, p+"attention.head_count")
	nKV, kvLayers := metaInt(h, p+"attention.head_count_kv")
	if nKV == 0 && kvLayers == nil {
		nKV = nHead
	}
	kLen, vLen := get("attention.key_length"), get("attention.value_length")
	if kLen == 0 && nHead > 0 {
		kLen = int64(a.Embd) / nHead
	}
	if vLen == 0 {
		vLen = kLen
	}
	if rank := get("attention.kv_lora_rank"); rank > 0 {
		// MLA（DeepSeek 等）：缓存压缩后的潜在向量与 RoPE 部分
		a.MLA = true
		a.KVPerTok = int64(a.Layers) * (rank + get("rope.dimension_count")) * 2
		a.KVLayers = a.Layers
	} else if kvLayers != nil {
		for _, n := range kvLayers {
			if n > 0 {
				a.KVPerTok += n * (kLen + vLen) * 2
				a.KVLayers++
			}
		}
	} else {
		a.KVPerTok = int64(a.Layers) * nKV * (kLen + vLen) * 2
		a.KVLayers = a.Layers
	}

	a.LayerSize = make([]int64, a.Layers)
	a.LayerExps = make([]int64, a.Layers)
	hasOutput := false
	for _, f := range files {
		for _, t := range f.Tensors {
			a.Total += t.Bytes
			if mm := blkRe.FindStringSubmatch(t.Name); mm != nil {
				i, _ := strconv.Atoi(mm[1])
				if i < a.Layers {
					a.LayerSize[i] += t.Bytes
					if expsRe.MatchString(t.Name) {
						a.LayerExps[i] += t.Bytes
						a.ExpsTotal += t.Bytes
					}
				}
				continue
			}
			switch {
			case strings.HasPrefix(t.Name, "token_embd."):
				a.TokEmbd += t.Bytes
			case strings.HasPrefix(t.Name, "output"):
				a.Output += t.Bytes
				if strings.HasPrefix(t.Name, "output.weight") {
					hasOutput = true
				}
			}
		}
	}
	if !hasOutput {
		// 共用词嵌入时，输出层会在 GPU 上另存一份
		a.Tied = true
		a.Output += a.TokEmbd
	}
	return a, nil
}
