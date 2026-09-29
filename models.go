package main

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

type Model struct {
	ID      string   `json:"id"`
	Name    string   `json:"name"`    // 文件名（不含 .gguf）
	File    string   `json:"file"`    // 相对模型目录的路径
	Path    string   `json:"path"`    // 绝对路径
	Root    string   `json:"root"`    // 所属模型目录
	Size    int64    `json:"size"`    // 字节（分片模型为总大小）
	Parts   int      `json:"parts"`   // 分片数
	Quant   string   `json:"quant"`   // 量化标识
	Mmprojs []string `json:"mmprojs"` // 同目录下的多模态投影文件
}

var splitRe = regexp.MustCompile(`(?i)-(\d{5})-of-(\d{5})\.gguf$`)

func modelID(path string) string {
	h := sha1.Sum([]byte(path))
	return hex.EncodeToString(h[:6])
}

// DirStatus 是一个模型目录的扫描结果，用于在设置页提示网络目录不可用等问题。
type DirStatus struct {
	Dir   string `json:"dir"`  // 配置中填写的目录
	Path  string `json:"path"` // 实际访问的路径（网络路径解析后）
	Net   bool   `json:"net"`  // 是否网络目录
	Count int    `json:"count"`
	Error string `json:"error,omitempty"`
}

// 单个目录的扫描时限；网络共享掉线时 ReadDir 可能长时间阻塞。
const scanTimeout = 10 * time.Second

// scanCall 是一次进行中的目录扫描。同一目录的并发请求共享结果；
// 网络共享掉线时读取可能一直阻塞，超时后的请求直接返回超时，不再重复发起。
type scanCall struct {
	start  time.Time
	done   chan struct{}
	models []Model
	err    error
}

var (
	scanMu    sync.Mutex
	scanCalls = map[string]*scanCall{}
)

// ScanModels 扫描模型目录：支持 目录/模型.gguf 与 目录/子目录/模型.gguf 两级结构。
// 各目录并行扫描并限时，网络目录无响应时跳过并在状态中说明。
func ScanModels(dirs []string) ([]Model, []DirStatus) {
	type result struct {
		models []Model
		st     DirStatus
	}
	res := make([]result, len(dirs))
	var wg sync.WaitGroup
	for i, root := range dirs {
		root = strings.TrimSpace(root)
		if root == "" {
			continue
		}
		wg.Add(1)
		go func(i int, root string) {
			defer wg.Done()
			st := DirStatus{Dir: root, Net: IsNetPath(root)}
			ms, err := scanRoot(root, &st)
			if err != nil {
				st.Error = err.Error()
			}
			st.Count = len(ms)
			res[i] = result{ms, st}
		}(i, root)
	}
	wg.Wait()
	var out []Model
	var sts []DirStatus
	seen := map[string]bool{}
	for _, r := range res {
		if r.st.Dir == "" {
			continue
		}
		sts = append(sts, r.st)
		for _, m := range r.models {
			if !seen[m.Path] {
				seen[m.Path] = true
				out = append(out, m)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return strings.ToLower(out[i].File) < strings.ToLower(out[j].File) })
	return out, sts
}

func scanRoot(root string, st *DirStatus) ([]Model, error) {
	abs, err := ResolveDir(root)
	if err != nil {
		return nil, err
	}
	st.Path = abs
	if !st.Net {
		st.Net = IsNetPath(abs)
	}
	scanMu.Lock()
	c := scanCalls[abs]
	if c == nil {
		c = &scanCall{start: time.Now(), done: make(chan struct{})}
		scanCalls[abs] = c
		go func() {
			c.models, c.err = scanTree(abs)
			scanMu.Lock()
			delete(scanCalls, abs)
			scanMu.Unlock()
			close(c.done)
		}()
	}
	scanMu.Unlock()
	wait := scanTimeout - time.Since(c.start)
	if wait <= 0 {
		return nil, errTimeout
	}
	select {
	case <-c.done:
		return c.models, c.err
	case <-time.After(wait):
		return nil, errTimeout
	}
}

func scanTree(abs string) ([]Model, error) {
	entries, err := os.ReadDir(abs)
	if err != nil {
		return nil, fmt.Errorf("无法读取目录：%v", err)
	}
	var out []Model
	seen := map[string]bool{}
	scanDir(abs, abs, &out, seen)
	for _, e := range entries {
		if e.IsDir() {
			scanDir(abs, filepath.Join(abs, e.Name()), &out, seen)
		}
	}
	return out, nil
}

func scanDir(root, dir string, out *[]Model, seen map[string]bool) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	var mmprojs []string
	type item struct {
		path  string
		size  int64
		parts int
	}
	var items []*item
	splits := map[string]*item{}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.EqualFold(filepath.Ext(name), ".gguf") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		p := filepath.Join(dir, name)
		if strings.Contains(strings.ToLower(name), "mmproj") {
			mmprojs = append(mmprojs, p)
			continue
		}
		if m := splitRe.FindStringSubmatch(name); m != nil {
			key := name[:len(name)-len(m[0])]
			it := splits[key]
			if it == nil {
				it = &item{}
				splits[key] = it
				items = append(items, it)
			}
			if m[1] == "00001" {
				it.path = p
			}
			it.size += info.Size()
			it.parts++
			continue
		}
		items = append(items, &item{path: p, size: info.Size(), parts: 1})
	}
	for _, it := range items {
		if it.path == "" || seen[it.path] {
			continue
		}
		seen[it.path] = true
		rel, _ := filepath.Rel(root, it.path)
		base := filepath.Base(it.path)
		name := strings.TrimSuffix(base, filepath.Ext(base))
		*out = append(*out, Model{
			ID: modelID(it.path), Name: name, File: filepath.ToSlash(rel), Path: it.path,
			Root: root, Size: it.size, Parts: it.parts, Quant: Quant(name), Mmprojs: append([]string{}, mmprojs...),
		})
	}
}

// Group 是一组相关模型：同一子目录下的所有模型，或模型目录根下去掉量化后缀后同名的模型。
type Group struct {
	ID       string   `json:"id"`
	Key      string   `json:"key"`
	Name     string   `json:"name"`
	Root     string   `json:"root"`
	Dir      string   `json:"dir"` // 子目录名，根目录下的模型为空
	Variants []Model  `json:"variants"`
	Mmprojs  []string `json:"mmprojs"`
}

var quantRe = regexp.MustCompile(`(?i)[-_.](ud-)?(i?q\d+(_[a-z0-9]+)*|f16|f32|bf16|fp16|fp8|mxfp4(_moe)?)$`)

// BaseName 去掉分片与量化后缀，如 Qwen3-8B-Q4_K_M -> Qwen3-8B。
func BaseName(name string) string {
	if mm := splitRe.FindStringSubmatch(name + ".gguf"); mm != nil {
		name = name[:len(name)-len(mm[0])+len(".gguf")]
	}
	for quantRe.MatchString(name) {
		name = quantRe.ReplaceAllString(name, "")
	}
	return name
}

// Quant 返回文件名中的量化标识（可能为空）。
func Quant(name string) string {
	b := BaseName(name)
	q := strings.Trim(name[len(b):], "-_.")
	if mm := splitRe.FindStringSubmatch(q + ".gguf"); mm != nil {
		q = q[:len(q)-len(mm[0])+len(".gguf")]
	}
	return q
}

func GroupModels(models []Model) []Group {
	idx := map[string]int{}
	var out []Group
	for _, m := range models {
		dir := filepath.Dir(m.Path)
		var key, name, sub string
		if samePath(dir, m.Root) {
			name = BaseName(m.Name)
			key = m.Root + "|" + strings.ToLower(name)
		} else {
			sub = filepath.Base(dir)
			name, key = sub, dir
		}
		i, ok := idx[key]
		if !ok {
			i = len(out)
			idx[key] = i
			out = append(out, Group{ID: modelID(key), Key: key, Name: name, Root: m.Root, Dir: sub, Mmprojs: m.Mmprojs})
		}
		out[i].Variants = append(out[i].Variants, m)
	}
	for i := range out {
		g := &out[i]
		// 单个模型时直接显示完整文件名
		if len(g.Variants) == 1 {
			g.Name = g.Variants[0].Name
		}
		sort.Slice(g.Variants, func(a, b int) bool { return g.Variants[a].Size < g.Variants[b].Size })
	}
	sort.Slice(out, func(a, b int) bool { return strings.ToLower(out[a].Name) < strings.ToLower(out[b].Name) })
	return out
}

func samePath(a, b string) bool {
	return filepath.Clean(a) == filepath.Clean(b)
}
