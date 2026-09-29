package main

import (
	"crypto/sha1"
	"encoding/hex"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
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

// ScanModels 扫描模型目录：支持 目录/模型.gguf 与 目录/子目录/模型.gguf 两级结构。
func ScanModels(dirs []string) []Model {
	var out []Model
	seen := map[string]bool{}
	for _, root := range dirs {
		root = strings.TrimSpace(root)
		if root == "" {
			continue
		}
		abs, err := filepath.Abs(root)
		if err != nil {
			continue
		}
		scanDir(abs, abs, &out, seen)
		entries, err := os.ReadDir(abs)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() {
				scanDir(abs, filepath.Join(abs, e.Name()), &out, seen)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return strings.ToLower(out[i].File) < strings.ToLower(out[j].File) })
	return out
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
