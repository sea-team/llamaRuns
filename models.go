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
			Root: root, Size: it.size, Parts: it.parts, Mmprojs: append([]string{}, mmprojs...),
		})
	}
}
