package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

type Candidate struct {
	RepoID    string `json:"repoId"`
	Downloads int64  `json:"downloads"`
	URL       string `json:"url"`
}

type ModelInfo struct {
	Source     string      `json:"source"` // hf / ms
	Keyword    string      `json:"keyword"`
	RepoID     string      `json:"repoId"`
	URL        string      `json:"url"`
	SearchURL  string      `json:"searchUrl"`
	Readme     string      `json:"readme"`
	Candidates []Candidate `json:"candidates"`
	FetchedAt  time.Time   `json:"fetchedAt"`
}

var (
	quantRe = regexp.MustCompile(`(?i)[-_.](ud-)?(i?q\d+(_[a-z0-9]+)*|f16|f32|bf16|fp16|fp8|mxfp4)$`)
	httpc   = &http.Client{Timeout: 20 * time.Second}
)

// Keywords 由模型文件名/所在目录推导搜索关键字，按优先级排列。
func Keywords(m Model) []string {
	var ks []string
	seen := map[string]bool{}
	push := func(s string) {
		s = strings.Trim(s, "-_. ")
		if s != "" && !seen[strings.ToLower(s)] {
			seen[strings.ToLower(s)] = true
			ks = append(ks, s)
		}
	}
	name := m.Name
	if mm := splitRe.FindStringSubmatch(name + ".gguf"); mm != nil {
		name = name[:len(name)-len(mm[0])+len(".gguf")]
	}
	for quantRe.MatchString(name) {
		name = quantRe.ReplaceAllString(name, "")
	}
	push(name)
	if i := strings.LastIndexAny(name, "-_"); i > 0 {
		push(name[:i])
	}
	// 子目录名常为仓库名，作为兜底关键字
	if dir := filepath.Dir(m.Path); !samePath(dir, m.Root) {
		push(filepath.Base(dir))
	}
	return ks
}

func samePath(a, b string) bool {
	return filepath.Clean(a) == filepath.Clean(b)
}

func score(repo string, kw string, downloads int64) float64 {
	r := strings.ToLower(repo)
	name := r[strings.LastIndex(r, "/")+1:]
	k := strings.ToLower(kw)
	s := 0.0
	switch {
	case name == k || name == k+"-gguf":
		s += 100
	case strings.Contains(name, k):
		s += 50
	}
	// 官方仓库（组织名是关键字前缀，如 Qwen/Qwen2.5-...）优先
	if owner := r[:max(strings.Index(r, "/"), 0)]; len(owner) >= 3 && strings.HasPrefix(k, owner) {
		s += 20
	}
	if strings.Contains(name, "gguf") {
		s += 10
	}
	d := float64(downloads)
	for d >= 10 {
		d /= 10
		s += 1
	}
	return s
}

type InfoFetcher struct {
	cacheDir string
	cfg      *ConfigStore
}

func (f *InfoFetcher) cachePath(id, src string) string {
	return filepath.Join(f.cacheDir, id+"_"+src+".json")
}

func (f *InfoFetcher) Fetch(m Model, src, repo string, refresh bool) (*ModelInfo, error) {
	if src != "ms" {
		src = "hf"
	}
	cp := f.cachePath(m.ID, src)
	if !refresh && repo == "" {
		if b, err := os.ReadFile(cp); err == nil {
			var mi ModelInfo
			if json.Unmarshal(b, &mi) == nil {
				return &mi, nil
			}
		}
	}
	mi := &ModelInfo{Source: src, FetchedAt: time.Now()}
	var err error
	for _, kw := range Keywords(m) {
		mi.Keyword = kw
		if src == "ms" {
			mi.Candidates, err = f.searchMS(kw)
		} else {
			mi.Candidates, err = f.searchHF(kw)
		}
		if err != nil {
			return nil, err
		}
		if len(mi.Candidates) > 0 {
			break
		}
	}
	if src == "ms" {
		mi.SearchURL = "https://www.modelscope.cn/models?name=" + url.QueryEscape(mi.Keyword)
	} else {
		mi.SearchURL = f.hfBase() + "/models?search=" + url.QueryEscape(mi.Keyword)
	}
	kw := mi.Keyword
	sort.SliceStable(mi.Candidates, func(i, j int) bool {
		a, b := mi.Candidates[i], mi.Candidates[j]
		return score(a.RepoID, kw, a.Downloads) > score(b.RepoID, kw, b.Downloads)
	})
	if repo == "" && len(mi.Candidates) > 0 {
		repo = mi.Candidates[0].RepoID
	}
	if repo != "" {
		mi.RepoID = repo
		if src == "ms" {
			mi.URL = "https://www.modelscope.cn/models/" + repo
			mi.Readme, err = f.get("https://www.modelscope.cn/api/v1/models/" + repo + "/repo?Revision=master&FilePath=README.md")
		} else {
			mi.URL = f.hfBase() + "/" + repo
			mi.Readme, err = f.get(f.hfBase() + "/" + repo + "/raw/main/README.md")
		}
		if err != nil {
			mi.Readme = "（获取 README 失败：" + err.Error() + "）"
		}
	}
	if b, err := json.Marshal(mi); err == nil {
		_ = os.MkdirAll(f.cacheDir, 0o755)
		_ = os.WriteFile(cp, b, 0o644)
	}
	return mi, nil
}

func (f *InfoFetcher) hfBase() string {
	return strings.TrimRight(f.cfg.Get().HFEndpoint, "/")
}

func (f *InfoFetcher) get(u string) (string, error) {
	resp, err := httpc.Get(u)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return string(b), nil
}

func (f *InfoFetcher) searchHF(kw string) ([]Candidate, error) {
	u := f.hfBase() + "/api/models?limit=20&sort=downloads&search=" + url.QueryEscape(kw)
	body, err := f.get(u)
	if err != nil {
		return nil, fmt.Errorf("HuggingFace 搜索失败：%v", err)
	}
	var list []struct {
		ID        string `json:"id"`
		Downloads int64  `json:"downloads"`
	}
	if err := json.Unmarshal([]byte(body), &list); err != nil {
		return nil, err
	}
	out := make([]Candidate, 0, len(list))
	for _, x := range list {
		out = append(out, Candidate{RepoID: x.ID, Downloads: x.Downloads, URL: f.hfBase() + "/" + x.ID})
	}
	return out, nil
}

func (f *InfoFetcher) searchMS(kw string) ([]Candidate, error) {
	payload, _ := json.Marshal(map[string]any{
		"PageSize": 20, "PageNumber": 1, "SortBy": "Default", "Target": "", "SingleCriterion": []any{}, "Name": kw,
	})
	req, _ := http.NewRequest(http.MethodPut, "https://www.modelscope.cn/api/v1/dolphin/models", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ModelScope 搜索失败：%v", err)
	}
	defer resp.Body.Close()
	var r struct {
		Data struct {
			Model struct {
				Models []struct {
					Path      string `json:"Path"`
					Name      string `json:"Name"`
					Downloads int64  `json:"Downloads"`
				} `json:"Models"`
			} `json:"Model"`
		} `json:"Data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&r); err != nil {
		return nil, fmt.Errorf("ModelScope 响应解析失败：%v", err)
	}
	var out []Candidate
	for _, x := range r.Data.Model.Models {
		id := x.Path + "/" + x.Name
		out = append(out, Candidate{RepoID: id, Downloads: x.Downloads, URL: "https://www.modelscope.cn/models/" + id})
	}
	return out, nil
}
