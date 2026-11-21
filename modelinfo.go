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
	"strconv"
	"strings"
	"sync"
	"time"
)

type Candidate struct {
	RepoID    string `json:"repoId"`
	Downloads int64  `json:"downloads"`
	URL       string `json:"url"`
}

// Recommend 是从模型说明中提取的推荐参数。
type Recommend struct {
	Key     string  `json:"key"` // 对应 Params 的 JSON 字段名
	Value   float64 `json:"value"`
	Source  string  `json:"source"`
	Snippet string  `json:"snippet"`
}

type ModelInfo struct {
	Source     string      `json:"source"` // hf / ms
	Keyword    string      `json:"keyword"`
	RepoID     string      `json:"repoId"`
	URL        string      `json:"url"`
	SearchURL  string      `json:"searchUrl"`
	Readme     string      `json:"readme"`
	BaseModel  string      `json:"baseModel"`
	Recommends []Recommend `json:"recommends"`
	Candidates []Candidate `json:"candidates"`
	FetchedAt  time.Time   `json:"fetchedAt"`
}

// Keywords 由模型文件名/所在目录推导搜索关键字，按优先级排列。
func Keywords(g Group) []string {
	var ks []string
	seen := map[string]bool{}
	push := func(s string) {
		s = strings.Trim(s, "-_. ")
		if s != "" && !seen[strings.ToLower(s)] {
			seen[strings.ToLower(s)] = true
			ks = append(ks, s)
		}
	}
	name := BaseName(g.Variants[0].Name)
	push(name)
	if g.Dir != "" {
		push(BaseName(strings.TrimSuffix(strings.TrimSuffix(g.Dir, "-GGUF"), "-gguf")))
	}
	if i := strings.LastIndexAny(name, "-_"); i > 0 {
		push(name[:i])
	}
	return ks
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

	mu      sync.Mutex
	clients map[string]*http.Client
}

// netKind 表示访问的站点类别，用于选择代理。
type netKind int

const (
	netHF     netKind = iota // HuggingFace
	netMS                    // ModelScope 等其它站点
	netGitHub                // GitHub（发布信息与下载）
)

// proxyFor 返回访问该类站点使用的代理：GitHub 优先使用 GitHub 代理，未配置时使用通用代理；
// HuggingFace 使用通用代理；其它站点仅在“全部走代理”时使用。空表示使用系统环境变量代理。
func proxyFor(cfg Config, kind netKind) string {
	proxy := strings.TrimSpace(cfg.Proxy)
	switch kind {
	case netGitHub:
		if gp := strings.TrimSpace(cfg.GitHubProxy); gp != "" {
			return gp
		}
	case netMS:
		if !cfg.ProxyAll {
			return ""
		}
	}
	return proxy
}

// client 返回访问该类站点的 HTTP 客户端（按代理地址缓存）。
func (f *InfoFetcher) client(kind netKind) (*http.Client, error) {
	return f.clientWith(proxyFor(f.cfg.Get(), kind), 20*time.Second)
}

// clientWith 按代理地址与总超时返回 HTTP 客户端；timeout 为 0 表示不限总时长（用于大文件下载）。
func (f *InfoFetcher) clientWith(proxy string, timeout time.Duration) (*http.Client, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.clients == nil {
		f.clients = map[string]*http.Client{}
	}
	key := proxy + "|" + timeout.String()
	if c := f.clients[key]; c != nil {
		return c, nil
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.ResponseHeaderTimeout = 30 * time.Second
	if proxy != "" {
		u, err := url.Parse(proxy)
		if err != nil || u.Host == "" {
			return nil, fmt.Errorf("代理地址无效：%s", proxy)
		}
		tr.Proxy = http.ProxyURL(u)
	}
	c := &http.Client{Timeout: timeout, Transport: tr}
	f.clients[key] = c
	return c, nil
}

func (f *InfoFetcher) cachePath(id, src string) string {
	return filepath.Join(f.cacheDir, id+"_"+src+".json")
}

func (f *InfoFetcher) Fetch(g Group, src, repo string, refresh bool) (*ModelInfo, error) {
	if src != "ms" {
		src = "hf"
	}
	cp := f.cachePath(g.ID, src)
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
	// 优先搜索 GGUF 仓库，全部关键字都无结果时再放宽为普通搜索
search:
	for _, gguf := range []bool{true, false} {
		for _, kw := range Keywords(g) {
			mi.Keyword = kw
			if src == "ms" {
				mi.Candidates, err = f.searchMS(kw, gguf)
			} else {
				mi.Candidates, err = f.searchHF(kw, gguf)
			}
			if err != nil {
				return nil, err
			}
			if len(mi.Candidates) > 0 {
				break search
			}
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
		mi.URL = f.repoURL(src, repo)
		mi.Readme, err = f.repoFile(src, repo, "README.md")
		if err != nil {
			mi.Readme = "（获取 README 失败：" + err.Error() + "）"
		}
		mi.BaseModel = baseModel(mi.Readme)
		mi.Recommends = f.recommend(src, repo, mi.Readme, mi.BaseModel)
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

func (f *InfoFetcher) repoURL(src, repo string) string {
	if src == "ms" {
		return "https://www.modelscope.cn/models/" + repo
	}
	return f.hfBase() + "/" + repo
}

func (f *InfoFetcher) repoFile(src, repo, file string) (string, error) {
	if src == "ms" {
		return f.get(netMS, "https://www.modelscope.cn/api/v1/models/"+repo+"/repo?Revision=master&FilePath="+url.QueryEscape(file))
	}
	return f.get(netHF, f.hfBase()+"/"+repo+"/raw/main/"+file)
}

func (f *InfoFetcher) get(kind netKind, u string) (string, error) {
	c, err := f.client(kind)
	if err != nil {
		return "", err
	}
	resp, err := c.Get(u)
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

func (f *InfoFetcher) searchHF(kw string, gguf bool) ([]Candidate, error) {
	u := f.hfBase() + "/api/models?limit=20&sort=downloads&search=" + url.QueryEscape(kw)
	if gguf {
		u += "&filter=gguf"
	}
	body, err := f.get(netHF, u)
	if err != nil {
		return nil, fmt.Errorf("HuggingFace 搜索失败：%v（可在设置中配置代理或镜像地址）", err)
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

func (f *InfoFetcher) searchMS(kw string, gguf bool) ([]Candidate, error) {
	c, err := f.client(netMS)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"PageSize": 20, "PageNumber": 1, "SortBy": "Default", "Target": "", "SingleCriterion": []any{}, "Name": kw,
	}
	if gguf {
		body["Criterion"] = []any{map[string]any{"category": "libraries", "predicate": "contains", "values": []string{"gguf"}}}
	}
	payload, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPut, "https://www.modelscope.cn/api/v1/dolphin/models", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.Do(req)
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

// ---------- 推荐参数 ----------

var baseModelRe = regexp.MustCompile(`(?m)^base_model:\s*(?:\n\s*-\s*)?["']?([\w.\-]+/[\w.\-]+)`)

// baseModel 从 README 的 YAML 头中读取 base_model。
func baseModel(readme string) string {
	if !strings.HasPrefix(readme, "---") {
		return ""
	}
	end := strings.Index(readme[3:], "\n---")
	if end < 0 {
		return ""
	}
	if m := baseModelRe.FindStringSubmatch(readme[:end+3]); m != nil {
		return m[1]
	}
	return ""
}

type recRule struct {
	key      string
	re       *regexp.Regexp
	min, max float64
}

const recSep = `(?:\s*(?:of|to|is|=|:)\s*|\s+|["'` + "`" + `*:=]+\s*)["'` + "`" + `]?`

var recRules = []recRule{
	{"temp", regexp.MustCompile(`(?i)(?:--temp(?:erature)?\b|\btemperature\b)` + recSep + `(\d+(?:\.\d+)?)`), 0, 2},
	{"topP", regexp.MustCompile(`(?i)(?:--top-p\b|\btop[_ -]?p\b)` + recSep + `(\d+(?:\.\d+)?)`), 0, 1},
	{"topK", regexp.MustCompile(`(?i)(?:--top-k\b|\btop[_ -]?k\b)` + recSep + `(\d+)\b`), 0, 1000},
	{"minP", regexp.MustCompile(`(?i)(?:--min-p\b|\bmin[_ -]?p\b)` + recSep + `(\d+(?:\.\d+)?)`), 0, 1},
	{"presencePenalty", regexp.MustCompile(`(?i)(?:--presence-penalty\b|\bpresence[_ ]penalty\b)` + recSep + `(\d+(?:\.\d+)?)`), 0, 3},
	{"repeatPenalty", regexp.MustCompile(`(?i)(?:--repeat-penalty\b|\brepe(?:at|tition)[_ ]penalty\b)` + recSep + `(\d+(?:\.\d+)?)`), 0.5, 3},
	{"ctxSize", regexp.MustCompile(`(?:--ctx-size|(?:^|\s)-c)[\s=]+(\d{3,7})\b`), 512, 4 << 20},
}

// extractRecs 在文本中查找每个参数第一次出现的推荐值。
func extractRecs(text, source string, have map[string]bool) []Recommend {
	var out []Recommend
	for _, r := range recRules {
		if have[r.key] {
			continue
		}
		for _, m := range r.re.FindAllStringSubmatchIndex(text, 20) {
			v, err := strconv.ParseFloat(text[m[2]:m[3]], 64)
			if err != nil || v < r.min || v > r.max {
				continue
			}
			a, b := max(m[0]-60, 0), min(m[1]+40, len(text))
			snip := strings.Join(strings.Fields(text[a:b]), " ")
			out = append(out, Recommend{Key: r.key, Value: v, Source: source, Snippet: snip})
			have[r.key] = true
			break
		}
	}
	return out
}

// recommend 按优先级提取推荐参数：仓库 README -> 基础模型 README -> generation_config.json。
func (f *InfoFetcher) recommend(src, repo, readme, base string) []Recommend {
	have := map[string]bool{}
	out := extractRecs(stripFrontMatter(readme), repo+" README", have)
	hasSampling := func() bool { return have["temp"] || have["topP"] || have["topK"] }
	if !hasSampling() && base != "" {
		if rd, err := f.repoFile(src, base, "README.md"); err == nil {
			out = append(out, extractRecs(stripFrontMatter(rd), base+" README", have)...)
		}
	}
	for _, r := range []string{repo, base} {
		if r == "" {
			continue
		}
		body, err := f.repoFile(src, r, "generation_config.json")
		if err != nil {
			continue
		}
		var gc map[string]any
		if json.Unmarshal([]byte(body), &gc) != nil {
			continue
		}
		for k, key := range map[string]string{"temperature": "temp", "top_p": "topP", "top_k": "topK", "min_p": "minP", "repetition_penalty": "repeatPenalty"} {
			if v, ok := gc[k].(float64); ok && !have[key] {
				out = append(out, Recommend{Key: key, Value: v, Source: r + " generation_config.json", Snippet: fmt.Sprintf(`"%s": %v`, k, v)})
				have[key] = true
			}
		}
	}
	order := map[string]int{}
	for i, r := range recRules {
		order[r.key] = i
	}
	sort.SliceStable(out, func(i, j int) bool { return order[out[i].Key] < order[out[j].Key] })
	return out
}

func stripFrontMatter(s string) string {
	if strings.HasPrefix(s, "---") {
		if i := strings.Index(s[3:], "\n---"); i >= 0 {
			return s[i+7:]
		}
	}
	return s
}
