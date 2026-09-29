package main

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed web
var webFS embed.FS

type App struct {
	cfg  *ConfigStore
	mgr  *Manager
	mon  *SysMonitor
	info *InfoFetcher

	siMu  sync.Mutex
	probe map[string]ServerInfo // 配置路径 -> 探测结果
}

func (a *App) probeCached(path string, refresh bool) ServerInfo {
	a.siMu.Lock()
	defer a.siMu.Unlock()
	if a.probe == nil {
		a.probe = map[string]ServerInfo{}
	}
	si, ok := a.probe[path]
	if refresh || !ok {
		si = ProbeServer(path)
		a.probe[path] = si
	}
	return si
}

type ServersInfo struct {
	GPU   ServerInfo  `json:"gpu"`
	CPU   *ServerInfo `json:"cpu"` // 未配置 CPU 版时为 nil
	GPUOK bool        `json:"gpuOk"`
}

func (a *App) servers(refresh bool) ServersInfo {
	cfg := a.cfg.Get()
	var s ServersInfo
	s.GPU = a.probeCached(cfg.ServerPathGPU, refresh)
	if strings.TrimSpace(cfg.ServerPathCPU) != "" {
		c := a.probeCached(cfg.ServerPathCPU, refresh)
		s.CPU = &c
	}
	s.GPUOK = s.GPU.GPU
	return s
}

func (a *App) groups() []Group {
	return GroupModels(ScanModels(a.cfg.Get().ModelDirs))
}

func (a *App) findGroup(id string) (Group, bool) {
	for _, g := range a.groups() {
		if g.ID == id {
			return g, true
		}
	}
	return Group{}, false
}

// runReq 是启动/生成命令的请求；未指定的项使用上次启动的选择。
type runReq struct {
	Variant string  `json:"variant"`
	Mode    *string `json:"mode"` // cpu / gpu / 空（按参数配置）
	Vision  *bool   `json:"vision"`
	Mmproj  string  `json:"mmproj"`
	Extra   *string `json:"extra"`
}

type runPlan struct {
	model Model
	p     Params
	opt   RunOpt
	bin   string
	last  LastRun
	mode  string // 实际运行方式
}

func (a *App) plan(g Group, r runReq) (*runPlan, error) {
	cfg := a.cfg.Get()
	last, hasLast := cfg.LastRun[g.Key]
	variant := r.Variant
	if variant == "" {
		variant = last.Variant
	}
	model := g.Variants[0]
	for _, v := range g.Variants {
		if v.Path == variant {
			model = v
		}
	}
	vision := len(g.Mmprojs) > 0 && (!hasLast || last.Vision)
	if r.Vision != nil {
		vision = *r.Vision
	}
	mmproj := r.Mmproj
	if mmproj == "" {
		mmproj = last.Mmproj
	}
	if !contains(g.Mmprojs, mmproj) {
		mmproj = ""
		if len(g.Mmprojs) > 0 {
			mmproj = g.Mmprojs[0]
		}
	}
	extra := last.Extra
	if r.Extra != nil {
		extra = *r.Extra
	}
	reqMode := last.Mode
	if r.Mode != nil {
		reqMode = *r.Mode
	}
	if reqMode != "cpu" && reqMode != "gpu" {
		reqMode = ""
	}
	sv := a.servers(false)
	p := a.cfg.Effective(g.Key)
	mode := runMode(p, RunOpt{Mode: reqMode, GPUOK: sv.GPUOK})
	bin, cpuBuild, err := ServerBinary(cfg, mode)
	if err != nil {
		err = fmt.Errorf("找不到 %s 版 llama-server：%v", strings.ToUpper(mode), err)
	}
	opt := RunOpt{Mode: reqMode, GPUOK: sv.GPUOK, CPUBuild: cpuBuild, Extra: extra}
	if cpuBuild && sv.CPU != nil {
		opt.DeviceFlag = sv.CPU.DeviceFlag
	} else {
		opt.DeviceFlag = sv.GPU.DeviceFlag
	}
	if vision {
		opt.Mmproj = mmproj
	}
	return &runPlan{model: model, p: p, opt: opt, bin: bin,
		last: LastRun{Variant: model.Path, Mode: reqMode, Vision: vision, Mmproj: mmproj, Extra: extra}, mode: mode}, err
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, err any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprint(err)})
}

// withGroup 解析路径中的模型组 ID。
func (a *App) withGroup(h func(http.ResponseWriter, *http.Request, Group)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		g, ok := a.findGroup(r.PathValue("id"))
		if !ok {
			writeErr(w, 404, "模型不存在，请刷新列表")
			return
		}
		h(w, r, g)
	}
}

func (a *App) Routes() http.Handler {
	mux := http.NewServeMux()
	sub, _ := fs.Sub(webFS, "web")
	mux.Handle("GET /", http.FileServer(http.FS(sub)))

	mux.HandleFunc("GET /api/config", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"config": a.cfg.Get(), "defaults": defaultParams(), "os": runtime.GOOS})
	})
	mux.HandleFunc("PUT /api/config", func(w http.ResponseWriter, r *http.Request) {
		var c Config
		if err := json.NewDecoder(r.Body).Decode(&c); err != nil {
			writeErr(w, 400, err)
			return
		}
		if err := a.cfg.UpdateSettings(c); err != nil {
			writeErr(w, 500, err)
			return
		}
		writeJSON(w, map[string]any{"ok": true})
	})
	mux.HandleFunc("GET /api/server", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, a.servers(r.URL.Query().Get("refresh") == "1"))
	})
	mux.HandleFunc("GET /api/fs", func(w http.ResponseWriter, r *http.Request) {
		res, err := listDir(r.URL.Query().Get("path"), r.URL.Query().Get("files") == "1")
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		writeJSON(w, res)
	})
	mux.HandleFunc("GET /api/groups", func(w http.ResponseWriter, r *http.Request) {
		cfg := a.cfg.Get()
		type row struct {
			Group
			HasParams   bool   `json:"hasParams"`
			LastVariant string `json:"lastVariant"`
		}
		gs := a.groups()
		out := make([]row, 0, len(gs))
		for _, g := range gs {
			_, ok := cfg.Models[g.Key]
			out = append(out, row{g, ok, cfg.LastRun[g.Key].Variant})
		}
		writeJSON(w, out)
	})
	mux.HandleFunc("GET /api/groups/{id}/params", a.withGroup(func(w http.ResponseWriter, r *http.Request, g Group) {
		cfg := a.cfg.Get()
		writeJSON(w, map[string]any{"group": g, "params": cfg.Models[g.Key], "inherit": merge(defaultParams(), cfg.Global)})
	}))
	mux.HandleFunc("PUT /api/groups/{id}/params", a.withGroup(func(w http.ResponseWriter, r *http.Request, g Group) {
		var p Params
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			writeErr(w, 400, err)
			return
		}
		if err := a.cfg.SetModelParams(g.Key, p); err != nil {
			writeErr(w, 500, err)
			return
		}
		writeJSON(w, map[string]any{"ok": true})
	}))
	mux.HandleFunc("GET /api/groups/{id}/run", a.withGroup(func(w http.ResponseWriter, r *http.Request, g Group) {
		pl, _ := a.plan(g, runReq{})
		writeJSON(w, map[string]any{"variant": pl.last.Variant, "mode": pl.mode, "vision": pl.last.Vision, "mmproj": pl.last.Mmproj, "extra": pl.last.Extra, "gpuOk": pl.opt.GPUOK})
	}))
	mux.HandleFunc("POST /api/groups/{id}/command", a.withGroup(func(w http.ResponseWriter, r *http.Request, g Group) {
		var req runReq
		_ = json.NewDecoder(r.Body).Decode(&req)
		pl, perr := a.plan(g, req)
		bin := pl.bin
		if perr != nil || bin == "" {
			bin = "llama-server"
		}
		o := pl.opt
		o.Port = derefI(pl.p.Port)
		if o.Port == 0 {
			o.Port = a.cfg.Get().BasePort
		}
		args, err := BuildArgs(pl.model, pl.p, o)
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		o.CLI = true
		cliArgs, _ := BuildArgs(pl.model, pl.p, o)
		res := map[string]any{
			"server": quoteCmd(bin, args), "cli": quoteCmd(CLIBinary(pl.bin), cliArgs),
			"autoPort": derefI(pl.p.Port) == 0, "mode": pl.mode,
		}
		if perr != nil {
			res["warn"] = perr.Error()
		}
		writeJSON(w, res)
	}))
	mux.HandleFunc("POST /api/groups/{id}/start", a.withGroup(func(w http.ResponseWriter, r *http.Request, g Group) {
		var req runReq
		_ = json.NewDecoder(r.Body).Decode(&req)
		pl, err := a.plan(g, req)
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		in, err := a.mgr.Start(g, pl.model, pl.p, pl.opt, pl.bin)
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		_ = a.cfg.SetLastRun(g.Key, pl.last)
		writeJSON(w, in)
	}))
	mux.HandleFunc("POST /api/groups/{id}/terminal", a.withGroup(func(w http.ResponseWriter, r *http.Request, g Group) {
		var req runReq
		_ = json.NewDecoder(r.Body).Decode(&req)
		pl, err := a.plan(g, req)
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		o := pl.opt
		o.CLI = true
		args, err := BuildArgs(pl.model, pl.p, o)
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		cli, err := exec.LookPath(CLIBinary(pl.bin))
		if err != nil {
			writeErr(w, 400, "找不到 llama-cli（应与 llama-server 位于同一目录）："+CLIBinary(pl.bin))
			return
		}
		term, err := OpenTerminal("llama-cli · "+pl.model.Name, filepath.Dir(cli), cli, args)
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		_ = a.cfg.SetLastRun(g.Key, pl.last)
		writeJSON(w, map[string]any{"ok": true, "terminal": term})
	}))
	mux.HandleFunc("GET /api/groups/{id}/info", a.withGroup(func(w http.ResponseWriter, r *http.Request, g Group) {
		q := r.URL.Query()
		mi, err := a.info.Fetch(g, q.Get("source"), q.Get("repo"), q.Get("refresh") == "1")
		if err != nil {
			writeErr(w, 502, err)
			return
		}
		writeJSON(w, mi)
	}))
	mux.HandleFunc("GET /api/instances", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, a.mgr.List())
	})
	mux.HandleFunc("POST /api/instances/{id}/stop", func(w http.ResponseWriter, r *http.Request) {
		if err := a.mgr.Stop(r.PathValue("id")); err != nil {
			writeErr(w, 400, err)
			return
		}
		writeJSON(w, map[string]any{"ok": true})
	})
	mux.HandleFunc("DELETE /api/instances/{id}", func(w http.ResponseWriter, r *http.Request) {
		if err := a.mgr.Remove(r.PathValue("id")); err != nil {
			writeErr(w, 400, err)
			return
		}
		writeJSON(w, map[string]any{"ok": true})
	})
	mux.HandleFunc("GET /api/instances/{id}/logs", a.handleLogs)
	mux.HandleFunc("GET /api/stats", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, a.mon.Get())
	})
	mux.HandleFunc("GET /api/stats/history", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, a.mon.History())
	})
	mux.HandleFunc("GET /api/host", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, hostInfo())
	})
	mux.HandleFunc("GET /api/guide", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, a.guide(r.URL.Query().Get("refresh") == "1"))
	})
	return mux
}

// ---------- 目录浏览（用于在页面上选择目录/文件） ----------

type FSList struct {
	Path   string   `json:"path"`
	Parent string   `json:"parent"`
	Sep    string   `json:"sep"`
	Dirs   []string `json:"dirs"`
	Files  []string `json:"files"`
	Roots  []string `json:"roots"`
}

func listDir(path string, files bool) (*FSList, error) {
	if strings.TrimSpace(path) == "" {
		if h, err := os.UserHomeDir(); err == nil {
			path = h
		} else {
			path, _ = os.Getwd()
		}
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	// 选中的是文件时展示其所在目录
	if st, err := os.Stat(abs); err == nil && !st.IsDir() {
		abs = filepath.Dir(abs)
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		return nil, fmt.Errorf("无法读取目录：%v", err)
	}
	res := &FSList{Path: abs, Sep: string(os.PathSeparator), Dirs: []string{}, Files: []string{}}
	if p := filepath.Dir(abs); p != abs {
		res.Parent = p
	}
	for _, e := range entries {
		isDir := e.IsDir()
		if e.Type()&os.ModeSymlink != 0 {
			if st, err := os.Stat(filepath.Join(abs, e.Name())); err == nil {
				isDir = st.IsDir()
			}
		}
		if isDir {
			res.Dirs = append(res.Dirs, e.Name())
		} else if files {
			res.Files = append(res.Files, e.Name())
		}
	}
	sort.Slice(res.Dirs, func(i, j int) bool { return strings.ToLower(res.Dirs[i]) < strings.ToLower(res.Dirs[j]) })
	sort.Slice(res.Files, func(i, j int) bool { return strings.ToLower(res.Files[i]) < strings.ToLower(res.Files[j]) })
	if runtime.GOOS == "windows" {
		for c := 'A'; c <= 'Z'; c++ {
			d := string(c) + `:\`
			if _, err := os.Stat(d); err == nil {
				res.Roots = append(res.Roots, d)
			}
		}
	} else {
		res.Roots = []string{"/"}
	}
	if h, err := os.UserHomeDir(); err == nil {
		res.Roots = append(res.Roots, h)
	}
	return res, nil
}

// handleLogs 以 SSE 推送实例日志：先发送缓冲区中的历史，再实时推送。
// 为减少前端重绘，新行按 200ms 合并为一批发送。
func (a *App) handleLogs(w http.ResponseWriter, r *http.Request) {
	in := a.mgr.Get(r.PathValue("id"))
	if in == nil {
		writeErr(w, 404, "实例不存在")
		return
	}
	fl, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, 500, "不支持流式输出")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	hist, ch := in.logs.Subscribe()
	defer in.logs.Unsubscribe(ch)
	send := func(lines []LogLine) {
		b, _ := json.Marshal(lines)
		fmt.Fprintf(w, "data: %s\n\n", b)
		fl.Flush()
	}
	send(hist)
	tick := time.NewTicker(200 * time.Millisecond)
	defer tick.Stop()
	ping := time.NewTicker(15 * time.Second)
	defer ping.Stop()
	var batch []LogLine
	for {
		select {
		case <-r.Context().Done():
			return
		case l := <-ch:
			batch = append(batch, l)
		case <-tick.C:
			if len(batch) > 0 {
				send(batch)
				batch = batch[:0]
			}
		case <-ping.C:
			fmt.Fprint(w, ": ping\n\n")
			fl.Flush()
		}
	}
}
