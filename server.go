package main

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
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

	siMu sync.Mutex
	si   *ServerInfo
	siOf string // 探测时使用的配置路径
}

func (a *App) serverInfo(refresh bool) ServerInfo {
	path := a.cfg.Get().LlamaServerPath
	a.siMu.Lock()
	defer a.siMu.Unlock()
	if refresh || a.si == nil || a.siOf != path {
		si := ProbeServer(path)
		a.si, a.siOf = &si, path
	}
	return *a.si
}

func (a *App) findModel(id string) (Model, bool) {
	for _, m := range ScanModels(a.cfg.Get().ModelDirs) {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
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

func (a *App) Routes() http.Handler {
	mux := http.NewServeMux()
	sub, _ := fs.Sub(webFS, "web")
	mux.Handle("GET /", http.FileServer(http.FS(sub)))

	mux.HandleFunc("GET /api/config", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"config": a.cfg.Get(), "defaults": defaultParams()})
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
		writeJSON(w, a.serverInfo(r.URL.Query().Get("refresh") == "1"))
	})
	mux.HandleFunc("GET /api/models", func(w http.ResponseWriter, r *http.Request) {
		cfg := a.cfg.Get()
		models := ScanModels(cfg.ModelDirs)
		type row struct {
			Model
			HasParams bool `json:"hasParams"`
		}
		out := make([]row, 0, len(models))
		for _, m := range models {
			_, ok := cfg.Models[m.Path]
			out = append(out, row{m, ok})
		}
		writeJSON(w, out)
	})
	mux.HandleFunc("GET /api/models/{id}/params", func(w http.ResponseWriter, r *http.Request) {
		m, ok := a.findModel(r.PathValue("id"))
		if !ok {
			writeErr(w, 404, "模型不存在")
			return
		}
		cfg := a.cfg.Get()
		inherit := merge(defaultParams(), cfg.Global)
		writeJSON(w, map[string]any{"model": m, "params": cfg.Models[m.Path], "inherit": inherit})
	})
	mux.HandleFunc("PUT /api/models/{id}/params", func(w http.ResponseWriter, r *http.Request) {
		m, ok := a.findModel(r.PathValue("id"))
		if !ok {
			writeErr(w, 404, "模型不存在")
			return
		}
		var p Params
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			writeErr(w, 400, err)
			return
		}
		if err := a.cfg.SetModelParams(m.Path, p); err != nil {
			writeErr(w, 500, err)
			return
		}
		writeJSON(w, map[string]any{"ok": true})
	})
	mux.HandleFunc("GET /api/models/{id}/command", func(w http.ResponseWriter, r *http.Request) {
		m, ok := a.findModel(r.PathValue("id"))
		if !ok {
			writeErr(w, 404, "模型不存在")
			return
		}
		si := a.serverInfo(false)
		p := a.cfg.Effective(m.Path)
		port := derefI(p.Port)
		if port == 0 {
			port = a.cfg.Get().BasePort
		}
		args, err := BuildArgs(m, p, port, si.GPU, si.DeviceFlag, r.URL.Query().Get("extra"))
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		bin := si.Path
		if bin == "" {
			bin = "llama-server"
		}
		writeJSON(w, map[string]any{"command": quoteCmd(bin, args), "autoPort": derefI(p.Port) == 0})
	})
	mux.HandleFunc("GET /api/models/{id}/info", func(w http.ResponseWriter, r *http.Request) {
		m, ok := a.findModel(r.PathValue("id"))
		if !ok {
			writeErr(w, 404, "模型不存在")
			return
		}
		q := r.URL.Query()
		mi, err := a.info.Fetch(m, q.Get("source"), q.Get("repo"), q.Get("refresh") == "1")
		if err != nil {
			writeErr(w, 502, err)
			return
		}
		writeJSON(w, mi)
	})
	mux.HandleFunc("POST /api/models/{id}/start", func(w http.ResponseWriter, r *http.Request) {
		m, ok := a.findModel(r.PathValue("id"))
		if !ok {
			writeErr(w, 404, "模型不存在")
			return
		}
		var body struct {
			Extra string `json:"extra"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		si := a.serverInfo(false)
		in, err := a.mgr.Start(m, body.Extra, si.GPU, si.DeviceFlag)
		if err != nil {
			writeErr(w, 400, err)
			return
		}
		writeJSON(w, in)
	})
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
	return mux
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
