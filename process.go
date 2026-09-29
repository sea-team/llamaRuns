package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------- 日志环形缓冲 ----------

type LogLine struct {
	N    int64  `json:"n"`
	Text string `json:"t"`
}

type LogBuffer struct {
	mu    sync.Mutex
	max   int
	lines []LogLine
	next  int64
	subs  map[chan LogLine]struct{}
}

func NewLogBuffer(max int) *LogBuffer {
	return &LogBuffer{max: max, subs: map[chan LogLine]struct{}{}}
}

func (b *LogBuffer) Add(text string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	l := LogLine{N: b.next, Text: text}
	b.next++
	b.lines = append(b.lines, l)
	if over := len(b.lines) - b.max; over > 0 {
		b.lines = append(b.lines[:0:0], b.lines[over:]...)
	}
	for ch := range b.subs {
		select {
		case ch <- l:
		default: // 订阅者太慢时丢弃，避免阻塞进程输出
		}
	}
}

func (b *LogBuffer) Subscribe() ([]LogLine, chan LogLine) {
	b.mu.Lock()
	defer b.mu.Unlock()
	ch := make(chan LogLine, 512)
	b.subs[ch] = struct{}{}
	return append([]LogLine(nil), b.lines...), ch
}

func (b *LogBuffer) Unsubscribe(ch chan LogLine) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.subs, ch)
}

// ---------- 实例 ----------

type Instance struct {
	ID        string    `json:"id"` // 与模型 ID 相同
	ModelName string    `json:"modelName"`
	ModelPath string    `json:"modelPath"`
	Host      string    `json:"host"`
	Port      int       `json:"port"`
	Args      []string  `json:"args"`
	Status    string    `json:"status"` // loading / running / stopping / stopped / error
	PID       int       `json:"pid"`
	StartedAt time.Time `json:"startedAt"`
	ExitMsg   string    `json:"exitMsg"`

	cmd  *exec.Cmd
	logs *LogBuffer
	done chan struct{}
}

type Manager struct {
	mu        sync.Mutex
	cfg       *ConfigStore
	instances map[string]*Instance
	order     []string
}

func NewManager(cfg *ConfigStore) *Manager {
	return &Manager{cfg: cfg, instances: map[string]*Instance{}}
}

// ResolveServerPath 返回 llama-server 可执行文件路径，未配置时从 PATH 查找。
func ResolveServerPath(configured string) (string, error) {
	p := strings.TrimSpace(configured)
	if p == "" {
		p = "llama-server"
	}
	if st, err := os.Stat(p); err == nil && st.IsDir() {
		for _, n := range []string{"llama-server", "llama-server.exe"} {
			if _, err := os.Stat(p + string(os.PathSeparator) + n); err == nil {
				return p + string(os.PathSeparator) + n, nil
			}
		}
	}
	return exec.LookPath(p)
}

// BuildArgs 根据生效参数生成命令行参数。
func BuildArgs(m Model, p Params, port int, gpuOK, deviceFlag bool, extra string) ([]string, error) {
	a := []string{"-m", m.Path}
	add := func(k string, v any) { a = append(a, k, fmt.Sprint(v)) }
	mode := deref(p.Mode, "gpu")
	if !gpuOK {
		mode = "cpu"
	}
	if mode == "cpu" {
		add("-ngl", 0)
		if deviceFlag {
			add("--device", "none")
		}
	} else {
		if p.NGpuLayers != nil {
			add("-ngl", *p.NGpuLayers)
		}
		if d := deref(p.Device, ""); d != "" {
			add("--device", d)
		}
	}
	if p.CtxSize != nil {
		add("-c", *p.CtxSize)
	}
	if v := derefI(p.Threads); v > 0 {
		add("-t", v)
	}
	if v := derefI(p.BatchSize); v > 0 {
		add("-b", v)
	}
	if v := derefI(p.UBatchSize); v > 0 {
		add("-ub", v)
	}
	if v := derefI(p.Parallel); v > 0 {
		add("-np", v)
	}
	if v := deref(p.FlashAttn, ""); v != "" && v != "none" {
		add("-fa", v)
	}
	if v := deref(p.CacheTypeK, ""); v != "" {
		add("-ctk", v)
	}
	if v := deref(p.CacheTypeV, ""); v != "" {
		add("-ctv", v)
	}
	if deref(p.Mlock, false) {
		a = append(a, "--mlock")
	}
	if deref(p.NoMmap, false) {
		a = append(a, "--no-mmap")
	}
	if deref(p.Jinja, false) {
		a = append(a, "--jinja")
	}
	if p.Temp != nil {
		add("--temp", *p.Temp)
	}
	if p.TopK != nil {
		add("--top-k", *p.TopK)
	}
	if p.TopP != nil {
		add("--top-p", *p.TopP)
	}
	if p.MinP != nil {
		add("--min-p", *p.MinP)
	}
	if p.RepeatPenalty != nil {
		add("--repeat-penalty", *p.RepeatPenalty)
	}
	if v := deref(p.APIKey, ""); v != "" {
		add("--api-key", v)
	}
	if v := deref(p.Mmproj, ""); v != "" {
		add("--mmproj", v)
	}
	alias := deref(p.Alias, "")
	if alias == "" {
		alias = m.Name
	}
	add("-a", alias)
	add("--host", deref(p.Host, "127.0.0.1"))
	add("--port", port)
	for _, s := range []string{deref(p.ExtraArgs, ""), extra} {
		parts, err := SplitArgs(s)
		if err != nil {
			return nil, err
		}
		a = append(a, parts...)
	}
	return a, nil
}

func deref[T any](p *T, def T) T {
	if p == nil {
		return def
	}
	return *p
}

func derefI(p *int) int { return deref(p, 0) }

// SplitArgs 按类 shell 规则切分参数，支持单/双引号。
func SplitArgs(s string) ([]string, error) {
	var out []string
	var cur strings.Builder
	var quote rune
	has := false
	for _, r := range s {
		switch {
		case quote != 0:
			if r == quote {
				quote = 0
			} else {
				cur.WriteRune(r)
			}
		case r == '"' || r == '\'':
			quote, has = r, true
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			if has {
				out = append(out, cur.String())
				cur.Reset()
				has = false
			}
		default:
			cur.WriteRune(r)
			has = true
		}
	}
	if quote != 0 {
		return nil, errors.New("附加参数中引号未闭合")
	}
	if has {
		out = append(out, cur.String())
	}
	return out, nil
}

func portFree(host string, port int) bool {
	l, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
	if err != nil {
		return false
	}
	l.Close()
	return true
}

func (m *Manager) pickPort(host string, want, base int) (int, error) {
	used := map[int]bool{}
	for _, in := range m.instances {
		if in.alive() {
			used[in.Port] = true
		}
	}
	if want > 0 {
		if used[want] || !portFree(host, want) {
			return 0, fmt.Errorf("端口 %d 已被占用", want)
		}
		return want, nil
	}
	for p := base; p < base+1000; p++ {
		if !used[p] && portFree(host, p) {
			return p, nil
		}
	}
	return 0, errors.New("找不到可用端口")
}

func (in *Instance) alive() bool {
	return in.Status == "loading" || in.Status == "running" || in.Status == "stopping"
}

// Start 启动模型。非多开模式下会先停止其它正在运行的实例。
func (m *Manager) Start(model Model, extra string, gpuOK, deviceFlag bool) (*Instance, error) {
	cfg := m.cfg.Get()
	bin, err := ResolveServerPath(cfg.LlamaServerPath)
	if err != nil {
		return nil, fmt.Errorf("找不到 llama-server：%v", err)
	}
	if !cfg.AllowMulti {
		m.StopAll(model.ID)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if old := m.instances[model.ID]; old != nil && old.alive() {
		return nil, errors.New("该模型已在运行")
	}
	p := m.cfg.Effective(model.Path)
	host := deref(p.Host, "127.0.0.1")
	port, err := m.pickPort(host, derefI(p.Port), cfg.BasePort)
	if err != nil {
		return nil, err
	}
	args, err := BuildArgs(model, p, port, gpuOK, deviceFlag, extra)
	if err != nil {
		return nil, err
	}
	in := &Instance{
		ID: model.ID, ModelName: model.Name, ModelPath: model.Path, Host: host, Port: port,
		Args: args, Status: "loading", StartedAt: time.Now(),
		logs: NewLogBuffer(cfg.MaxLogLines), done: make(chan struct{}),
	}
	in.logs.Add("$ " + quoteCmd(bin, args))
	cmd := exec.Command(bin, args...)
	pr, pw := io.Pipe()
	cmd.Stdout, cmd.Stderr = pw, pw
	prepareCmd(cmd)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("启动失败：%v", err)
	}
	in.cmd, in.PID = cmd, cmd.Process.Pid
	if _, ok := m.instances[model.ID]; !ok {
		m.order = append(m.order, model.ID)
	}
	m.instances[model.ID] = in

	go pumpLogs(pr, in.logs)
	go func() {
		err := cmd.Wait()
		pw.Close()
		m.mu.Lock()
		if in.Status == "stopping" {
			in.Status = "stopped"
			in.ExitMsg = "已停止"
		} else {
			in.Status = "error"
			in.ExitMsg = "进程异常退出"
			if err != nil {
				in.ExitMsg += "：" + err.Error()
			}
		}
		m.mu.Unlock()
		in.logs.Add("[面板] " + in.ExitMsg)
		close(in.done)
	}()
	go m.waitReady(in)
	return in, nil
}

func (m *Manager) waitReady(in *Instance) {
	host := in.Host
	if host == "0.0.0.0" || host == "::" || host == "" {
		host = "127.0.0.1"
	}
	url := "http://" + net.JoinHostPort(host, strconv.Itoa(in.Port)) + "/health"
	client := &http.Client{Timeout: 2 * time.Second}
	for {
		select {
		case <-in.done:
			return
		case <-time.After(time.Second):
		}
		resp, err := client.Get(url)
		if err != nil {
			continue
		}
		resp.Body.Close()
		if resp.StatusCode == 200 {
			m.mu.Lock()
			if in.Status == "loading" {
				in.Status = "running"
			}
			m.mu.Unlock()
			in.logs.Add("[面板] 模型已就绪：" + url[:len(url)-len("/health")])
			return
		}
	}
}

// pumpLogs 按 \n 或 \r 切分输出行。
func pumpLogs(r io.Reader, lb *LogBuffer) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	sc.Split(func(data []byte, atEOF bool) (int, []byte, error) {
		if i := bytes.IndexAny(data, "\r\n"); i >= 0 {
			return i + 1, data[:i], nil
		}
		if atEOF && len(data) > 0 {
			return len(data), data, nil
		}
		return 0, nil, nil
	})
	for sc.Scan() {
		if t := strings.TrimRight(sc.Text(), " "); t != "" {
			lb.Add(t)
		}
	}
	io.Copy(io.Discard, r)
}

func quoteCmd(bin string, args []string) string {
	parts := []string{bin}
	for _, a := range args {
		if a == "" || strings.ContainsAny(a, " \t\"'") {
			a = strconv.Quote(a)
		}
		parts = append(parts, a)
	}
	return strings.Join(parts, " ")
}

// Stop 停止实例，先尝试优雅退出，超时后强制结束。
func (m *Manager) Stop(id string) error {
	m.mu.Lock()
	in := m.instances[id]
	if in == nil || !in.alive() {
		m.mu.Unlock()
		return errors.New("实例未运行")
	}
	in.Status = "stopping"
	m.mu.Unlock()
	if runtime.GOOS == "windows" {
		_ = in.cmd.Process.Kill()
	} else {
		_ = in.cmd.Process.Signal(os.Interrupt)
	}
	select {
	case <-in.done:
	case <-time.After(8 * time.Second):
		_ = in.cmd.Process.Kill()
		<-in.done
	}
	return nil
}

func (m *Manager) StopAll(except string) {
	m.mu.Lock()
	var ids []string
	for id, in := range m.instances {
		if id != except && in.alive() {
			ids = append(ids, id)
		}
	}
	m.mu.Unlock()
	var wg sync.WaitGroup
	for _, id := range ids {
		wg.Add(1)
		go func(id string) { defer wg.Done(); _ = m.Stop(id) }(id)
	}
	wg.Wait()
}

func (m *Manager) Remove(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	in := m.instances[id]
	if in == nil {
		return errors.New("实例不存在")
	}
	if in.alive() {
		return errors.New("请先停止实例")
	}
	delete(m.instances, id)
	for i, v := range m.order {
		if v == id {
			m.order = append(m.order[:i], m.order[i+1:]...)
			break
		}
	}
	return nil
}

func (m *Manager) List() []Instance {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Instance, 0, len(m.order))
	for _, id := range m.order {
		out = append(out, *m.instances[id])
	}
	return out
}

func (m *Manager) Get(id string) *Instance {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.instances[id]
}

// ---------- llama-server 能力探测 ----------

type ServerInfo struct {
	Path       string   `json:"path"`
	Version    string   `json:"version"`
	Devices    []string `json:"devices"`
	DeviceFlag bool     `json:"deviceFlag"` // 支持 --list-devices / --device
	GPU        bool     `json:"gpu"`
	Error      string   `json:"error"`
}

var devRe = regexp.MustCompile(`^\s+(\S+):\s+(.+)$`)

func ProbeServer(configured string) ServerInfo {
	var si ServerInfo
	bin, err := ResolveServerPath(configured)
	if err != nil {
		si.Error = "找不到 llama-server：" + err.Error()
		return si
	}
	si.Path = bin
	run := func(arg string) (string, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		c := exec.CommandContext(ctx, bin, arg)
		prepareCmd(c)
		out, err := c.CombinedOutput()
		return string(out), err
	}
	if out, err := run("--version"); err == nil || out != "" {
		for _, l := range strings.Split(out, "\n") {
			if strings.HasPrefix(strings.TrimSpace(l), "version:") {
				si.Version = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(l), "version:"))
			}
		}
	} else {
		si.Error = "执行 --version 失败：" + err.Error()
	}
	if out, err := run("--list-devices"); err == nil {
		si.DeviceFlag = true
		started := false
		for _, l := range strings.Split(out, "\n") {
			if strings.Contains(l, "Available devices") {
				started = true
				continue
			}
			if started {
				if mm := devRe.FindStringSubmatch(strings.TrimRight(l, "\r")); mm != nil {
					si.Devices = append(si.Devices, mm[1]+": "+mm[2])
				}
			}
		}
		si.GPU = len(si.Devices) > 0
	} else {
		// 旧版本不支持 --list-devices：按平台/驱动粗略判断
		_, nv := exec.LookPath("nvidia-smi")
		si.GPU = nv == nil || runtime.GOOS == "darwin"
	}
	return si
}
