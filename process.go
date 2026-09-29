package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
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
	ID        string    `json:"id"` // 与模型组 ID 相同
	GroupName string    `json:"groupName"`
	ModelName string    `json:"modelName"`
	Mode      string    `json:"mode"`
	Vision    bool      `json:"vision"`
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

// RunOpt 是单次启动/生成命令的选项。
type RunOpt struct {
	Port       int
	Mode       string // 本次指定的运行方式，空表示按参数配置
	GPUOK      bool   // 系统与 GPU 版程序可用 GPU
	DeviceFlag bool   // 所用程序支持 --device
	CPUBuild   bool   // 使用的是独立的 CPU 版程序
	Mmproj     string // 启用视觉时的投影文件，为空表示不启用
	Extra      string // 本次附加参数
	CLI        bool   // 生成 llama-cli 参数（去掉服务端专用参数）
}

// argSpec 描述 Params 字段（JSON 名）到命令行参数的映射。
type argSpec struct {
	key, flag string
	kind      int  // argVal：带值；argTrue：为 true 时加 flag；argFalse：为 false 时加 flag
	gpu       bool // 仅 GPU 方式有效
	server    bool // 仅 llama-server 有效
}

const (
	argVal = iota
	argTrue
	argFalse
)

var argSpecs = []argSpec{
	{key: "nGpuLayers", flag: "-ngl", gpu: true},
	{key: "fit", flag: "-fit", gpu: true},
	{key: "fitTarget", flag: "-fitt", gpu: true},
	{key: "fitCtx", flag: "-fitc", gpu: true},
	{key: "device", flag: "--device", gpu: true},
	{key: "splitMode", flag: "-sm", gpu: true},
	{key: "tensorSplit", flag: "-ts", gpu: true},
	{key: "mainGpu", flag: "-mg", gpu: true},
	{key: "cpuMoe", flag: "-cmoe", kind: argTrue, gpu: true},
	{key: "nCpuMoe", flag: "-ncmoe", gpu: true},
	{key: "kvOffload", flag: "-nkvo", kind: argFalse, gpu: true},
	{key: "ctxSize", flag: "-c"},
	{key: "threads", flag: "-t"},
	{key: "threadsBatch", flag: "-tb"},
	{key: "batchSize", flag: "-b"},
	{key: "ubatchSize", flag: "-ub"},
	{key: "parallel", flag: "-np", server: true},
	{key: "flashAttn", flag: "-fa"},
	{key: "cacheTypeK", flag: "-ctk"},
	{key: "cacheTypeV", flag: "-ctv"},
	{key: "cacheRam", flag: "-cram", server: true},
	{key: "mlock", flag: "--mlock", kind: argTrue},
	{key: "noMmap", flag: "--no-mmap", kind: argTrue},
	{key: "contextShift", flag: "--context-shift", kind: argTrue},
	{key: "temp", flag: "--temp"},
	{key: "topK", flag: "--top-k"},
	{key: "topP", flag: "--top-p"},
	{key: "minP", flag: "--min-p"},
	{key: "repeatPenalty", flag: "--repeat-penalty"},
	{key: "presencePenalty", flag: "--presence-penalty"},
	{key: "seed", flag: "-s"},
	{key: "nPredict", flag: "-n"},
	{key: "jinja", flag: "--jinja", kind: argTrue},
	{key: "jinja", flag: "--no-jinja", kind: argFalse},
	{key: "reasoning", flag: "-rea"},
	{key: "reasoningBudget", flag: "--reasoning-budget"},
	{key: "chatTemplateKwargs", flag: "--chat-template-kwargs"},
	{key: "apiKey", flag: "--api-key", server: true},
	{key: "webui", flag: "--no-webui", kind: argFalse, server: true},
	{key: "metrics", flag: "--metrics", kind: argTrue, server: true},
}

// BuildArgs 根据生效参数生成命令行参数。
func BuildArgs(m Model, p Params, o RunOpt) ([]string, error) {
	if o.Mode != "" {
		p.Mode = &o.Mode
	}
	cpu := EffectiveMode(p, o.GPUOK) == "cpu"
	a := []string{"-m", m.Path}
	if cpu && !o.CPUBuild {
		a = append(a, "-ngl", "0")
		if o.DeviceFlag {
			a = append(a, "--device", "none")
		}
	}
	vals := map[string]any{}
	b, _ := json.Marshal(p)
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber() // 避免大整数被格式化为科学计数法
	_ = dec.Decode(&vals)
	for _, s := range argSpecs {
		v, ok := vals[s.key]
		if !ok || (s.gpu && cpu) || (s.server && o.CLI) {
			continue
		}
		switch s.kind {
		case argTrue:
			if v == true {
				a = append(a, s.flag)
			}
		case argFalse:
			if v == false {
				a = append(a, s.flag)
			}
		default:
			str := strings.TrimSpace(fmt.Sprint(v))
			if str == "" || str == "none" && s.key == "flashAttn" {
				continue
			}
			a = append(a, s.flag, str)
		}
	}
	if o.Mmproj != "" {
		a = append(a, "--mmproj", o.Mmproj)
	}
	if !o.CLI {
		alias := deref(p.Alias, "")
		if alias == "" {
			alias = m.Name
		}
		a = append(a, "-a", alias, "--host", deref(p.Host, "127.0.0.1"), "--port", fmt.Sprint(o.Port))
	}
	for _, s := range []string{deref(p.ExtraArgs, ""), o.Extra} {
		parts, err := SplitArgs(s)
		if err != nil {
			return nil, err
		}
		a = append(a, parts...)
	}
	return a, nil
}

func runMode(p Params, o RunOpt) string {
	if o.Mode != "" {
		p.Mode = &o.Mode
	}
	return EffectiveMode(p, o.GPUOK)
}

// EffectiveMode 返回实际运行方式：无可用 GPU 时强制为 cpu。
func EffectiveMode(p Params, gpuOK bool) string {
	if !gpuOK || deref(p.Mode, "gpu") == "cpu" {
		return "cpu"
	}
	return "gpu"
}

// ServerBinary 按运行方式选择 llama-server：GPU 方式用 GPU 版；
// CPU 方式优先用 CPU 版，未配置时退回 GPU 版（并通过参数禁用 GPU）。
func ServerBinary(cfg Config, mode string) (bin string, cpuBuild bool, err error) {
	if mode == "cpu" && strings.TrimSpace(cfg.ServerPathCPU) != "" {
		bin, err = ResolveServerPath(cfg.ServerPathCPU)
		return bin, true, err
	}
	bin, err = ResolveServerPath(cfg.ServerPathGPU)
	return bin, false, err
}

// CLIBinary 返回与 llama-server 同目录的 llama-cli。
func CLIBinary(server string) string {
	if server == "" {
		return "llama-cli"
	}
	dir := filepath.Dir(server)
	name := "llama-cli"
	if strings.EqualFold(filepath.Ext(server), ".exe") {
		name += ".exe"
	}
	p := filepath.Join(dir, name)
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return p // 即使不存在也给出同目录路径，便于用户对照
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

// Start 启动模型组 g 中的模型 model。非多开模式下会先停止其它正在运行的实例。
func (m *Manager) Start(g Group, model Model, p Params, o RunOpt, bin string) (*Instance, error) {
	cfg := m.cfg.Get()
	if !cfg.AllowMulti {
		m.StopAll(g.ID)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if old := m.instances[g.ID]; old != nil && old.alive() {
		return nil, errors.New("该模型已在运行")
	}
	host := deref(p.Host, "127.0.0.1")
	port, err := m.pickPort(host, derefI(p.Port), cfg.BasePort)
	if err != nil {
		return nil, err
	}
	o.Port = port
	args, err := BuildArgs(model, p, o)
	if err != nil {
		return nil, err
	}
	in := &Instance{
		ID: g.ID, GroupName: g.Name, ModelName: model.Name, ModelPath: model.Path, Host: host, Port: port,
		Mode: runMode(p, o), Vision: o.Mmproj != "",
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
	if _, ok := m.instances[g.ID]; !ok {
		m.order = append(m.order, g.ID)
	}
	m.instances[g.ID] = in

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
	parts := []string{shellQuote(bin)}
	for _, a := range args {
		parts = append(parts, shellQuote(a))
	}
	cmd := strings.Join(parts, " ")
	if runtime.GOOS == "windows" && strings.HasPrefix(cmd, `"`) {
		// PowerShell 需用 & 调用带引号的路径（在 cmd 中使用时去掉开头的 &）
		return "& " + cmd
	}
	return cmd
}

func shellQuote(a string) string {
	if a != "" && !strings.ContainsAny(a, " \t\"'&|<>()$`;*?!#%^") {
		return a
	}
	if runtime.GOOS == "windows" {
		return `"` + strings.ReplaceAll(a, `"`, `\"`) + `"`
	}
	return "'" + strings.ReplaceAll(a, "'", `'\''`) + "'"
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
