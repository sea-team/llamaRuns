package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/process"
)

type GPUStat struct {
	Vendor   string  `json:"vendor"`
	Name     string  `json:"name"`
	Driver   string  `json:"driver"`
	CUDA     string  `json:"cuda"`
	Power    float64 `json:"power"`      // W
	PowerCap float64 `json:"powerLimit"` // W
	Fan      float64 `json:"fan"`        // %
	PState   string  `json:"pstate"`
	Util     float64 `json:"util"`     // %
	MemUsed  uint64  `json:"memUsed"`  // bytes
	MemTotal uint64  `json:"memTotal"` // bytes
	Temp     float64 `json:"temp"`     // ℃，0 表示未知
}

type SysStat struct {
	CPUPercent float64            `json:"cpuPercent"`
	CPUCores   int                `json:"cpuCores"`
	CPUTemp    float64            `json:"cpuTemp"`
	MemTotal   uint64             `json:"memTotal"`
	MemUsed    uint64             `json:"memUsed"`
	MemPercent float64            `json:"memPercent"`
	SwapTotal  uint64             `json:"swapTotal"`
	SwapUsed   uint64             `json:"swapUsed"`
	GPUs       []GPUStat          `json:"gpus"`
	ProcMem    map[string]uint64  `json:"procMem"` // 实例 ID -> RSS
	ProcCPU    map[string]float64 `json:"procCpu"`
	Time       int64              `json:"time"`
}

// HistPoint 是一次采样的占用率（百分比），用于绘制曲线。
type HistPoint struct {
	T    int64     `json:"t"` // 毫秒时间戳
	CPU  float64   `json:"cpu"`
	Mem  float64   `json:"mem"`
	GPU  []float64 `json:"gpu"`
	VRAM []float64 `json:"vram"`
}

const histMax = 90 // 约 3 分钟（每 2 秒一次）

type SysMonitor struct {
	mu    sync.RWMutex
	stat  SysStat
	hist  []HistPoint
	mgr   *Manager
	procs map[int32]*process.Process
}

func NewSysMonitor(mgr *Manager) *SysMonitor {
	s := &SysMonitor{mgr: mgr, procs: map[int32]*process.Process{}}
	s.stat.CPUCores, _ = cpu.Counts(true)
	go s.loop()
	return s
}

func (s *SysMonitor) Get() SysStat {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.stat
}

func (s *SysMonitor) History() []HistPoint {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]HistPoint{}, s.hist...)
}

func (s *SysMonitor) loop() {
	for {
		st := SysStat{CPUCores: s.stat.CPUCores}
		if p, err := cpu.Percent(1500*time.Millisecond, false); err == nil && len(p) > 0 {
			st.CPUPercent = p[0]
		}
		if vm, err := mem.VirtualMemory(); err == nil {
			st.MemTotal, st.MemUsed, st.MemPercent = vm.Total, vm.Used, vm.UsedPercent
		}
		if sw, err := mem.SwapMemory(); err == nil {
			st.SwapTotal, st.SwapUsed = sw.Total, sw.Used
		}
		st.CPUTemp = cpuTemp()
		st.GPUs = gpuStats()
		st.ProcMem, st.ProcCPU = s.procStats()
		st.Time = time.Now().UnixMilli()
		hp := HistPoint{T: st.Time, CPU: st.CPUPercent, Mem: st.MemPercent}
		for _, g := range st.GPUs {
			hp.GPU = append(hp.GPU, g.Util)
			v := 0.0
			if g.MemTotal > 0 {
				v = float64(g.MemUsed) / float64(g.MemTotal) * 100
			}
			hp.VRAM = append(hp.VRAM, v)
		}
		s.mu.Lock()
		s.stat = st
		s.hist = append(s.hist, hp)
		if len(s.hist) > histMax {
			s.hist = s.hist[len(s.hist)-histMax:]
		}
		s.mu.Unlock()
		time.Sleep(500 * time.Millisecond)
	}
}

func (s *SysMonitor) procStats() (map[string]uint64, map[string]float64) {
	pm, pc := map[string]uint64{}, map[string]float64{}
	live := map[int32]bool{}
	for _, in := range s.mgr.List() {
		if !in.alive() || in.PID == 0 {
			continue
		}
		pid := int32(in.PID)
		live[pid] = true
		p := s.procs[pid]
		if p == nil {
			var err error
			if p, err = process.NewProcess(pid); err != nil {
				continue
			}
			s.procs[pid] = p
		}
		if mi, err := p.MemoryInfo(); err == nil {
			pm[in.ID] = mi.RSS
		}
		if c, err := p.Percent(0); err == nil {
			pc[in.ID] = c
		}
	}
	for pid := range s.procs {
		if !live[pid] {
			delete(s.procs, pid)
		}
	}
	return pm, pc
}

var cpuSensorKeys = []string{"coretemp", "k10temp", "zenpower", "cpu", "package", "tctl", "tdie", "x86_pkg", "soc", "tsens", "acpitz", "thermal_zone"}

func cpuTemp() float64 {
	ts, _ := host.SensorsTemperatures()
	best, fallback := 0.0, 0.0
	for _, t := range ts {
		if t.Temperature <= 0 || t.Temperature > 150 {
			continue
		}
		k := strings.ToLower(t.SensorKey)
		if strings.Contains(k, "gpu") || strings.Contains(k, "amdgpu") || strings.Contains(k, "nvme") || strings.Contains(k, "battery") {
			continue
		}
		matched := false
		for _, key := range cpuSensorKeys {
			if strings.Contains(k, key) {
				matched = true
				break
			}
		}
		if matched && t.Temperature > best {
			best = t.Temperature
		}
		if t.Temperature > fallback {
			fallback = t.Temperature
		}
	}
	if best == 0 {
		best = fallback
	}
	return best
}

func gpuStats() []GPUStat {
	if g := nvidiaStats(); len(g) > 0 {
		return g
	}
	if g := rocmStats(); len(g) > 0 {
		return g
	}
	return amdSysfsStats()
}

func lookTool(name string) string {
	if p, err := exec.LookPath(name); err == nil {
		return p
	}
	return ""
}

var (
	nvidiaSMI = lookTool("nvidia-smi")
	rocmSMI   = lookTool("rocm-smi")
	amdSMI    = lookTool("amd-smi")
)

func runTool(bin string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c := exec.CommandContext(ctx, bin, args...)
	prepareCmd(c)
	out, err := c.CombinedOutput()
	return string(out), err
}

var (
	cudaOnce sync.Once
	cudaVer  string
	cudaRe   = regexp.MustCompile(`CUDA Version:\s*([\d.]+)`)
)

// nvidiaCUDA 返回驱动支持的最高 CUDA 版本（来自 nvidia-smi 表头）。
func nvidiaCUDA() string {
	cudaOnce.Do(func() {
		if out, err := runTool(nvidiaSMI); err == nil {
			if m := cudaRe.FindStringSubmatch(out); m != nil {
				cudaVer = m[1]
			}
		}
	})
	return cudaVer
}

// HostInfo 是不随时间变化的系统信息。
type HostInfo struct {
	Hostname  string `json:"hostname"`
	OS        string `json:"os"`
	Platform  string `json:"platform"`
	Arch      string `json:"arch"`
	CPUModel  string `json:"cpuModel"`
	Physical  int    `json:"physicalCores"`
	Logical   int    `json:"logicalCores"`
	BootTime  uint64 `json:"bootTime"`
	GoVersion string `json:"goVersion"`
}

var hostInfo = sync.OnceValue(func() HostInfo {
	h := HostInfo{OS: runtime.GOOS, Arch: runtime.GOARCH, GoVersion: runtime.Version()}
	if hi, err := host.Info(); err == nil {
		h.Hostname, h.BootTime = hi.Hostname, hi.BootTime
		h.Platform = strings.TrimSpace(hi.Platform + " " + hi.PlatformVersion)
	}
	if ci, err := cpu.Info(); err == nil && len(ci) > 0 {
		h.CPUModel = strings.TrimSpace(ci[0].ModelName)
	}
	h.Physical, _ = cpu.Counts(false)
	h.Logical, _ = cpu.Counts(true)
	return h
})

// rocmStats 解析 rocm-smi 的 JSON 输出（字段名随版本略有差异，按关键字匹配）。
func rocmStats() []GPUStat {
	if rocmSMI == "" {
		return nil
	}
	out, err := runTool(rocmSMI, "--showproductname", "--showuse", "--showmeminfo", "vram", "--showtemp", "--json")
	if err != nil {
		return nil
	}
	var data map[string]map[string]any
	if i := strings.Index(out, "{"); i >= 0 {
		out = out[i:]
	}
	if json.Unmarshal([]byte(out), &data) != nil {
		return nil
	}
	keys := make([]string, 0, len(data))
	for k := range data {
		if strings.HasPrefix(k, "card") {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	var gs []GPUStat
	for _, k := range keys {
		g := GPUStat{Vendor: "AMD", Name: "AMD " + k}
		for f, v := range data[k] {
			lf := strings.ToLower(f)
			str := fmt.Sprint(v)
			num, _ := strconv.ParseFloat(strings.TrimSpace(str), 64)
			switch {
			case strings.Contains(lf, "card series") || strings.Contains(lf, "card model") && g.Name == "AMD "+k:
				g.Name = str
			case strings.Contains(lf, "gpu use"):
				g.Util = num
			case strings.Contains(lf, "vram total used"):
				g.MemUsed = uint64(num)
			case strings.Contains(lf, "vram total memory"):
				g.MemTotal = uint64(num)
			case strings.Contains(lf, "temperature") && (g.Temp == 0 || strings.Contains(lf, "edge")):
				g.Temp = num
			}
		}
		gs = append(gs, g)
	}
	return gs
}

func nvidiaStats() []GPUStat {
	if nvidiaSMI == "" {
		return nil
	}
	out, err := runTool(nvidiaSMI, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,driver_version,power.draw,power.limit,fan.speed,pstate", "--format=csv,noheader,nounits")
	if err != nil {
		return nil
	}
	var gs []GPUStat
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 10 {
			continue
		}
		num := func(s string) float64 { v, _ := strconv.ParseFloat(strings.TrimSpace(s), 64); return v }
		gs = append(gs, GPUStat{
			Vendor: "NVIDIA", Name: strings.TrimSpace(f[0]), Util: num(f[1]),
			MemUsed: uint64(num(f[2]) * 1024 * 1024), MemTotal: uint64(num(f[3]) * 1024 * 1024), Temp: num(f[4]),
			Driver: strings.TrimSpace(f[5]), CUDA: nvidiaCUDA(), Power: num(f[6]), PowerCap: num(f[7]), Fan: num(f[8]), PState: strings.TrimSpace(f[9]),
		})
	}
	return gs
}

// amdSysfsStats 读取 Linux amdgpu 驱动的 sysfs 信息。
func amdSysfsStats() []GPUStat {
	cards, _ := filepath.Glob("/sys/class/drm/card[0-9]*/device/gpu_busy_percent")
	var gs []GPUStat
	for _, busy := range cards {
		dev := filepath.Dir(busy)
		read := func(name string) float64 {
			b, err := os.ReadFile(filepath.Join(dev, name))
			if err != nil {
				return 0
			}
			v, _ := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
			return v
		}
		g := GPUStat{Vendor: "AMD", Name: "AMD GPU (" + filepath.Base(filepath.Dir(dev)) + ")", Util: read("gpu_busy_percent"),
			MemUsed: uint64(read("mem_info_vram_used")), MemTotal: uint64(read("mem_info_vram_total"))}
		if hw, _ := filepath.Glob(filepath.Join(dev, "hwmon", "hwmon*", "temp1_input")); len(hw) > 0 {
			if b, err := os.ReadFile(hw[0]); err == nil {
				v, _ := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
				g.Temp = v / 1000
			}
		}
		gs = append(gs, g)
	}
	return gs
}
