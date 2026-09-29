package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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
	Name     string  `json:"name"`
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

type SysMonitor struct {
	mu    sync.RWMutex
	stat  SysStat
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
		s.mu.Lock()
		s.stat = st
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

// GPUInfo 返回显卡工具的原始输出（nvidia-smi / rocm-smi / amd-smi），都不存在时 Tool 为空。
type GPUInfo struct {
	Tool   string `json:"tool"`
	Output string `json:"output"`
}

func GetGPUInfo() GPUInfo {
	switch {
	case nvidiaSMI != "":
		out, _ := runTool(nvidiaSMI)
		return GPUInfo{"nvidia-smi", out}
	case rocmSMI != "":
		out, _ := runTool(rocmSMI)
		return GPUInfo{"rocm-smi", out}
	case amdSMI != "":
		out, _ := runTool(amdSMI, "list")
		return GPUInfo{"amd-smi", out}
	}
	return GPUInfo{}
}

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
		g := GPUStat{Name: "AMD " + k}
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
	out, err := runTool(nvidiaSMI, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits")
	if err != nil {
		return nil
	}
	var gs []GPUStat
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 5 {
			continue
		}
		num := func(s string) float64 { v, _ := strconv.ParseFloat(strings.TrimSpace(s), 64); return v }
		gs = append(gs, GPUStat{
			Name: strings.TrimSpace(f[0]), Util: num(f[1]),
			MemUsed: uint64(num(f[2]) * 1024 * 1024), MemTotal: uint64(num(f[3]) * 1024 * 1024), Temp: num(f[4]),
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
		g := GPUStat{Name: "AMD GPU (" + filepath.Base(filepath.Dir(dev)) + ")", Util: read("gpu_busy_percent"),
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
