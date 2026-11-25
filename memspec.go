package main

import (
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// MemSpec 是本机内存规格（尽力检测，取不到的项为空/0），用于推荐配置估算内存带宽。
type MemSpec struct {
	Type     string  `json:"type"`     // ddr3 / ddr4 / ddr5 / lpddr4x / lpddr5 / lpddr5x / apple
	Speed    int     `json:"speed"`    // MT/s
	Modules  int     `json:"modules"`  // 已安装的内存条数
	Channels int     `json:"channels"` // 推测的通道数
	BW       float64 `json:"bw"`       // 已知带宽 GB/s（Apple 芯片按型号）
	Chip     string  `json:"chip"`     // Apple 芯片型号
}

// SMBIOS 内存类型代码。
var smbiosMemType = map[int]string{24: "ddr3", 26: "ddr4", 29: "lpddr3", 30: "lpddr4", 34: "ddr5", 35: "lpddr5"}

// appleBW 是 Apple 芯片的内存带宽（GB/s），按名称最长匹配。
var appleBW = []struct {
	name string
	bw   float64
}{
	{"M1 Ultra", 800}, {"M2 Ultra", 800}, {"M3 Ultra", 819},
	{"M1 Max", 400}, {"M2 Max", 400}, {"M3 Max", 400}, {"M4 Max", 546},
	{"M1 Pro", 200}, {"M2 Pro", 200}, {"M3 Pro", 150}, {"M4 Pro", 273},
	{"M1", 68}, {"M2", 100}, {"M3", 100}, {"M4", 120}, {"M5", 153},
}

// AppleBandwidth 按芯片名称返回带宽，未知返回 0。
func AppleBandwidth(chip string) float64 {
	for _, a := range appleBW {
		if strings.Contains(chip, a.name) {
			return a.bw
		}
	}
	return 0
}

var detectMemSpec = sync.OnceValue(func() MemSpec {
	var m MemSpec
	switch runtime.GOOS {
	case "windows":
		o, err := runTool("powershell.exe", "-NoProfile", "-Command",
			`Get-CimInstance Win32_PhysicalMemory | ForEach-Object { "$($_.SMBIOSMemoryType),$($_.ConfiguredClockSpeed),$($_.Speed)" }`)
		if err == nil {
			m = parseWinMem(o)
		}
	case "darwin":
		chip := hostInfo().CPUModel
		if strings.Contains(chip, "Apple") {
			m = MemSpec{Type: "apple", Chip: chip, BW: AppleBandwidth(chip)}
		}
	default:
		// dmidecode 需要 root 权限，普通用户取不到时留空，由用户选择
		if bin := lookTool("dmidecode"); bin != "" {
			if o, err := runTool(bin, "-t", "17"); err == nil {
				m = parseDMIMem(o)
			}
		}
	}
	if m.Type != "apple" && m.Modules > 0 {
		m.Channels = min(m.Modules, 2) // 消费级平台多为双通道；4 条内存通常仍是双通道
		if m.Modules >= 8 {
			m.Channels = 4
		}
	}
	return m
})

// memMinMTs 是各类内存的最低传输率；低于它说明读到的是时钟频率（MHz），需乘 2。
var memMinMTs = map[string]int{"ddr3": 800, "ddr4": 1600, "ddr5": 3200, "lpddr3": 1066, "lpddr4": 1600, "lpddr4x": 1600, "lpddr5": 3200, "lpddr5x": 4800}

func fixMTs(typ string, v int) int {
	if min := memMinMTs[typ]; v > 0 && v < min && v*2 >= min {
		return v * 2
	}
	return v
}

// parseWinMem 解析 “SMBIOS 类型,配置频率,标称频率” 每行一条内存。
func parseWinMem(o string) MemSpec {
	var m MemSpec
	for _, line := range strings.Split(o, "\n") {
		f := strings.Split(strings.TrimSpace(line), ",")
		if len(f) < 3 {
			continue
		}
		t, _ := strconv.Atoi(f[0])
		cfg, _ := strconv.Atoi(f[1])
		spd, _ := strconv.Atoi(f[2])
		m.Modules++
		if s := smbiosMemType[t]; s != "" {
			m.Type = s
		}
		if cfg == 0 {
			cfg = spd
		}
		// 部分机器（常见于 DDR5）报告的是时钟频率 MHz，DDR 的传输率 MT/s 是它的 2 倍
		if cfg > 0 && spd >= cfg*19/10 {
			cfg *= 2
		}
		cfg = fixMTs(smbiosMemType[t], cfg)
		if cfg > 0 && (m.Speed == 0 || cfg < m.Speed) {
			m.Speed = cfg // 多条内存按最低频率运行
		}
	}
	return m
}

var (
	dmiTypeRe  = regexp.MustCompile(`(?m)^\s*Type:\s*(\S+)`)
	dmiSpeedRe = regexp.MustCompile(`(?m)^\s*(Configured Memory Speed|Configured Clock Speed|Speed):\s*(\d+)\s*MT/s`)
)

// parseDMIMem 解析 dmidecode -t 17 的输出。
func parseDMIMem(o string) MemSpec {
	var m MemSpec
	for _, dev := range strings.Split(o, "Memory Device")[1:] {
		if strings.Contains(dev, "No Module Installed") || strings.Contains(dev, "Size: 0") {
			continue
		}
		m.Modules++
		if mm := dmiTypeRe.FindStringSubmatch(dev); mm != nil {
			if t := strings.ToLower(mm[1]); strings.Contains(t, "ddr") {
				m.Type = t
			}
		}
		spd := 0
		for _, mm := range dmiSpeedRe.FindAllStringSubmatch(dev, -1) {
			v, _ := strconv.Atoi(mm[2])
			if v > 0 && (spd == 0 || strings.HasPrefix(mm[1], "Configured")) {
				spd = v
			}
		}
		spd = fixMTs(m.Type, spd)
		if spd > 0 && (m.Speed == 0 || spd < m.Speed) {
			m.Speed = spd
		}
	}
	return m
}
