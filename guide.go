package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// DetectedGPU 是检测到的显卡（用于引导选择下载包）。
type DetectedGPU struct {
	Vendor string `json:"vendor"` // NVIDIA / AMD / Intel / Qualcomm / Apple
	Name   string `json:"name"`
}

var detectGPUs = sync.OnceValue(func() []DetectedGPU {
	var out []DetectedGPU
	seen := map[string]bool{}
	add := func(v, n string) {
		if !seen[v+n] {
			seen[v+n] = true
			out = append(out, DetectedGPU{v, n})
		}
	}
	for _, g := range nvidiaStats() {
		add("NVIDIA", g.Name)
	}
	switch runtime.GOOS {
	case "windows":
		if o, err := runTool("powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"); err == nil {
			for _, n := range strings.Split(o, "\n") {
				if n = strings.TrimSpace(n); n != "" {
					if v := vendorOf(n); v != "" {
						add(v, n)
					}
				}
			}
		}
	case "darwin":
		if runtime.GOARCH == "arm64" {
			add("Apple", "Apple Silicon GPU (Metal)")
		}
	default: // linux / android
		vendors := map[string]string{"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}
		cards, _ := filepath.Glob("/sys/class/drm/card[0-9]*/device/vendor")
		for _, c := range cards {
			if b, err := os.ReadFile(c); err == nil {
				if v := vendors[strings.TrimSpace(string(b))]; v != "" {
					add(v, v+" GPU")
				}
			}
		}
		if _, err := os.Stat("/sys/class/kgsl/kgsl-3d0"); err == nil {
			add("Qualcomm", "Qualcomm Adreno GPU")
		}
	}
	// 同一厂商既有具体型号又有通用名称时去掉通用名称
	var res []DetectedGPU
	for _, g := range out {
		if g.Name == g.Vendor+" GPU" {
			dup := false
			for _, o := range out {
				if o.Vendor == g.Vendor && o.Name != g.Name {
					dup = true
				}
			}
			if dup {
				continue
			}
		}
		res = append(res, g)
	}
	return res
})

func vendorOf(name string) string {
	n := strings.ToLower(name)
	switch {
	case strings.Contains(n, "nvidia") || strings.Contains(n, "geforce") || strings.Contains(n, "quadro") || strings.Contains(n, "rtx"):
		return "NVIDIA"
	case strings.Contains(n, "amd") || strings.Contains(n, "radeon"):
		return "AMD"
	case strings.Contains(n, "intel"):
		return "Intel"
	case strings.Contains(n, "adreno") || strings.Contains(n, "qualcomm"):
		return "Qualcomm"
	}
	return "" // 忽略 Microsoft Basic Display 等虚拟显卡
}

// ---------- GitHub release ----------

type ReleaseAsset struct {
	Name string `json:"name"`
	URL  string `json:"url"`
	Size int64  `json:"size"`
}

type Release struct {
	Tag       string         `json:"tag"`
	URL       string         `json:"url"`
	Published time.Time      `json:"published"`
	Assets    []ReleaseAsset `json:"-"`
}

type releaseCache struct {
	mu  sync.Mutex
	rel *Release
	at  time.Time
}

var relCache releaseCache

func (a *App) latestRelease(refresh bool) (*Release, error) {
	relCache.mu.Lock()
	defer relCache.mu.Unlock()
	if !refresh && relCache.rel != nil && time.Since(relCache.at) < 30*time.Minute {
		return relCache.rel, nil
	}
	body, err := a.info.get(true, "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5")
	if err != nil {
		return nil, fmt.Errorf("获取 GitHub 发布信息失败：%v（可在设置中配置代理）", err)
	}
	var list []struct {
		Tag       string    `json:"tag_name"`
		URL       string    `json:"html_url"`
		Published time.Time `json:"published_at"`
		Assets    []struct {
			Name string `json:"name"`
			URL  string `json:"browser_download_url"`
			Size int64  `json:"size"`
		} `json:"assets"`
	}
	if err := json.Unmarshal([]byte(body), &list); err != nil {
		return nil, err
	}
	for _, r := range list {
		if len(r.Assets) < 5 { // 跳过没有二进制包的发布
			continue
		}
		rel := &Release{Tag: r.Tag, URL: r.URL, Published: r.Published}
		for _, as := range r.Assets {
			rel.Assets = append(rel.Assets, ReleaseAsset{as.Name, as.URL, as.Size})
		}
		relCache.rel, relCache.at = rel, time.Now()
		return rel, nil
	}
	return nil, fmt.Errorf("未找到包含二进制包的发布")
}

// ---------- 推荐 ----------

type Recommend2 struct {
	Kind   string         `json:"kind"` // cpu / gpu
	Title  string         `json:"title"`
	Note   string         `json:"note"`
	Assets []ReleaseAsset `json:"assets"` // 第一个为主包，其余为依赖（如 CUDA 运行库）
	Alt    bool           `json:"alt"`    // 备选方案
}

type GuideInfo struct {
	OS         string        `json:"os"`
	Arch       string        `json:"arch"`
	GPUs       []DetectedGPU `json:"gpus"`
	CUDA       string        `json:"cuda"`
	Release    *Release      `json:"release"`
	Recs       []Recommend2  `json:"recs"`
	Error      string        `json:"error"`
	Servers    ServersInfo   `json:"servers"`
	ModelDirs  int           `json:"modelDirs"`
	ModelCount int           `json:"modelCount"`
}

func (a *App) guide(refresh bool) GuideInfo {
	gi := GuideInfo{OS: runtime.GOOS, Arch: runtime.GOARCH, GPUs: detectGPUs(), CUDA: nvidiaCUDA()}
	gi.Servers = a.servers(false)
	cfg := a.cfg.Get()
	gi.ModelDirs = len(cfg.ModelDirs)
	for _, g := range a.groups() {
		gi.ModelCount += len(g.Variants)
	}
	rel, err := a.latestRelease(refresh)
	if err != nil {
		gi.Error = err.Error()
		return gi
	}
	gi.Release = rel
	gi.Recs = recommendAssets(rel.Assets, gi.OS, gi.Arch, gi.GPUs, gi.CUDA)
	return gi
}

func hasVendor(gs []DetectedGPU, v string) bool {
	for _, g := range gs {
		if g.Vendor == v {
			return true
		}
	}
	return false
}

func findAsset(assets []ReleaseAsset, pattern string) *ReleaseAsset {
	re := regexp.MustCompile(pattern)
	for i := range assets {
		if re.MatchString(assets[i].Name) {
			return &assets[i]
		}
	}
	return nil
}

// pickCUDA 在匹配 pattern（含一个 (\d+\.\d+) 版本分组）的包中，选择不超过驱动支持版本的最高 CUDA 版本；
// 没有时退而选择同一主版本中最低的（CUDA 小版本兼容），ok 表示是否在驱动支持范围内。
func pickCUDA(assets []ReleaseAsset, pattern, maxVer string) (a *ReleaseAsset, ver string, ok bool) {
	re := regexp.MustCompile(pattern)
	type c struct {
		a   *ReleaseAsset
		ver string
		v   float64
	}
	var cs []c
	for i := range assets {
		if m := re.FindStringSubmatch(assets[i].Name); m != nil {
			v, _ := strconv.ParseFloat(m[1], 64)
			cs = append(cs, c{&assets[i], m[1], v})
		}
	}
	if len(cs) == 0 {
		return nil, "", false
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].v > cs[j].v })
	if maxVer == "" {
		return cs[0].a, cs[0].ver, true
	}
	for _, x := range cs {
		if verLE(x.ver, maxVer) {
			return x.a, x.ver, true
		}
	}
	major := strings.Split(maxVer, ".")[0]
	for i := len(cs) - 1; i >= 0; i-- {
		if strings.Split(cs[i].ver, ".")[0] == major {
			return cs[i].a, cs[i].ver, false
		}
	}
	x := cs[len(cs)-1]
	return x.a, x.ver, false
}

// cudaNote 生成 CUDA 包的说明。
func cudaNote(driverCUDA, ver string, ok bool, extra string) string {
	if driverCUDA == "" {
		return extra
	}
	if ok {
		return "驱动支持 CUDA " + driverCUDA + "。" + extra
	}
	return "当前驱动仅支持 CUDA " + driverCUDA + "，没有完全匹配的包；CUDA " + ver + " 在同一主版本内通常可以运行，如启动报错请升级显卡驱动或改用 Vulkan 版。" + extra
}

func verLE(a, b string) bool {
	pa, pb := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < 2; i++ {
		x, y := 0, 0
		if i < len(pa) {
			x, _ = strconv.Atoi(pa[i])
		}
		if i < len(pb) {
			y, _ = strconv.Atoi(pb[i])
		}
		if x != y {
			return x < y
		}
	}
	return true
}

func recommendAssets(assets []ReleaseAsset, goos, arch string, gpus []DetectedGPU, cuda string) []Recommend2 {
	var recs []Recommend2
	add := func(kind, title, note string, alt bool, list ...*ReleaseAsset) {
		var as []ReleaseAsset
		for _, x := range list {
			if x != nil {
				as = append(as, *x)
			}
		}
		if len(as) > 0 {
			recs = append(recs, Recommend2{Kind: kind, Title: title, Note: note, Assets: as, Alt: alt})
		}
	}
	x := map[string]string{"amd64": "x64", "arm64": "arm64"}[arch]
	nv, amd, intel, qc := hasVendor(gpus, "NVIDIA"), hasVendor(gpus, "AMD"), hasVendor(gpus, "Intel"), hasVendor(gpus, "Qualcomm")
	b := `^llama-b\d+-bin-`
	switch goos {
	case "windows":
		add("cpu", "CPU 版", "无独立显卡或仅用 CPU 推理时使用", false, findAsset(assets, b+`win-cpu-`+x+`\.zip$`))
		if nv {
			main, ver, ok := pickCUDA(assets, b+`win-cuda-(\d+\.\d+)-`+x+`\.zip$`, cuda)
			rt := findAsset(assets, `^cudart-llama-(b\d+-)?bin-win-cuda-`+regexp.QuoteMeta(ver)+`-`+x+`\.zip$`)
			add("gpu", "NVIDIA CUDA "+ver+" 版", cudaNote(cuda, ver, ok, "需同时下载 CUDA 运行库（cudart），解压到同一目录"), false, main, rt)
			add("gpu", "Vulkan 版", "兼容性好，无需 CUDA 运行库，速度通常略低于 CUDA", true, findAsset(assets, b+`win-vulkan-`+x+`\.zip$`))
		}
		if amd {
			add("gpu", "Vulkan 版（AMD 推荐）", "适用于大多数 AMD 显卡", false, findAsset(assets, b+`win-vulkan-`+x+`\.zip$`))
			add("gpu", "ROCm/HIP 版", "仅支持 ROCm 覆盖的较新 Radeon 显卡", true, findAsset(assets, b+`win-(rocm|hip)[^/]*-`+x+`\.zip$`))
		}
		if intel && !nv && !amd {
			add("gpu", "Vulkan 版（Intel）", "适用于 Intel 核显/Arc 独显", false, findAsset(assets, b+`win-vulkan-`+x+`\.zip$`))
			add("gpu", "SYCL 版（Intel Arc）", "Intel Arc 独显可获得更好性能，需安装 oneAPI 运行库", true, findAsset(assets, b+`win-sycl-`+x+`\.zip$`))
		}
		if qc {
			add("gpu", "OpenCL Adreno 版", "适用于骁龙 Windows 设备", false, findAsset(assets, b+`win-opencl-adreno-arm64\.zip$`))
		}
	case "linux":
		cpu := findAsset(assets, b+`ubuntu-`+x+`\.tar\.gz$`)
		add("cpu", "CPU 版", "Ubuntu 编译，其它发行版如缺少依赖请自行编译", false, cpu)
		if nv {
			main, ver, ok := pickCUDA(assets, b+`ubuntu-cuda-(\d+\.\d+)-`+x+`\.tar\.gz$`, cuda)
			rt := findAsset(assets, `^cudart-llama-(b\d+-)?bin-ubuntu-cuda-`+regexp.QuoteMeta(ver)+`-`+x+`\.tar\.gz$`)
			add("gpu", "NVIDIA CUDA "+ver+" 版", cudaNote(cuda, ver, ok, "未安装 CUDA 运行库时请同时下载 cudart 包"), false, main, rt)
			add("gpu", "Vulkan 版", "无需 CUDA 运行库", true, findAsset(assets, b+`ubuntu-vulkan-`+x+`\.tar\.gz$`))
		}
		if amd {
			add("gpu", "ROCm 版", "需安装 ROCm；不支持的显卡请用 Vulkan 版", false, findAsset(assets, b+`ubuntu-rocm-[\d.]+-`+x+`\.tar\.gz$`))
			add("gpu", "Vulkan 版", "适用于大多数 AMD 显卡", true, findAsset(assets, b+`ubuntu-vulkan-`+x+`\.tar\.gz$`))
		}
		if intel && !nv && !amd {
			add("gpu", "Vulkan 版（Intel）", "", false, findAsset(assets, b+`ubuntu-vulkan-`+x+`\.tar\.gz$`))
			add("gpu", "SYCL 版（Intel Arc）", "需安装 oneAPI 运行库", true, findAsset(assets, b+`ubuntu-sycl-fp16-`+x+`\.tar\.gz$`))
		}
		if qc {
			add("gpu", "Snapdragon 版", "适用于骁龙 Linux 设备", false, findAsset(assets, b+`linux-arm64-snapdragon\.tar\.gz$`))
		}
	case "android":
		add("cpu", "Android 版", "在 Termux 中解压运行", false, findAsset(assets, b+`android-arm64\.tar\.gz$`))
		if qc {
			add("gpu", "Android Snapdragon 版", "使用 Adreno GPU / Hexagon 加速，适用于骁龙设备", false, findAsset(assets, b+`android-arm64-snapdragon\.tar\.gz$`))
		}
	case "darwin":
		add("gpu", "macOS 版", "同一程序同时支持 Metal GPU 与 CPU，只需配置 GPU 版路径；首次运行如被系统拦截，请在“隐私与安全性”中允许", false,
			findAsset(assets, b+`macos-`+x+`\.tar\.gz$`))
	}
	return recs
}
