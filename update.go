package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------- llama.cpp 更新检测 ----------

// UpdateTarget 是一个可更新的 llama-server（GPU 版 / CPU 版）。
type UpdateTarget struct {
	Key        string         `json:"key"` // gpu / cpu
	Label      string         `json:"label"`
	Path       string         `json:"path"`
	Dir        string         `json:"dir"`
	Version    string         `json:"version"`
	Build      int            `json:"build"` // 当前构建号，0 表示未知
	HasUpdate  bool           `json:"hasUpdate"`
	CanUpdate  bool           `json:"canUpdate"`
	Reason     string         `json:"reason"`  // 不能自动更新的原因
	Backend    string         `json:"backend"` // 识别到的包类型
	Asset      *ReleaseAsset  `json:"asset"`   // 推荐的更新包
	Deps       []ReleaseAsset `json:"deps"`    // 需要同时下载的依赖（如 CUDA 运行库）
	Candidates []string       `json:"candidates"`
	SameAs     string         `json:"sameAs"` // 与另一个目标是同一目录时为其 key
}

type UpdateCheck struct {
	Release     *Release       `json:"release"`
	LatestBuild int            `json:"latestBuild"`
	Targets     []UpdateTarget `json:"targets"`
	HasUpdate   bool           `json:"hasUpdate"`
	Error       string         `json:"error"`
	Task        UpdateTask     `json:"task"`
	CheckedAt   time.Time      `json:"checkedAt"`
}

const reasonNoAsset = "未找到与当前程序对应的发布包，请手动选择"

var (
	buildRe    = regexp.MustCompile(`\bbuild\s+(\d+)`)
	buildOldRe = regexp.MustCompile(`^\s*b?(\d+)(?:\s|\(|$)`)
)

// parseBuild 从 `--version` 的版本号或发布标签中取构建号：
// 新版 "1.0.0-dev (build 12602, commit …)"，旧版 "6500 (abc1234)"，标签 "b6500"。
func parseBuild(s string) int {
	m := buildRe.FindStringSubmatch(s)
	if m == nil {
		m = buildOldRe.FindStringSubmatch(s)
	}
	if m != nil {
		n, _ := strconv.Atoi(m[1])
		return n
	}
	return 0
}

// assetFamily 去掉发布包名中的构建号，用于跨版本匹配同一种包。
var familyRe = regexp.MustCompile(`-b\d+-`)

func assetFamily(name string) string { return familyRe.ReplaceAllString(name, "-") }

// detectBackend 根据程序目录中的文件识别发布包类型。
func detectBackend(files []string, goos string) string {
	has := func(s string) bool {
		for _, f := range files {
			if strings.Contains(strings.ToLower(f), s) {
				return true
			}
		}
		return false
	}
	switch {
	case goos == "darwin":
		return "macos"
	case has("ggml-cuda"):
		return "cuda"
	case has("ggml-hip"), has("ggml-rocm"):
		return "rocm"
	case has("ggml-sycl"):
		return "sycl"
	case has("ggml-openvino"):
		return "openvino"
	case has("ggml-hexagon"), has("snapdragon"):
		return "snapdragon"
	case has("ggml-opencl"):
		return "opencl"
	case has("ggml-vulkan"):
		return "vulkan"
	}
	return "cpu"
}

var cudaLibRe = regexp.MustCompile(`(?i)^(?:cudart64_|cublas64_|libcudart\.so\.|libcublas\.so\.)(\d+)`)

// cudaMajorOf 从目录中的 CUDA 运行库文件名识别主版本号（如 cudart64_12.dll -> 12）。
func cudaMajorOf(files []string) string {
	for _, f := range files {
		if m := cudaLibRe.FindStringSubmatch(f); m != nil {
			return m[1]
		}
	}
	return ""
}

// backendPattern 返回某种包在当前系统上的资产名正则；CUDA 包含一个版本号分组。
func backendPattern(backend, goos, arch string) string {
	x := map[string]string{"amd64": "x64", "arm64": "arm64"}[arch]
	b := `^llama-b\d+-bin-`
	pick := func(m map[string]string, suffix string) string {
		if v, ok := m[backend]; ok {
			return b + v + suffix
		}
		return ""
	}
	switch goos {
	case "windows":
		return pick(map[string]string{
			"cpu": `win-cpu-` + x, "cuda": `win-cuda-(\d+\.\d+)-` + x, "vulkan": `win-vulkan-` + x,
			"rocm": `win-(?:rocm|hip)[^/]*-` + x, "sycl": `win-sycl-` + x, "opencl": `win-opencl-adreno-` + x,
			"openvino": `win-openvino-[\d.]+-` + x,
		}, `\.zip$`)
	case "linux":
		return pick(map[string]string{
			"cpu": `ubuntu-` + x, "cuda": `ubuntu-cuda-(\d+\.\d+)-` + x, "vulkan": `ubuntu-vulkan-` + x,
			"rocm": `ubuntu-rocm-[\d.]+-` + x, "sycl": `ubuntu-sycl-fp16-` + x, "openvino": `ubuntu-openvino-[\d.]+-` + x,
			"snapdragon": `linux-arm64-snapdragon`,
		}, `\.tar\.gz$`)
	case "android":
		if backend == "snapdragon" || backend == "opencl" {
			return b + `android-arm64-snapdragon\.tar\.gz$`
		}
		return b + `android-arm64\.tar\.gz$`
	case "darwin":
		return b + `macos-` + x + `\.tar\.gz$`
	}
	return ""
}

// pickUpdateAsset 为已安装的程序选择更新包：优先使用上次安装记录的同一种包，
// 否则按目录中的文件识别包类型；CUDA 包尽量保持已安装的 CUDA 主版本。
func pickUpdateAsset(assets []ReleaseAsset, files []string, goos, arch, driverCUDA, installed string) (main *ReleaseAsset, backend string) {
	backend = detectBackend(files, goos)
	if installed != "" {
		for i := range assets {
			if assetFamily(assets[i].Name) == installed {
				return &assets[i], backend
			}
		}
	}
	pat := backendPattern(backend, goos, arch)
	if pat == "" {
		return nil, backend
	}
	if backend != "cuda" {
		return findAsset(assets, pat), backend
	}
	// 同一 CUDA 主版本内选不超过驱动支持版本的最高包
	if major := cudaMajorOf(files); major != "" {
		re := regexp.MustCompile(pat)
		var same []ReleaseAsset
		for _, a := range assets {
			if m := re.FindStringSubmatch(a.Name); m != nil && strings.Split(m[1], ".")[0] == major {
				same = append(same, a)
			}
		}
		if len(same) > 0 {
			a, _, _ := pickCUDA(same, pat, driverCUDA)
			return a, backend
		}
	}
	a, _, _ := pickCUDA(assets, pat, driverCUDA)
	return a, backend
}

// cudaDeps 返回 CUDA 包需要的运行库包（目录中已有同一主版本的运行库时不需要）。
func cudaDeps(assets []ReleaseAsset, main *ReleaseAsset, files []string, goos string) []ReleaseAsset {
	m := regexp.MustCompile(`-(win|ubuntu)-cuda-(\d+\.\d+)-(\w+)\.(zip|tar\.gz)$`).FindStringSubmatch(main.Name)
	if m == nil {
		return nil
	}
	if cudaMajorOf(files) == strings.Split(m[2], ".")[0] {
		return nil
	}
	rt := findAsset(assets, `^cudart-llama-(b\d+-)?bin-`+m[1]+`-cuda-`+regexp.QuoteMeta(m[2])+`-`+m[3]+`\.`+regexp.QuoteMeta(m[4])+`$`)
	if rt == nil {
		return nil
	}
	return []ReleaseAsset{*rt}
}

// candidateAssets 列出当前系统与架构可用的全部主程序包，供手动选择。
func candidateAssets(assets []ReleaseAsset, goos, arch string) []string {
	x := map[string]string{"amd64": "x64", "arm64": "arm64"}[arch]
	osKey := map[string][]string{"windows": {"-win-"}, "linux": {"-ubuntu-", "-linux-"}, "android": {"-android-"}, "darwin": {"-macos-"}}[goos]
	var out []string
	for _, a := range assets {
		if !strings.HasPrefix(a.Name, "llama-b") || !strings.Contains(a.Name, "-bin-") {
			continue
		}
		okOS := false
		for _, k := range osKey {
			okOS = okOS || strings.Contains(a.Name, k)
		}
		if okOS && (x == "" || strings.Contains(a.Name, x)) {
			out = append(out, a.Name)
		}
	}
	sort.Strings(out)
	return out
}

func listFiles(dir string) []string {
	es, _ := os.ReadDir(dir)
	out := make([]string, 0, len(es))
	for _, e := range es {
		out = append(out, e.Name())
	}
	return out
}

// dirWritable 检查目录是否可写。
func dirWritable(dir string) bool {
	f, err := os.CreateTemp(dir, ".llamaRuns-w-*")
	if err != nil {
		return false
	}
	f.Close()
	os.Remove(f.Name())
	return true
}

func (a *App) checkUpdate(refresh bool) UpdateCheck {
	uc := UpdateCheck{Task: a.upd.Status(), CheckedAt: time.Now()}
	rel, err := a.latestRelease(refresh)
	if err != nil {
		uc.Error = err.Error()
	} else {
		uc.Release = rel
		uc.LatestBuild = parseBuild(rel.Tag)
	}
	cfg := a.cfg.Get()
	sv := a.servers(refresh)
	add := func(key, label, conf string, si ServerInfo) {
		t := UpdateTarget{Key: key, Label: label, Path: si.Path, Version: si.Version, Build: parseBuild(si.Version)}
		switch {
		case si.Path == "":
			t.Reason = "未找到程序"
		case strings.TrimSpace(conf) == "":
			t.Reason = "程序来自系统 PATH，请先在设置中配置程序目录"
		}
		if si.Path != "" {
			if abs, err := filepath.Abs(si.Path); err == nil {
				t.Path = abs
			}
			t.Dir = filepath.Dir(t.Path)
		}
		for _, o := range uc.Targets {
			if o.Dir != "" && sameDir(o.Dir, t.Dir) {
				t.SameAs = o.Key
			}
		}
		if rel != nil && t.Dir != "" {
			files := listFiles(t.Dir)
			t.Asset, t.Backend = pickUpdateAsset(rel.Assets, files, runtime.GOOS, runtime.GOARCH, nvidiaCUDA(), cfg.Installed[t.Dir])
			if t.Asset != nil {
				t.Deps = cudaDeps(rel.Assets, t.Asset, files, runtime.GOOS)
			}
			t.Candidates = candidateAssets(rel.Assets, runtime.GOOS, runtime.GOARCH)
			t.HasUpdate = t.Build == 0 || t.Build < uc.LatestBuild
			if t.Reason == "" && t.Asset == nil {
				t.Reason = reasonNoAsset
			}
			if t.Reason == "" && !dirWritable(t.Dir) {
				t.Reason = "程序目录不可写：" + t.Dir
			}
		}
		t.CanUpdate = (t.Reason == "" || t.Reason == reasonNoAsset) && t.SameAs == "" && rel != nil
		if t.HasUpdate && t.Path != "" && t.SameAs == "" {
			uc.HasUpdate = true
		}
		uc.Targets = append(uc.Targets, t)
	}
	add("gpu", "GPU 版", cfg.ServerPathGPU, sv.GPU)
	if sv.CPU != nil {
		add("cpu", "CPU 版", cfg.ServerPathCPU, *sv.CPU)
	}
	return uc
}

// ---------- 更新任务 ----------

type UpdateTask struct {
	Running  bool      `json:"running"`
	Tag      string    `json:"tag"`
	Step     string    `json:"step"`
	Done     int64     `json:"done"`  // 已下载字节
	Total    int64     `json:"total"` // 需下载字节
	Log      []string  `json:"log"`
	Error    string    `json:"error"`
	OK       bool      `json:"ok"`
	Started  time.Time `json:"started"`
	Finished time.Time `json:"finished"`
}

type Updater struct {
	mu     sync.Mutex
	task   UpdateTask
	cancel context.CancelFunc
}

func (u *Updater) Status() UpdateTask {
	u.mu.Lock()
	defer u.mu.Unlock()
	t := u.task
	t.Log = append([]string(nil), u.task.Log...)
	return t
}

func (u *Updater) logf(format string, args ...any) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.task.Log = append(u.task.Log, time.Now().Format("15:04:05 ")+fmt.Sprintf(format, args...))
}

func (u *Updater) step(s string) {
	u.mu.Lock()
	u.task.Step = s
	u.mu.Unlock()
	u.logf("%s", s)
}

func (u *Updater) Cancel() bool {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.cancel != nil && u.task.Running {
		u.cancel()
		return true
	}
	return false
}

// updateJob 是一个目录的更新内容。
type updateJob struct {
	target UpdateTarget
	main   ReleaseAsset
	deps   []ReleaseAsset
}

// StartUpdate 在后台更新指定的程序（keys 为 gpu / cpu）；assets 可为某个目标指定主程序包名。
func (a *App) StartUpdate(keys []string, assets map[string]string) error {
	uc := a.checkUpdate(false)
	if uc.Release == nil {
		return errors.New(uc.Error)
	}
	var jobs []updateJob
	for _, t := range uc.Targets {
		if !contains(keys, t.Key) {
			continue
		}
		if t.SameAs != "" {
			continue // 与另一个目标同目录，只更新一次
		}
		if t.Reason != "" && !(t.Reason == reasonNoAsset && assets[t.Key] != "") {
			return fmt.Errorf("%s 不能自动更新：%s", t.Label, t.Reason)
		}
		main := t.Asset
		deps := t.Deps
		if n := assets[t.Key]; n != "" && (main == nil || n != main.Name) {
			main = nil
			for i := range uc.Release.Assets {
				if uc.Release.Assets[i].Name == n {
					main = &uc.Release.Assets[i]
				}
			}
			if main == nil {
				return fmt.Errorf("发布 %s 中没有 %s", uc.Release.Tag, n)
			}
			deps = cudaDeps(uc.Release.Assets, main, listFiles(t.Dir), runtime.GOOS)
		}
		if main == nil {
			return fmt.Errorf("%s 未找到可用的更新包", t.Label)
		}
		jobs = append(jobs, updateJob{t, *main, deps})
	}
	if len(jobs) == 0 {
		return errors.New("没有需要更新的程序")
	}
	u := a.upd
	u.mu.Lock()
	if u.task.Running {
		u.mu.Unlock()
		return errors.New("更新正在进行中")
	}
	ctx, cancel := context.WithCancel(context.Background())
	u.cancel = cancel
	u.task = UpdateTask{Running: true, Tag: uc.Release.Tag, Started: time.Now()}
	for _, j := range jobs {
		u.task.Total += j.main.Size
		for _, d := range j.deps {
			u.task.Total += d.Size
		}
	}
	u.mu.Unlock()

	go func() {
		defer cancel()
		err := a.runUpdate(ctx, uc.Release.Tag, jobs)
		u.mu.Lock()
		u.task.Running, u.task.Finished = false, time.Now()
		if err != nil {
			u.task.Error = err.Error()
			if errors.Is(err, context.Canceled) {
				u.task.Error = "已取消"
			}
			u.task.Step = "更新失败"
		} else {
			u.task.OK, u.task.Step = true, "更新完成"
		}
		u.mu.Unlock()
		if err != nil {
			u.logf("✘ %s", u.Status().Error)
		} else {
			u.logf("✔ 已更新到 %s", uc.Release.Tag)
		}
	}()
	return nil
}

func (a *App) runUpdate(ctx context.Context, tag string, jobs []updateJob) error {
	u := a.upd
	cacheDir := filepath.Join(a.cfg.Dir(), "cache", "update")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return err
	}
	// 先下载全部文件，避免停止实例后才发现下载失败
	for _, j := range jobs {
		for _, as := range append([]ReleaseAsset{j.main}, j.deps...) {
			u.step("下载 " + as.Name)
			if err := a.download(ctx, as, filepath.Join(cacheDir, as.Name)); err != nil {
				if ctx.Err() != nil {
					return ctx.Err()
				}
				return fmt.Errorf("下载 %s 失败：%w（可在设置中配置 GitHub 代理）", as.Name, err)
			}
		}
	}
	for _, j := range jobs {
		if err := a.installJob(ctx, tag, j, cacheDir); err != nil {
			return fmt.Errorf("%s：%w", j.target.Label, err)
		}
	}
	// 清理下载的安装包
	for _, j := range jobs {
		for _, as := range append([]ReleaseAsset{j.main}, j.deps...) {
			os.Remove(filepath.Join(cacheDir, as.Name))
		}
	}
	return nil
}

// installJob 解压、替换一个目录中的程序并验证，失败时回滚；替换期间停止并在之后重启使用该程序的实例。
func (a *App) installJob(ctx context.Context, tag string, j updateJob, cacheDir string) error {
	u := a.upd
	dir := j.target.Dir
	stage := filepath.Join(dir, ".llamaRuns-new")
	backup := filepath.Join(dir, ".llamaRuns-old")
	os.RemoveAll(stage)
	os.RemoveAll(backup)
	defer os.RemoveAll(stage)

	u.step("解压 " + j.main.Name)
	if err := extractArchive(filepath.Join(cacheDir, j.main.Name), stage, true); err != nil {
		return fmt.Errorf("解压失败：%w", err)
	}
	for _, d := range j.deps {
		u.step("解压 " + d.Name)
		if err := extractArchive(filepath.Join(cacheDir, d.Name), stage, false); err != nil {
			return fmt.Errorf("解压失败：%w", err)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	specs := a.mgr.StopByDir(dir)
	if len(specs) > 0 {
		u.logf("已停止 %d 个使用该程序的实例", len(specs))
	}
	restart := func() {
		for _, sp := range specs {
			if _, err := a.mgr.Relaunch(sp); err != nil {
				u.logf("重新启动 %s 失败：%v", sp.g.Name, err)
			} else {
				u.logf("已重新启动 %s", sp.g.Name)
			}
		}
	}
	u.step("替换 " + dir + " 中的程序")
	moved, err := replaceFiles(stage, dir, backup)
	if err != nil {
		rollback(moved, dir, backup)
		restart()
		return fmt.Errorf("替换文件失败，已还原：%w", err)
	}
	u.step("验证新版本")
	si := ProbeServer(j.target.Path)
	if si.Error != "" || parseBuild(si.Version) == 0 {
		rollback(moved, dir, backup)
		a.probeCached(a.confPath(j.target.Key), true)
		restart()
		msg := si.Error
		if msg == "" {
			msg = "无法识别版本"
		}
		return fmt.Errorf("新版本无法运行（%s），已还原旧版本", msg)
	}
	u.logf("新版本：%s", si.Version)
	if err := os.RemoveAll(backup); err != nil {
		u.logf("旧文件备份未能删除（可能正被占用）：%s", backup)
	}
	_ = a.cfg.SetInstalled(dir, assetFamily(j.main.Name))
	// 刷新探测缓存（同目录的另一个目标也一并刷新）
	a.probeCached(a.confPath("gpu"), true)
	if c := a.confPath("cpu"); strings.TrimSpace(c) != "" {
		a.probeCached(c, true)
	}
	restart()
	return nil
}

func (a *App) confPath(key string) string {
	cfg := a.cfg.Get()
	if key == "cpu" {
		return cfg.ServerPathCPU
	}
	return cfg.ServerPathGPU
}

// download 下载发布包到 dst（已存在且校验通过时跳过），校验 sha256。
func (a *App) download(ctx context.Context, as ReleaseAsset, dst string) error {
	u := a.upd
	want := strings.TrimPrefix(as.Digest, "sha256:")
	if st, err := os.Stat(dst); err == nil && st.Size() == as.Size {
		if want == "" || fileSHA256(dst) == want {
			u.addDone(as.Size)
			u.logf("使用已下载的 %s", as.Name)
			return nil
		}
	}
	c, err := a.info.clientWith(proxyFor(a.cfg.Get(), netGitHub), 0)
	if err != nil {
		return err
	}
	// 长时间无数据时中断下载
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", as.URL, nil)
	resp, err := c.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	h := sha256.New()
	last := time.Now()
	var lmu sync.Mutex
	stalled := false
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		t := time.NewTicker(5 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				lmu.Lock()
				idle := time.Since(last)
				lmu.Unlock()
				if idle > 60*time.Second {
					lmu.Lock()
					stalled = true
					lmu.Unlock()
					cancel()
					return
				}
			}
		}
	}()
	buf := make([]byte, 256<<10)
	var n int64
	for {
		k, rerr := resp.Body.Read(buf)
		if k > 0 {
			if _, err := f.Write(buf[:k]); err != nil {
				f.Close()
				return err
			}
			h.Write(buf[:k])
			n += int64(k)
			u.addDone(int64(k))
			lmu.Lock()
			last = time.Now()
			lmu.Unlock()
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Close()
			os.Remove(tmp)
			u.addDone(-n)
			lmu.Lock()
			st := stalled
			lmu.Unlock()
			if st {
				return errors.New("超过 60 秒没有收到数据")
			}
			return rerr
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	if as.Size > 0 && n != as.Size {
		os.Remove(tmp)
		u.addDone(-n)
		return fmt.Errorf("文件大小不符（%d / %d）", n, as.Size)
	}
	if want != "" && hex.EncodeToString(h.Sum(nil)) != want {
		os.Remove(tmp)
		u.addDone(-n)
		return errors.New("sha256 校验失败")
	}
	return os.Rename(tmp, dst)
}

func (u *Updater) addDone(n int64) {
	u.mu.Lock()
	u.task.Done += n
	u.mu.Unlock()
}

func fileSHA256(p string) string {
	f, err := os.Open(p)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ---------- 解压与替换 ----------

// archiveRoot 决定解压时去掉的前缀：主程序包取 llama-server 所在目录；
// 其它包所有文件位于同一顶层目录时去掉该目录。
func archiveRoot(names []string, main bool) (string, error) {
	if main {
		for _, n := range names {
			b := path.Base(n)
			if b == "llama-server" || b == "llama-server.exe" {
				return path.Dir(n), nil
			}
		}
		return "", errors.New("安装包中没有 llama-server")
	}
	top := ""
	for i, n := range names {
		seg := strings.SplitN(strings.TrimPrefix(n, "./"), "/", 2)
		if len(seg) < 2 {
			return "", nil
		}
		if i == 0 {
			top = seg[0]
		} else if seg[0] != top {
			return "", nil
		}
	}
	return top, nil
}

// relUnder 返回 name 去掉 root 前缀后的相对路径；不在 root 下或含非法路径时返回空。
func relUnder(name, root string) string {
	name = strings.TrimPrefix(path.Clean("/"+strings.ReplaceAll(name, `\`, "/")), "/")
	root = strings.TrimPrefix(path.Clean("/"+root), "/")
	if root != "" {
		if !strings.HasPrefix(name, root+"/") {
			return ""
		}
		name = strings.TrimPrefix(name, root+"/")
	}
	if name == "" || name == "." || strings.HasPrefix(name, "../") {
		return ""
	}
	return name
}

// extractArchive 把 zip / tar.gz 解压到 dst。
func extractArchive(file, dst string, main bool) error {
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	if strings.HasSuffix(file, ".zip") {
		zr, err := zip.OpenReader(file)
		if err != nil {
			return err
		}
		defer zr.Close()
		var names []string
		for _, f := range zr.File {
			if !f.FileInfo().IsDir() {
				names = append(names, f.Name)
			}
		}
		root, err := archiveRoot(names, main)
		if err != nil {
			return err
		}
		for _, f := range zr.File {
			rel := relUnder(f.Name, root)
			if rel == "" || f.FileInfo().IsDir() {
				continue
			}
			if err := writeEntry(dst, rel, f.Mode(), f.Open); err != nil {
				return err
			}
		}
		return nil
	}
	// tar.gz 只能顺序读取：先读一遍取文件列表决定前缀，再读一遍解压
	names, err := tarNames(file)
	if err != nil {
		return err
	}
	root, err := archiveRoot(names, main)
	if err != nil {
		return err
	}
	f, err := os.Open(file)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		rel := relUnder(h.Name, root)
		if rel == "" {
			continue
		}
		switch h.Typeflag {
		case tar.TypeReg:
			if err := writeEntry(dst, rel, os.FileMode(h.Mode), func() (io.ReadCloser, error) { return io.NopCloser(tr), nil }); err != nil {
				return err
			}
		case tar.TypeSymlink:
			if strings.Contains(h.Linkname, "..") || filepath.IsAbs(h.Linkname) {
				continue // 只接受同目录内的链接
			}
			p := filepath.Join(dst, filepath.FromSlash(rel))
			_ = os.MkdirAll(filepath.Dir(p), 0o755)
			os.Remove(p)
			if err := os.Symlink(h.Linkname, p); err != nil {
				return err
			}
		}
	}
}

func tarNames(file string) ([]string, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	tr := tar.NewReader(gz)
	var names []string
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return names, nil
		}
		if err != nil {
			return nil, err
		}
		if h.Typeflag == tar.TypeReg || h.Typeflag == tar.TypeSymlink {
			names = append(names, h.Name)
		}
	}
}

func writeEntry(dst, rel string, mode os.FileMode, open func() (io.ReadCloser, error)) error {
	p := filepath.Join(dst, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	r, err := open()
	if err != nil {
		return err
	}
	defer r.Close()
	perm := mode.Perm()
	if perm == 0 {
		perm = 0o644
	}
	if perm&0o400 == 0 {
		perm |= 0o600
	}
	w, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
	if err != nil {
		return err
	}
	if _, err := io.Copy(w, r); err != nil {
		w.Close()
		return err
	}
	return w.Close()
}

// replaceFiles 把 stage 中的文件移动到 dir，原有同名文件先移到 backup；返回已替换的相对路径。
// 用重命名而非覆盖：Windows 下正在使用的文件不能覆盖，但通常可以重命名。
func replaceFiles(stage, dir, backup string) ([]string, error) {
	var moved []string
	err := filepath.Walk(stage, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(stage, p)
		dest := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return err
		}
		if _, err := os.Lstat(dest); err == nil {
			b := filepath.Join(backup, rel)
			if err := os.MkdirAll(filepath.Dir(b), 0o755); err != nil {
				return err
			}
			if err := os.Rename(dest, b); err != nil {
				return err
			}
		}
		moved = append(moved, rel)
		return os.Rename(p, dest)
	})
	return moved, err
}

// rollback 还原 replaceFiles 替换过的文件。
func rollback(moved []string, dir, backup string) {
	for i := len(moved) - 1; i >= 0; i-- {
		rel := moved[i]
		dest := filepath.Join(dir, rel)
		b := filepath.Join(backup, rel)
		if _, err := os.Lstat(b); err == nil {
			os.Remove(dest)
			_ = os.Rename(b, dest)
		} else {
			os.Remove(dest) // 新增的文件直接删除
		}
	}
}
