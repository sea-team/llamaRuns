package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

// 网络共享目录（SMB / NFS 等）支持。
// Windows 直接使用 UNC 路径（\\主机\共享\目录）；其它系统需先挂载，
// 填写 smb://、nfs://、//主机/共享 或 主机:/导出路径 时自动查找对应的本地挂载点。

// netLoc 网络位置：主机 + 路径（SMB 为 /共享/子目录，NFS 为导出路径）。
type netLoc struct {
	Host string
	Path string
	SMB  bool
}

var nfsSrcRe = regexp.MustCompile(`^([A-Za-z0-9._\-\[\]:]+?):(/.*)$`)

// parseNetLoc 识别网络路径写法，不是网络路径时返回 false。
func parseNetLoc(p string) (netLoc, bool) {
	p = strings.TrimSpace(p)
	lower := strings.ToLower(p)
	var rest string
	smb := true
	switch {
	case strings.HasPrefix(lower, "smb://"), strings.HasPrefix(lower, "cifs://"):
		rest = p[strings.Index(p, "://")+3:]
	case strings.HasPrefix(lower, "nfs://"):
		rest, smb = p[6:], false
	case strings.HasPrefix(p, `\\`), strings.HasPrefix(p, "//"):
		rest = p[2:]
	default:
		// 主机:/导出路径（NFS 写法）；排除 Windows 盘符 C:/
		if m := nfsSrcRe.FindStringSubmatch(p); m != nil && len(m[1]) > 1 {
			return netLoc{Host: strings.ToLower(m[1]), Path: cleanNetPath(m[2])}, true
		}
		return netLoc{}, false
	}
	rest = strings.ReplaceAll(rest, `\`, "/")
	host, path, _ := strings.Cut(rest, "/")
	if i := strings.LastIndex(host, "@"); i >= 0 { // 去掉 user@
		host = host[i+1:]
	}
	if host == "" {
		return netLoc{}, false
	}
	return netLoc{Host: strings.ToLower(host), Path: cleanNetPath("/" + path), SMB: smb}, true
}

func cleanNetPath(p string) string {
	p = strings.TrimRight(filepath.ToSlash(p), "/")
	if p == "" {
		return "/"
	}
	return p
}

// IsNetPath 判断填写的目录是否为网络路径（含 Windows 映射的网络驱动器）。
func IsNetPath(p string) bool {
	if _, ok := parseNetLoc(p); ok {
		return true
	}
	if runtime.GOOS == "windows" {
		return isRemoteDrive(filepath.VolumeName(p))
	}
	for _, m := range netMounts() {
		if m.Target != "/" && (p == m.Target || strings.HasPrefix(p, m.Target+"/")) {
			return true
		}
	}
	return false
}

// ResolveDir 把用户填写的目录转换为本机可直接访问的绝对路径。
func ResolveDir(p string) (string, error) {
	p = strings.TrimSpace(p)
	if strings.HasPrefix(strings.ToLower(p), "file://") {
		p = p[7:]
	}
	loc, ok := parseNetLoc(p)
	if !ok {
		return filepath.Abs(p)
	}
	if runtime.GOOS == "windows" {
		// UNC：\\主机\共享\目录（NFS 需安装 Windows 的 NFS 客户端）
		return `\\` + loc.Host + strings.ReplaceAll(loc.Path, "/", `\`), nil
	}
	if t := findMount(loc, netMounts()); t != "" {
		return t, nil
	}
	return "", fmt.Errorf("未找到 %s 的本地挂载点，请先挂载（如 mount -t cifs / mount -t nfs）或直接填写挂载后的本地路径", displayNetLoc(loc))
}

func displayNetLoc(l netLoc) string {
	if l.SMB {
		return "//" + l.Host + l.Path
	}
	return l.Host + ":" + l.Path
}

// mountInfo 一条网络文件系统挂载记录。
type mountInfo struct {
	Loc    netLoc
	Target string
}

// findMount 在挂载表中找与网络位置匹配（最长前缀）的挂载点，返回对应的本地路径。
func findMount(loc netLoc, mounts []mountInfo) string {
	best, bestLen := "", -1
	for _, m := range mounts {
		if m.Loc.Host != loc.Host {
			continue
		}
		mp, lp := m.Loc.Path, loc.Path
		if m.Loc.SMB { // SMB 共享名不区分大小写
			mp, lp = strings.ToLower(mp), strings.ToLower(lp)
		}
		if lp != mp && !strings.HasPrefix(lp, strings.TrimSuffix(mp, "/")+"/") {
			continue
		}
		if len(mp) > bestLen {
			bestLen = len(mp)
			best = filepath.Join(m.Target, filepath.FromSlash(loc.Path[len(m.Loc.Path):]))
		}
	}
	return best
}

var netFSTypes = map[string]bool{
	"cifs": true, "smb3": true, "smbfs": true, "nfs": true, "nfs4": true,
	"fuse.sshfs": true, "sshfs": true, "afpfs": true, "webdav": true, "davfs": true,
}

var (
	mountsMu   sync.Mutex
	mountsAt   time.Time
	mountsLast []mountInfo
)

// netMounts 返回网络文件系统挂载表，缓存 5 秒（扫描时每个目录都要用到）。
func netMounts() []mountInfo {
	mountsMu.Lock()
	defer mountsMu.Unlock()
	if time.Since(mountsAt) > 5*time.Second {
		mountsLast, mountsAt = readNetMounts(), time.Now()
	}
	return mountsLast
}

// readNetMounts 读取当前系统的网络文件系统挂载（Linux：/proc/mounts 与 gvfs；macOS：mount 命令）。
func readNetMounts() []mountInfo {
	var out []mountInfo
	switch runtime.GOOS {
	case "linux", "android":
		f, err := os.Open("/proc/mounts")
		if err == nil {
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				fs := strings.Fields(sc.Text())
				if len(fs) < 3 || !netFSTypes[fs[2]] {
					continue
				}
				if m, ok := mountEntry(unescapeMount(fs[0]), unescapeMount(fs[1])); ok {
					out = append(out, m)
				}
			}
			f.Close()
		}
		// GNOME 文件管理器挂载的共享：/run/user/<uid>/gvfs/smb-share:server=主机,share=共享
		dir := fmt.Sprintf("/run/user/%d/gvfs", os.Getuid())
		if es, err := os.ReadDir(dir); err == nil {
			for _, e := range es {
				if m, ok := gvfsEntry(dir, e.Name()); ok {
					out = append(out, m)
				}
			}
		}
	case "darwin":
		b, err := exec.Command("mount").Output()
		if err != nil {
			return nil
		}
		// //user@host/share on /Volumes/share (smbfs, nodev, ...)
		re := regexp.MustCompile(`^(\S+) on (.+) \((\w+)`)
		for _, line := range strings.Split(string(b), "\n") {
			if m := re.FindStringSubmatch(line); m != nil && netFSTypes[m[3]] {
				if e, ok := mountEntry(m[1], m[2]); ok {
					out = append(out, e)
				}
			}
		}
	}
	return out
}

func mountEntry(src, target string) (mountInfo, bool) {
	loc, ok := parseNetLoc(src)
	if !ok {
		// sshfs：user@host:/path
		if i := strings.Index(src, "@"); i >= 0 {
			loc, ok = parseNetLoc(src[i+1:])
		}
		if !ok {
			return mountInfo{}, false
		}
	}
	return mountInfo{Loc: loc, Target: target}, true
}

func gvfsEntry(dir, name string) (mountInfo, bool) {
	kind, args, ok := strings.Cut(name, ":")
	if !ok || kind != "smb-share" {
		return mountInfo{}, false
	}
	kv := map[string]string{}
	for _, p := range strings.Split(args, ",") {
		k, v, _ := strings.Cut(p, "=")
		kv[k] = v
	}
	if kv["server"] == "" || kv["share"] == "" {
		return mountInfo{}, false
	}
	return mountInfo{Loc: netLoc{Host: strings.ToLower(kv["server"]), Path: "/" + kv["share"], SMB: true}, Target: filepath.Join(dir, name)}, true
}

// unescapeMount 还原 /proc/mounts 中的八进制转义（如空格 \040）。
func unescapeMount(s string) string {
	if !strings.Contains(s, `\`) {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+4 <= len(s) {
			var v int
			if _, err := fmt.Sscanf(s[i+1:i+4], "%03o", &v); err == nil {
				b.WriteByte(byte(v))
				i += 3
				continue
			}
		}
		b.WriteByte(s[i])
	}
	return b.String()
}

// netRoots 目录选择器中显示的网络位置。
func netRoots() []string {
	var out []string
	switch runtime.GOOS {
	case "windows":
		for c := 'A'; c <= 'Z'; c++ {
			if d := string(c) + `:\`; isRemoteDrive(d[:2]) {
				out = append(out, d)
			}
		}
	default:
		for _, m := range netMounts() {
			out = append(out, m.Target)
		}
	}
	return out
}

// errTimeout 网络目录无响应时返回的错误。
var errTimeout = fmt.Errorf("读取超时（网络目录无响应，请检查共享是否在线）")

// withTimeout 在限定时间内执行 f；超时后立即返回 errTimeout，f 在后台继续直到结束。
func withTimeout[T any](d time.Duration, f func() (T, error)) (T, error) {
	type res struct {
		v   T
		err error
	}
	ch := make(chan res, 1)
	go func() {
		v, err := f()
		ch <- res{v, err}
	}()
	select {
	case r := <-ch:
		return r.v, r.err
	case <-time.After(d):
		var zero T
		return zero, errTimeout
	}
}
