package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// 资产名取自 llama.cpp b12603 发布
var testAssets = func() []ReleaseAsset {
	var out []ReleaseAsset
	for _, n := range []string{
		"cudart-llama-b12603-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b12603-bin-ubuntu-cuda-13.5-x64.tar.gz",
		"cudart-llama-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip",
		"llama-b12603-bin-android-arm64.tar.gz", "llama-b12603-bin-android-arm64-snapdragon.tar.gz",
		"llama-b12603-bin-macos-arm64.tar.gz", "llama-b12603-bin-ubuntu-x64.tar.gz",
		"llama-b12603-bin-ubuntu-cuda-12.8-x64.tar.gz", "llama-b12603-bin-ubuntu-cuda-13.5-x64.tar.gz",
		"llama-b12603-bin-ubuntu-vulkan-x64.tar.gz", "llama-b12603-bin-ubuntu-rocm-10.0-x64.tar.gz",
		"llama-b12603-bin-win-cpu-x64.zip", "llama-b12603-bin-win-cuda-12.4-x64.zip", "llama-b12603-bin-win-cuda-13.5-x64.zip",
		"llama-b12603-bin-win-vulkan-x64.zip", "llama-b12603-bin-win-rocm-10.0-x64.zip", "llama-b12603-bin-win-sycl-x64.zip",
		"llama-b12603-ui.tar.gz",
	} {
		out = append(out, ReleaseAsset{Name: n})
	}
	return out
}()

func TestParseBuild(t *testing.T) {
	for in, want := range map[string]int{"6500 (abc1234)": 6500, "1.0.0-dev (build 12602, commit 418fe97b5)": 12602, "1.0.0-dev": 0, "b12603": 12603, " 42": 42, "unknown": 0, "": 0} {
		if got := parseBuild(in); got != want {
			t.Errorf("parseBuild(%q)=%d want %d", in, got, want)
		}
	}
	if f := assetFamily("llama-b12603-bin-win-cuda-12.4-x64.zip"); f != "llama-bin-win-cuda-12.4-x64.zip" {
		t.Errorf("assetFamily=%s", f)
	}
}

func TestPickUpdateAsset(t *testing.T) {
	cases := []struct {
		name, goos, arch, cuda, installed string
		files                             []string
		want, wantDep                     string
	}{
		{"win cpu", "windows", "amd64", "", "", []string{"llama-server.exe", "ggml-cpu-haswell.dll"}, "llama-b12603-bin-win-cpu-x64.zip", ""},
		{"win vulkan", "windows", "amd64", "", "", []string{"ggml-vulkan.dll"}, "llama-b12603-bin-win-vulkan-x64.zip", ""},
		{"win cuda12 有运行库", "windows", "amd64", "13.0", "", []string{"ggml-cuda.dll", "cudart64_12.dll"}, "llama-b12603-bin-win-cuda-12.4-x64.zip", ""},
		{"win cuda 无运行库", "windows", "amd64", "13.5", "", []string{"ggml-cuda.dll"}, "llama-b12603-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"},
		{"win hip", "windows", "amd64", "", "", []string{"ggml-hip.dll"}, "llama-b12603-bin-win-rocm-10.0-x64.zip", ""},
		{"linux cpu", "linux", "amd64", "", "", []string{"llama-server", "libggml-cpu-haswell.so"}, "llama-b12603-bin-ubuntu-x64.tar.gz", ""},
		{"linux cuda", "linux", "amd64", "12.9", "", []string{"libggml-cuda.so"}, "llama-b12603-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b12603-bin-ubuntu-cuda-12.8-x64.tar.gz"},
		{"linux rocm", "linux", "amd64", "", "", []string{"libggml-hip.so"}, "llama-b12603-bin-ubuntu-rocm-10.0-x64.tar.gz", ""},
		{"安装记录优先", "linux", "amd64", "", "llama-bin-ubuntu-vulkan-x64.tar.gz", []string{"libggml-cuda.so"}, "llama-b12603-bin-ubuntu-vulkan-x64.tar.gz", ""},
		{"android", "android", "arm64", "", "", []string{"llama-server"}, "llama-b12603-bin-android-arm64.tar.gz", ""},
		{"macos", "darwin", "arm64", "", "", []string{"llama-server"}, "llama-b12603-bin-macos-arm64.tar.gz", ""},
		{"linux sycl 无包", "linux", "amd64", "", "", []string{"libggml-sycl.so"}, "", ""},
	}
	for _, c := range cases {
		a, _ := pickUpdateAsset(testAssets, c.files, c.goos, c.arch, c.cuda, c.installed)
		got := ""
		if a != nil {
			got = a.Name
		}
		if got != c.want {
			t.Errorf("%s: got %q want %q", c.name, got, c.want)
			continue
		}
		if a == nil {
			continue
		}
		dep := ""
		if ds := cudaDeps(testAssets, a, c.files, c.goos); len(ds) > 0 {
			dep = ds[0].Name
		}
		if dep != c.wantDep {
			t.Errorf("%s: dep got %q want %q", c.name, dep, c.wantDep)
		}
	}
	if cs := candidateAssets(testAssets, "windows", "amd64"); len(cs) != 6 {
		t.Errorf("candidates windows: %v", cs)
	}
}

func TestRelUnder(t *testing.T) {
	cases := []struct{ name, root, want string }{
		{"llama-b1/llama-server", "llama-b1", "llama-server"},
		{"llama-b1/sub/a.so", "llama-b1", "sub/a.so"},
		{"other/x", "llama-b1", ""},
		{"llama-server.exe", ".", "llama-server.exe"},
		{"../../etc/passwd", ".", "etc/passwd"}, // Clean 后不会越出目标目录
		{`dir\a.dll`, "dir", "a.dll"},
	}
	for _, c := range cases {
		if got := relUnder(c.name, c.root); got != c.want {
			t.Errorf("relUnder(%q,%q)=%q want %q", c.name, c.root, got, c.want)
		}
	}
}

func writeTarGz(t *testing.T, p string, files map[string]string, links map[string]string) {
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	for n, c := range files {
		_ = tw.WriteHeader(&tar.Header{Name: n, Mode: 0o755, Size: int64(len(c)), Typeflag: tar.TypeReg})
		_, _ = tw.Write([]byte(c))
	}
	for n, l := range links {
		_ = tw.WriteHeader(&tar.Header{Name: n, Linkname: l, Typeflag: tar.TypeSymlink})
	}
	tw.Close()
	gz.Close()
	f.Close()
}

func TestExtractReplaceRollback(t *testing.T) {
	tmp := t.TempDir()
	// tar.gz：顶层目录 llama-b2/
	tgz := filepath.Join(tmp, "main.tar.gz")
	links := map[string]string{}
	if runtime.GOOS != "windows" {
		links["llama-b2/libllama.so"] = "libllama.so.1"
	}
	writeTarGz(t, tgz, map[string]string{"llama-b2/llama-server": "new", "llama-b2/libllama.so.1": "lib2", "llama-b2/LICENSE": "L"}, links)
	// zip：扁平
	zp := filepath.Join(tmp, "dep.zip")
	zf, _ := os.Create(zp)
	zw := zip.NewWriter(zf)
	w, _ := zw.Create("cudart64_13.dll")
	w.Write([]byte("rt"))
	zw.Close()
	zf.Close()

	dir := filepath.Join(tmp, "bin")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "llama-server"), []byte("old"), 0o755)
	os.WriteFile(filepath.Join(dir, "keep.txt"), []byte("k"), 0o644)
	stage := filepath.Join(dir, ".new")
	if err := extractArchive(tgz, stage, true); err != nil {
		t.Fatal(err)
	}
	if err := extractArchive(zp, stage, false); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(stage, "llama-server")); string(b) != "new" {
		t.Fatalf("staged server = %q", b)
	}
	if st, err := os.Stat(filepath.Join(stage, "llama-server")); err != nil || st.Mode().Perm()&0o100 == 0 && runtime.GOOS != "windows" {
		t.Errorf("执行权限丢失：%v", st.Mode())
	}
	backup := filepath.Join(dir, ".old")
	moved, err := replaceFiles(stage, dir, backup)
	if err != nil {
		t.Fatal(err)
	}
	read := func(n string) string { b, _ := os.ReadFile(filepath.Join(dir, n)); return string(b) }
	if read("llama-server") != "new" || read("cudart64_13.dll") != "rt" || read("keep.txt") != "k" {
		t.Fatalf("替换结果不对：%q %q %q", read("llama-server"), read("cudart64_13.dll"), read("keep.txt"))
	}
	if runtime.GOOS != "windows" && read("libllama.so") != "lib2" {
		t.Errorf("符号链接未生效")
	}
	rollback(moved, dir, backup)
	if read("llama-server") != "old" || read("keep.txt") != "k" {
		t.Errorf("回滚后 llama-server=%q", read("llama-server"))
	}
	if _, err := os.Stat(filepath.Join(dir, "cudart64_13.dll")); err == nil {
		t.Errorf("回滚后新增文件应删除")
	}
	// 主程序包中没有 llama-server 时报错
	bad := filepath.Join(tmp, "bad.tar.gz")
	writeTarGz(t, bad, map[string]string{"x/readme": "r"}, nil)
	if err := extractArchive(bad, filepath.Join(tmp, "s2"), true); err == nil {
		t.Errorf("缺少 llama-server 应报错")
	}
}

func TestProxyFor(t *testing.T) {
	c := Config{Proxy: "http://p:1"}
	if proxyFor(c, netGitHub) != "http://p:1" || proxyFor(c, netHF) != "http://p:1" || proxyFor(c, netMS) != "" {
		t.Error("通用代理")
	}
	c.GitHubProxy = "socks5://g:2"
	c.ProxyAll = true
	if proxyFor(c, netGitHub) != "socks5://g:2" || proxyFor(c, netMS) != "http://p:1" {
		t.Error("GitHub 代理")
	}
}
