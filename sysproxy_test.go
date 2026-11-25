package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseWinProxy(t *testing.T) {
	cases := map[string]string{
		"127.0.0.1:7890": "http://127.0.0.1:7890",
		"http=127.0.0.1:7890;https=127.0.0.1:7891;socks=127.0.0.1:7892": "http://127.0.0.1:7891",
		"socks=127.0.0.1:1080":  "socks5://127.0.0.1:1080",
		"socks5://1.2.3.4:1080": "socks5://1.2.3.4:1080",
		"":                      "",
	}
	for in, want := range cases {
		if got := parseWinProxy(in); got != want {
			t.Errorf("parseWinProxy(%q) = %q，期望 %q", in, got, want)
		}
	}
}

func TestParseScutilProxy(t *testing.T) {
	o := `<dictionary> {
  HTTPEnable : 1
  HTTPPort : 7890
  HTTPProxy : 127.0.0.1
  HTTPSEnable : 1
  HTTPSPort : 7891
  HTTPSProxy : 127.0.0.1
  SOCKSEnable : 0
}`
	if got := parseScutilProxy(o); got != "http://127.0.0.1:7891" {
		t.Fatalf("应优先取 HTTPS 代理：%q", got)
	}
	if got := parseScutilProxy("<dictionary> {\n  HTTPEnable : 0\n}"); got != "" {
		t.Fatalf("未启用时应为空：%q", got)
	}
}

func TestAutoProxyEnvFirst(t *testing.T) {
	t.Setenv("HTTPS_PROXY", "http://10.0.0.1:3128")
	req, _ := http.NewRequest("GET", "https://api.github.com/", nil)
	u, err := autoProxy(req)
	if err != nil || u == nil || u.Host != "10.0.0.1:3128" {
		t.Fatalf("环境变量代理应优先：%v %v", u, err)
	}
	if p, src := EffectiveProxy(" socks5://h:1 "); p != "socks5://h:1" || src != "设置" {
		t.Fatalf("设置的代理应优先：%s %s", p, src)
	}
}

func TestMirrorURL(t *testing.T) {
	src := "https://github.com/ggml-org/llama.cpp/releases/download/b1/a.zip"
	if got := mirrorURL("https://ghfast.top/", src); got != "https://ghfast.top/"+src {
		t.Fatal(got)
	}
	if got := mirrorURL("https://m.example/get?u={url}", src); got != "https://m.example/get?u="+src {
		t.Fatal(got)
	}
	if mirrorURL(" ", src) != src {
		t.Fatal("空镜像应返回原地址")
	}
}

// 镜像下载失败（或内容被篡改）时应改为直连，并以 sha256 校验结果为准。
func TestDownloadMirrorFallback(t *testing.T) {
	data := []byte("llama.cpp release package")
	sum := sha256.Sum256(data)
	direct := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write(data) }))
	defer direct.Close()
	var mirrorHits int
	mirror := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mirrorHits++
		w.Write([]byte("tampered package!!!!!!!!")) // 长度相同、内容不同
	}))
	defer mirror.Close()

	dir := t.TempDir()
	cfg, err := LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	c := cfg.Get()
	c.GitHubMirror = mirror.URL + "/{url}"
	t.Setenv("HTTPS_PROXY", "")
	t.Setenv("HTTP_PROXY", "")
	if err := cfg.UpdateSettings(c); err != nil {
		t.Fatal(err)
	}
	a := &App{cfg: cfg, info: &InfoFetcher{cfg: cfg}, upd: &Updater{}}
	as := ReleaseAsset{Name: "a.zip", URL: direct.URL + "/a.zip", Size: int64(len(data)), Digest: "sha256:" + hex.EncodeToString(sum[:])}
	dst := filepath.Join(dir, "a.zip")
	if err := a.download(context.Background(), as, dst); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(dst); string(b) != string(data) || mirrorHits != 1 {
		t.Fatalf("应先试镜像、校验失败后直连：hits=%d content=%q", mirrorHits, b)
	}
	log := strings.Join(a.upd.Status().Log, "\n")
	if !strings.Contains(log, "镜像下载失败") {
		t.Fatalf("日志应说明改为直连：%s", log)
	}
	// 没有 sha256 时不走镜像
	os.Remove(dst)
	as.Digest = ""
	if err := a.download(context.Background(), as, dst); err != nil || mirrorHits != 1 {
		t.Fatalf("无校验值时不应使用镜像：hits=%d err=%v", mirrorHits, err)
	}
}
