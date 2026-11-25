package main

import (
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
)

// 系统代理：代理地址留空时，先用环境变量（HTTPS_PROXY 等），再用系统设置
// （Windows Internet 选项 / macOS 网络设置）。Go 默认只认环境变量，
// 而 Clash、v2rayN 等的“系统代理”模式只写系统设置，不跟随就会直连失败。

var sysProxy struct {
	mu  sync.Mutex
	at  time.Time
	val string
}

// SystemProxy 返回系统代理地址（缓存 30 秒），没有时返回空。
func SystemProxy() string {
	sysProxy.mu.Lock()
	defer sysProxy.mu.Unlock()
	if time.Since(sysProxy.at) > 30*time.Second {
		sysProxy.val, sysProxy.at = readSystemProxy(), time.Now()
	}
	return sysProxy.val
}

// parseWinProxy 解析 Windows ProxyServer：“主机:端口”，或按协议分别设置的
// “http=主机:端口;https=主机:端口;socks=主机:端口”。
func parseWinProxy(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if !strings.Contains(s, "=") {
		if strings.Contains(s, "://") {
			return s
		}
		return "http://" + s
	}
	m := map[string]string{}
	for _, part := range strings.Split(s, ";") {
		if k, v, ok := strings.Cut(strings.TrimSpace(part), "="); ok && v != "" {
			m[strings.ToLower(k)] = v
		}
	}
	for _, k := range []string{"https", "http"} {
		if v := m[k]; v != "" {
			if strings.Contains(v, "://") {
				return v
			}
			return "http://" + v
		}
	}
	if v := m["socks"]; v != "" {
		return "socks5://" + v
	}
	return ""
}

// autoProxy 用于未配置代理的请求：环境变量优先，其次系统代理。
func autoProxy(req *http.Request) (*url.URL, error) {
	if u, err := http.ProxyFromEnvironment(req); u != nil || err != nil {
		return u, err
	}
	if p := SystemProxy(); p != "" && !isLocalHost(req.URL.Hostname()) {
		return url.Parse(p)
	}
	return nil, nil
}

func isLocalHost(h string) bool {
	return h == "localhost" || h == "127.0.0.1" || h == "::1"
}

// EffectiveProxy 说明实际使用的代理（用于界面显示）。
func EffectiveProxy(configured string) (proxy, source string) {
	if p := strings.TrimSpace(configured); p != "" {
		return p, "设置"
	}
	req, _ := http.NewRequest("GET", "https://api.github.com/", nil)
	if u, _ := http.ProxyFromEnvironment(req); u != nil {
		return u.String(), "环境变量"
	}
	if p := SystemProxy(); p != "" {
		return p, "系统代理"
	}
	return "", "直连"
}

var scutilRe = regexp.MustCompile(`(?m)^\s*(HTTPSEnable|HTTPSProxy|HTTPSPort|HTTPEnable|HTTPProxy|HTTPPort|SOCKSEnable|SOCKSProxy|SOCKSPort)\s*:\s*(\S+)`)

// parseScutilProxy 解析 macOS scutil --proxy 输出，依次取 HTTPS、HTTP、SOCKS 代理。
func parseScutilProxy(o string) string {
	v := map[string]string{}
	for _, m := range scutilRe.FindAllStringSubmatch(o, -1) {
		v[m[1]] = m[2]
	}
	for _, p := range []struct{ pre, scheme string }{{"HTTPS", "http"}, {"HTTP", "http"}, {"SOCKS", "socks5"}} {
		if v[p.pre+"Enable"] == "1" && v[p.pre+"Proxy"] != "" {
			port := v[p.pre+"Port"]
			if port == "" {
				port = "80"
			}
			return p.scheme + "://" + v[p.pre+"Proxy"] + ":" + port
		}
	}
	return ""
}
