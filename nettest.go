package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// 测试 GitHub 连接：用界面上填写（未保存）的代理与镜像访问 GitHub API 和发布包下载，
// 便于在更新前确认网络设置是否可用。

type netProbe struct {
	OK     bool   `json:"ok"`
	MS     int64  `json:"ms"`
	Detail string `json:"detail"`
}

type netTestRes struct {
	Proxy       string    `json:"proxy"`       // 实际使用的代理
	ProxySource string    `json:"proxySource"` // 设置 / 环境变量 / 系统代理 / 直连
	API         netProbe  `json:"api"`
	Download    netProbe  `json:"download"`
	Mirror      *netProbe `json:"mirror,omitempty"`
}

func (a *App) handleNetTest(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Proxy  string `json:"proxy"`
		Mirror string `json:"mirror"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	cfg := a.cfg.Get()
	proxy := strings.TrimSpace(req.Proxy)
	if proxy == "" {
		proxy = strings.TrimSpace(cfg.Proxy) // GitHub 代理留空时使用通用代理
	}
	res := netTestRes{}
	res.Proxy, res.ProxySource = EffectiveProxy(proxy)
	c, err := a.info.clientWith(proxy, 20*time.Second)
	if err != nil {
		writeErr(w, 400, err)
		return
	}
	// 1. GitHub API：检测更新依赖它，同时读取剩余请求次数
	start := time.Now()
	resp, err := c.Get("https://api.github.com/rate_limit")
	res.API.MS = time.Since(start).Milliseconds()
	if err != nil {
		res.API.Detail = netErrText(err)
	} else {
		var rl struct {
			Rate struct {
				Remaining int   `json:"remaining"`
				Limit     int   `json:"limit"`
				Reset     int64 `json:"reset"`
			} `json:"rate"`
		}
		_ = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&rl)
		resp.Body.Close()
		res.API.OK = resp.StatusCode == 200
		res.API.Detail = fmt.Sprintf("HTTP %d", resp.StatusCode)
		if rl.Rate.Limit > 0 {
			res.API.Detail += fmt.Sprintf("，本小时剩余 %d/%d 次请求", rl.Rate.Remaining, rl.Rate.Limit)
			if rl.Rate.Remaining == 0 {
				res.API.OK = false
				res.API.Detail += fmt.Sprintf("，%s 后恢复", time.Until(time.Unix(rl.Rate.Reset, 0)).Round(time.Minute))
			}
		}
	}
	// 2. 发布包下载：取最新发布中最小的文件，读取前 256KB 测速
	asset := ReleaseAsset{}
	if rel, err := a.latestRelease(false); err == nil && len(rel.Assets) > 0 {
		as := append([]ReleaseAsset(nil), rel.Assets...)
		sort.Slice(as, func(i, j int) bool { return as[i].Size < as[j].Size })
		asset = as[0]
	}
	if asset.URL == "" {
		res.Download.Detail = "未取得发布信息，无法测试下载"
	} else {
		res.Download = probeDownload(r.Context(), c, asset.URL)
		if m := strings.TrimSpace(req.Mirror); m != "" {
			mp := probeDownload(r.Context(), c, mirrorURL(m, asset.URL))
			res.Mirror = &mp
		}
	}
	writeJSON(w, res)
}

// probeDownload 读取最多 256KB，返回耗时与速度。
func probeDownload(ctx context.Context, c *http.Client, u string) netProbe {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
	req.Header.Set("Range", "bytes=0-262143")
	start := time.Now()
	resp, err := c.Do(req)
	if err != nil {
		return netProbe{MS: time.Since(start).Milliseconds(), Detail: netErrText(err)}
	}
	defer resp.Body.Close()
	n, err := io.Copy(io.Discard, io.LimitReader(resp.Body, 256<<10))
	d := time.Since(start)
	p := netProbe{MS: d.Milliseconds()}
	if resp.StatusCode != 200 && resp.StatusCode != 206 {
		p.Detail = fmt.Sprintf("HTTP %d", resp.StatusCode)
		return p
	}
	if ct := resp.Header.Get("Content-Type"); strings.HasPrefix(ct, "text/html") {
		p.Detail = "返回的是网页而不是文件，地址格式可能不对"
		return p
	}
	if err != nil && n == 0 {
		p.Detail = netErrText(err)
		return p
	}
	p.OK = true
	p.Detail = fmt.Sprintf("%.0f KB/s", float64(n)/1024/max(d.Seconds(), 0.001))
	return p
}

// netErrText 把常见网络错误转成易懂的说明，并附上原始信息。
func netErrText(err error) string {
	e := err.Error()
	var hint string
	switch {
	case strings.Contains(e, "proxyconnect") && strings.Contains(e, "refused"):
		hint = "连不上代理（连接被拒绝）：代理软件没运行或端口不对"
	case strings.Contains(e, "refused"):
		hint = "连接被拒绝"
	case strings.Contains(e, "no such host"), strings.Contains(e, "server misbehaving"):
		hint = "域名解析失败"
	case strings.Contains(e, "Timeout"), strings.Contains(e, "timeout"), strings.Contains(e, "deadline exceeded"):
		hint = "连接超时"
	case strings.Contains(e, "certificate"):
		hint = "证书校验失败（可能被网络劫持，或代理软件需要信任证书）"
	case strings.Contains(e, "reset by peer"), strings.Contains(e, "EOF"):
		hint = "连接被中断"
	default:
		return e
	}
	return hint + "（" + e + "）"
}
