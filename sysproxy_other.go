//go:build !windows

package main

import "runtime"

// readSystemProxy 读取 macOS 网络设置中的代理（scutil --proxy）；Linux 通过环境变量设置代理，这里不处理。
func readSystemProxy() string {
	if runtime.GOOS != "darwin" {
		return ""
	}
	o, err := runTool("scutil", "--proxy")
	if err != nil {
		return ""
	}
	return parseScutilProxy(o)
}
