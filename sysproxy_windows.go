package main

import (
	"syscall"
	"unsafe"
)

// readSystemProxy 读取 Windows “Internet 选项”中的代理（Clash、v2rayN 等开启“系统代理”时写入这里）。
func readSystemProxy() string {
	var k syscall.Handle
	path, _ := syscall.UTF16PtrFromString(`Software\Microsoft\Windows\CurrentVersion\Internet Settings`)
	if syscall.RegOpenKeyEx(syscall.HKEY_CURRENT_USER, path, 0, syscall.KEY_READ, &k) != nil {
		return ""
	}
	defer syscall.RegCloseKey(k)
	query := func(name string) ([]byte, uint32, bool) {
		n, _ := syscall.UTF16PtrFromString(name)
		var typ, size uint32
		if syscall.RegQueryValueEx(k, n, nil, &typ, nil, &size) != nil || size == 0 {
			return nil, 0, false
		}
		buf := make([]byte, size)
		if syscall.RegQueryValueEx(k, n, nil, &typ, &buf[0], &size) != nil {
			return nil, 0, false
		}
		return buf[:size], typ, true
	}
	if b, typ, ok := query("ProxyEnable"); !ok || typ != syscall.REG_DWORD || len(b) < 4 || *(*uint32)(unsafe.Pointer(&b[0])) == 0 {
		return ""
	}
	b, typ, ok := query("ProxyServer")
	if !ok || typ != syscall.REG_SZ || len(b) < 2 {
		return ""
	}
	u16 := unsafe.Slice((*uint16)(unsafe.Pointer(&b[0])), len(b)/2)
	return parseWinProxy(syscall.UTF16ToString(u16))
}
