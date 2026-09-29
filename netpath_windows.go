package main

import (
	"syscall"
	"unsafe"
)

var (
	modKernel32       = syscall.NewLazyDLL("kernel32.dll")
	procGetDriveType  = modKernel32.NewProc("GetDriveTypeW")
	modNetapi32       = syscall.NewLazyDLL("netapi32.dll")
	procNetShareEnum  = modNetapi32.NewProc("NetShareEnum")
	procNetApiBufFree = modNetapi32.NewProc("NetApiBufferFree")
)

// isRemoteDrive 判断盘符（如 "Z:"）是否为映射的网络驱动器。
func isRemoteDrive(vol string) bool {
	if len(vol) != 2 || vol[1] != ':' {
		return false
	}
	p, err := syscall.UTF16PtrFromString(vol + `\`)
	if err != nil {
		return false
	}
	r, _, _ := procGetDriveType.Call(uintptr(unsafe.Pointer(p)))
	return r == 4 // DRIVE_REMOTE
}

// listShares 列出服务器上的磁盘共享（隐藏共享 xxx$ 除外）。
func listShares(host string) ([]string, error) {
	type shareInfo1 struct {
		Name   *uint16
		Type   uint32
		Remark *uint16
	}
	server, err := syscall.UTF16PtrFromString(`\\` + host)
	if err != nil {
		return nil, err
	}
	var buf *shareInfo1
	var read, total, resume uint32
	r, _, _ := procNetShareEnum.Call(uintptr(unsafe.Pointer(server)), 1, uintptr(unsafe.Pointer(&buf)),
		0xFFFFFFFF, uintptr(unsafe.Pointer(&read)), uintptr(unsafe.Pointer(&total)), uintptr(unsafe.Pointer(&resume)))
	if r != 0 && r != 234 { // 234 = ERROR_MORE_DATA
		return nil, syscall.Errno(r)
	}
	if buf == nil {
		return nil, nil
	}
	defer procNetApiBufFree.Call(uintptr(unsafe.Pointer(buf)))
	var out []string
	for _, s := range unsafe.Slice(buf, read) {
		if s.Type == 0 { // STYPE_DISKTREE，且不含 STYPE_SPECIAL
			out = append(out, utf16PtrString(s.Name))
		}
	}
	return out, nil
}

func utf16PtrString(p *uint16) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*uint16)(unsafe.Add(unsafe.Pointer(p), n*2)) != 0 {
		n++
	}
	return syscall.UTF16ToString(unsafe.Slice(p, n))
}
