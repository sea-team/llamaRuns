package main

import (
	"os/exec"
	"syscall"
)

// prepareCmd 在 Windows 下隐藏子进程控制台窗口。
func prepareCmd(c *exec.Cmd) {
	c.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
}
