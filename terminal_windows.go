package main

import (
	"os/exec"
	"strings"
	"syscall"
)

const createNewConsole = 0x00000010

func psQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", "''") + "'" }

// OpenTerminal 在新终端中运行命令：优先 Windows Terminal，否则打开新的 PowerShell 控制台窗口。
func OpenTerminal(title, dir, bin string, args []string) (string, error) {
	parts := []string{"&", psQuote(bin)}
	for _, a := range args {
		parts = append(parts, psQuote(a))
	}
	ps := strings.Join(parts, " ")
	if wt, err := exec.LookPath("wt.exe"); err == nil {
		// Windows Terminal 用 ; 分隔子命令，需转义
		c := exec.Command(wt, "new-tab", "--title", title, "-d", dir,
			"powershell.exe", "-NoLogo", "-NoExit", "-Command", strings.ReplaceAll(ps, ";", `\;`))
		if err := c.Start(); err == nil {
			go c.Wait()
			return "Windows Terminal", nil
		}
	}
	c := exec.Command("powershell.exe", "-NoLogo", "-NoExit", "-Command", ps)
	c.Dir = dir
	c.SysProcAttr = &syscall.SysProcAttr{CreationFlags: createNewConsole}
	if err := c.Start(); err != nil {
		return "", err
	}
	go c.Wait()
	return "PowerShell", nil
}
