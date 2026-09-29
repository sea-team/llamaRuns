//go:build !windows

package main

import (
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

func shQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'" }

// OpenTerminal 在新终端窗口中运行命令，命令结束后等待回车再关闭。
func OpenTerminal(title, dir, bin string, args []string) (string, error) {
	parts := []string{shQuote(bin)}
	for _, a := range args {
		parts = append(parts, shQuote(a))
	}
	script := "cd " + shQuote(dir) + " && " + strings.Join(parts, " ") + `; echo; printf '按回车键关闭…'; read _`

	start := func(name string, c *exec.Cmd) (string, error) {
		if err := c.Start(); err != nil {
			return "", err
		}
		go c.Wait()
		return name, nil
	}
	switch runtime.GOOS {
	case "darwin":
		esc := strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(script)
		return start("Terminal", exec.Command("osascript",
			"-e", `tell application "Terminal" to do script "`+esc+`"`,
			"-e", `tell application "Terminal" to activate`))
	case "android":
		return "", errors.New("Android 暂不支持自动打开终端，请复制命令在 Termux 中运行")
	}
	// WSL 中可借助 Windows Terminal 打开
	if distro := os.Getenv("WSL_DISTRO_NAME"); distro != "" {
		if wt, err := exec.LookPath("wt.exe"); err == nil {
			return start("Windows Terminal", exec.Command(wt, "new-tab", "--title", title,
				"wsl.exe", "-d", distro, "--", "bash", "-c", strings.ReplaceAll(script, ";", `\;`)))
		}
	}
	if os.Getenv("DISPLAY") == "" && os.Getenv("WAYLAND_DISPLAY") == "" {
		return "", errors.New("未检测到图形桌面，无法打开终端，请复制命令手动运行")
	}
	terms := []struct {
		bin  string
		args []string
	}{
		{"x-terminal-emulator", []string{"-e", "sh", "-c", script}},
		{"gnome-terminal", []string{"--title", title, "--", "sh", "-c", script}},
		{"konsole", []string{"-p", "tabtitle=" + title, "-e", "sh", "-c", script}},
		{"xfce4-terminal", []string{"--title", title, "-x", "sh", "-c", script}},
		{"kitty", []string{"--title", title, "sh", "-c", script}},
		{"alacritty", []string{"--title", title, "-e", "sh", "-c", script}},
		{"xterm", []string{"-T", title, "-e", "sh", "-c", script}},
	}
	for _, t := range terms {
		if p, err := exec.LookPath(t.bin); err == nil {
			return start(t.bin, exec.Command(p, t.args...))
		}
	}
	return "", errors.New("未找到可用的终端程序，请复制命令手动运行")
}
