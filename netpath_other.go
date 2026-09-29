//go:build !windows

package main

import "errors"

func isRemoteDrive(string) bool { return false }

func listShares(string) ([]string, error) {
	return nil, errors.New("仅 Windows 支持列出共享，请先挂载后选择本地挂载路径")
}
