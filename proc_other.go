//go:build !windows

package main

import "os/exec"

func prepareCmd(c *exec.Cmd) {}
