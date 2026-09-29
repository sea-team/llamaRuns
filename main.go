package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8686", "面板监听地址，局域网访问可设为 0.0.0.0:8686")
	cfgPath := flag.String("config", "", "配置文件路径（默认为程序所在目录下的 config.json）")
	flag.Parse()

	if *cfgPath == "" {
		dir := "."
		if exe, err := os.Executable(); err == nil {
			dir = filepath.Dir(exe)
		}
		*cfgPath = filepath.Join(dir, "config.json")
	}
	cfg, err := LoadConfig(*cfgPath)
	if err != nil {
		log.Fatalf("加载配置失败：%v", err)
	}
	mgr := NewManager(cfg)
	app := &App{
		cfg:  cfg,
		mgr:  mgr,
		mon:  NewSysMonitor(mgr),
		info: &InfoFetcher{cacheDir: filepath.Join(cfg.Dir(), "cache", "info"), cfg: cfg},
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sig
		log.Println("正在停止所有模型……")
		mgr.StopAll("")
		os.Exit(0)
	}()

	log.Printf("配置文件：%s", *cfgPath)
	fmt.Printf("llamaRunModel 面板已启动：http://%s\n", *addr)
	log.Fatal(http.ListenAndServe(*addr, app.Routes()))
}
