#!/usr/bin/env bash
# 交叉编译所有平台到 dist/
set -e
cd "$(dirname "$0")"
for t in windows/amd64 windows/arm64 linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 android/arm64; do
  os=${t%/*}; arch=${t#*/}; ext=""; [ "$os" = windows ] && ext=".exe"
  GOOS=$os GOARCH=$arch CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o "dist/llamaRunModel-$os-$arch$ext" .
  echo "built dist/llamaRunModel-$os-$arch$ext"
done
