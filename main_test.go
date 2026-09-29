package main

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestBaseNameQuant(t *testing.T) {
	cases := []struct{ in, base, quant string }{
		{"Qwen3-8B-Q4_K_M", "Qwen3-8B", "Q4_K_M"},
		{"qwen2.5-7b-instruct-q8_0", "qwen2.5-7b-instruct", "q8_0"},
		{"Llama-3.3-70B-Instruct-Q4_K_M-00001-of-00002", "Llama-3.3-70B-Instruct", "Q4_K_M"},
		{"gpt-oss-20b-MXFP4", "gpt-oss-20b", "MXFP4"},
		{"Qwen3-30B-A3B-UD-Q4_K_XL", "Qwen3-30B-A3B", "UD-Q4_K_XL"},
		{"model-BF16", "model", "BF16"},
		{"plainmodel", "plainmodel", ""},
	}
	for _, c := range cases {
		if b, q := BaseName(c.in), Quant(c.in); b != c.base || q != c.quant {
			t.Errorf("%s: got (%s,%s) want (%s,%s)", c.in, b, q, c.base, c.quant)
		}
	}
}

func TestGroupModels(t *testing.T) {
	ms := []Model{
		{Name: "gemma-3-4b-it-Q8_0", Path: "/m/gemma-3-4b-it-Q8_0.gguf", Root: "/m", Size: 8},
		{Name: "gemma-3-4b-it-Q4_K_M", Path: "/m/gemma-3-4b-it-Q4_K_M.gguf", Root: "/m", Size: 4},
		{Name: "a-Q4_K_M", Path: "/m/Qwen/a-Q4_K_M.gguf", Root: "/m", Size: 1},
		{Name: "b-Q8_0", Path: "/m/Qwen/b-Q8_0.gguf", Root: "/m", Size: 2},
		{Name: "solo", Path: "/m/solo.gguf", Root: "/m"},
	}
	gs := GroupModels(ms)
	if len(gs) != 3 {
		t.Fatalf("want 3 groups, got %d", len(gs))
	}
	byName := map[string]Group{}
	for _, g := range gs {
		byName[g.Name] = g
	}
	if g := byName["gemma-3-4b-it"]; len(g.Variants) != 2 || g.Variants[0].Size != 4 {
		t.Errorf("gemma group wrong: %+v", g)
	}
	if g := byName["Qwen"]; len(g.Variants) != 2 || g.Dir != "Qwen" {
		t.Errorf("dir group wrong: %+v", g)
	}
}

func TestExtractRecs(t *testing.T) {
	text := `For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0.
presence_penalty between 0 and 2. Example:
./llama-cli -hf Qwen/Qwen3-8B-GGUF:Q8_0 --jinja --color -ngl 99 -fa -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 -c 40960 -n 32768`
	got := map[string]float64{}
	for _, r := range extractRecs(text, "x", map[string]bool{}) {
		got[r.Key] = r.Value
	}
	want := map[string]float64{"temp": 0.6, "topP": 0.95, "topK": 20, "minP": 0, "presencePenalty": 1.5, "ctxSize": 40960}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestSplitArgs(t *testing.T) {
	got, err := SplitArgs(`--a 1 --b "x y" --c 'z'`)
	if err != nil || !reflect.DeepEqual(got, []string{"--a", "1", "--b", "x y", "--c", "z"}) {
		t.Errorf("got %v %v", got, err)
	}
	if _, err := SplitArgs(`--a "x`); err == nil {
		t.Error("want error for unclosed quote")
	}
}

func TestMergePriority(t *testing.T) {
	p := merge(merge(defaultParams(), Params{CtxSize: ptr(8192), Threads: ptr(4)}), Params{CtxSize: ptr(16384)})
	if *p.CtxSize != 16384 || *p.Threads != 4 || string(*p.NGpuLayers) != "auto" {
		t.Errorf("merge wrong: ctx=%d t=%d ngl=%s", *p.CtxSize, *p.Threads, *p.NGpuLayers)
	}
}

func TestBuildArgs(t *testing.T) {
	m := Model{Name: "x", Path: "/m/x.gguf"}
	p := merge(defaultParams(), Params{CtxSize: ptr(1048576), Temp: ptr(0.6), KVOffload: ptr(false), WebUI: ptr(false), Parallel: ptr(2)})
	join := func(a []string) string { return fmt.Sprint(a) }
	gpu, _ := BuildArgs(m, p, RunOpt{Port: 8080, GPUOK: true})
	if s := join(gpu); !strings.Contains(s, "-ngl auto") || !strings.Contains(s, "-c 1048576") || !strings.Contains(s, "-nkvo") ||
		!strings.Contains(s, "--temp 0.6") || !strings.Contains(s, "--no-webui") || !strings.Contains(s, "-np 2") {
		t.Errorf("gpu args: %s", s)
	}
	cpu, _ := BuildArgs(m, p, RunOpt{Port: 8080, GPUOK: true, Mode: "cpu", DeviceFlag: true})
	if s := join(cpu); !strings.Contains(s, "-ngl 0 --device none") || strings.Contains(s, "-nkvo") || strings.Contains(s, "-ngl auto") {
		t.Errorf("cpu args: %s", s)
	}
	cli, _ := BuildArgs(m, p, RunOpt{GPUOK: true, CLI: true})
	if s := join(cli); strings.Contains(s, "--port") || strings.Contains(s, "-np") || strings.Contains(s, "--no-webui") {
		t.Errorf("cli args: %s", s)
	}
}

func TestRecommendAssets(t *testing.T) {
	names := []string{
		"cudart-llama-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip",
		"cudart-llama-b11243-bin-ubuntu-cuda-12.8-x64.tar.gz",
		"llama-b11243-bin-android-arm64.tar.gz", "llama-b11243-bin-android-arm64-snapdragon.tar.gz",
		"llama-b11243-bin-macos-arm64.tar.gz", "llama-b11243-bin-ubuntu-x64.tar.gz",
		"llama-b11243-bin-ubuntu-cuda-12.8-x64.tar.gz", "llama-b11243-bin-ubuntu-cuda-13.4-x64.tar.gz",
		"llama-b11243-bin-ubuntu-vulkan-x64.tar.gz", "llama-b11243-bin-ubuntu-rocm-10.0-x64.tar.gz",
		"llama-b11243-bin-win-cpu-x64.zip", "llama-b11243-bin-win-cuda-12.4-x64.zip", "llama-b11243-bin-win-cuda-13.4-x64.zip",
		"llama-b11243-bin-win-vulkan-x64.zip", "llama-b11243-bin-win-rocm-10.0-x64.zip",
	}
	var as []ReleaseAsset
	for _, n := range names {
		as = append(as, ReleaseAsset{Name: n})
	}
	first := func(recs []Recommend2, kind string) []string {
		for _, r := range recs {
			if r.Kind == kind && !r.Alt {
				var out []string
				for _, a := range r.Assets {
					out = append(out, a.Name)
				}
				return out
			}
		}
		return nil
	}
	nv := []DetectedGPU{{"NVIDIA", "GTX 960"}}
	// 驱动支持 CUDA 12.6：应选 12.4 + 对应 cudart
	recs := recommendAssets(as, "windows", "amd64", nv, "12.6")
	if g := first(recs, "gpu"); !reflect.DeepEqual(g, []string{"llama-b11243-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"}) {
		t.Errorf("win nvidia 12.6: %v", g)
	}
	if c := first(recs, "cpu"); len(c) != 1 || c[0] != "llama-b11243-bin-win-cpu-x64.zip" {
		t.Errorf("win cpu: %v", c)
	}
	if g := first(recommendAssets(as, "windows", "amd64", nv, "13.5"), "gpu"); g[0] != "llama-b11243-bin-win-cuda-13.4-x64.zip" {
		t.Errorf("win nvidia 13.5: %v", g)
	}
	if g := first(recommendAssets(as, "linux", "amd64", nv, "12.9"), "gpu"); !reflect.DeepEqual(g, []string{"llama-b11243-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b11243-bin-ubuntu-cuda-12.8-x64.tar.gz"}) {
		t.Errorf("linux nvidia: %v", g)
	}
	if g := first(recommendAssets(as, "windows", "amd64", []DetectedGPU{{"AMD", "RX 6600"}}, ""), "gpu"); g[0] != "llama-b11243-bin-win-vulkan-x64.zip" {
		t.Errorf("win amd: %v", g)
	}
	if g := first(recommendAssets(as, "windows", "amd64", nil, ""), "gpu"); g != nil {
		t.Errorf("no gpu should have no gpu rec: %v", g)
	}
	if g := first(recommendAssets(as, "android", "arm64", []DetectedGPU{{"Qualcomm", "Adreno"}}, ""), "gpu"); g[0] != "llama-b11243-bin-android-arm64-snapdragon.tar.gz" {
		t.Errorf("android: %v", g)
	}
}

func TestPickCUDAFallback(t *testing.T) {
	as := []ReleaseAsset{{Name: "x-cuda-12.8-x64"}, {Name: "x-cuda-13.4-x64"}}
	a, ver, ok := pickCUDA(as, `cuda-(\d+\.\d+)-x64$`, "12.6")
	if a == nil || ver != "12.8" || ok {
		t.Errorf("fallback: %v %s %v", a, ver, ok)
	}
	if _, ver, ok := pickCUDA(as, `cuda-(\d+\.\d+)-x64$`, "13.5"); ver != "13.4" || !ok {
		t.Errorf("match: %s %v", ver, ok)
	}
}

func TestNetPath(t *testing.T) {
	cases := []struct {
		in   string
		ok   bool
		want netLoc
	}{
		{`\\NAS\Models\qwen`, true, netLoc{"nas", "/Models/qwen", true}},
		{"//nas/models/", true, netLoc{"nas", "/models", true}},
		{"smb://user@nas.local/share", true, netLoc{"nas.local", "/share", true}},
		{"nfs://10.0.0.2/export/llm", true, netLoc{"10.0.0.2", "/export/llm", false}},
		{"nas:/export/llm", true, netLoc{"nas", "/export/llm", false}},
		{`C:\models`, false, netLoc{}},
		{"C:/models", false, netLoc{}},
		{"/home/a/models", false, netLoc{}},
	}
	for _, c := range cases {
		got, ok := parseNetLoc(c.in)
		if ok != c.ok || got != c.want {
			t.Errorf("parseNetLoc(%q) = %+v,%v, want %+v,%v", c.in, got, ok, c.want, c.ok)
		}
	}

	mounts := []mountInfo{
		{netLoc{"nas", "/Models", true}, "/mnt/nas"},
		{netLoc{"nas", "/Models/big", true}, "/mnt/big"},
		{netLoc{"srv", "/export", false}, "/mnt/srv"},
	}
	for in, want := range map[string]string{
		"smb://NAS/models/qwen": "/mnt/nas/qwen",
		`\\nas\Models\big\x`:    "/mnt/big/x",
		"srv:/export/llm":       "/mnt/srv/llm",
		"nfs://srv/export":      "/mnt/srv",
		"nfs://srv/exports/llm": "",
		"smb://other/models":    "",
	} {
		loc, _ := parseNetLoc(in)
		if got := findMount(loc, mounts); got != want {
			t.Errorf("findMount(%q) = %q, want %q", in, got, want)
		}
	}

	if m, ok := gvfsEntry("/run/user/1000/gvfs", "smb-share:server=nas,share=models"); !ok || m.Loc != (netLoc{"nas", "/models", true}) {
		t.Errorf("gvfsEntry = %+v,%v", m, ok)
	}
	if got := unescapeMount(`/mnt/my\040share`); got != "/mnt/my share" {
		t.Errorf("unescapeMount = %q", got)
	}
}
