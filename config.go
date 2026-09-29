package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
)

// Params 是 llama-server 的启动参数。指针为 nil 表示“未设置”，
// 模型参数未设置的项回退到全局参数，全局未设置的项回退到内置默认值。
type Params struct {
	Mode          *string  `json:"mode,omitempty"`            // cpu / gpu
	Device        *string  `json:"device,omitempty"`          // --device，如 CUDA0、Vulkan0
	NGpuLayers    *FlexStr `json:"nGpuLayers,omitempty"`      // -ngl：数字、auto 或 all
	CtxSize       *int     `json:"ctxSize,omitempty"`         // -c
	Threads       *int     `json:"threads,omitempty"`         // -t
	BatchSize     *int     `json:"batchSize,omitempty"`       // -b
	UBatchSize    *int     `json:"ubatchSize,omitempty"`      // -ub
	Parallel      *int     `json:"parallel,omitempty"`        // -np
	FlashAttn     *string  `json:"flashAttn,omitempty"`       // -fa on/off/auto
	CacheTypeK    *string  `json:"cacheTypeK,omitempty"`      // -ctk
	CacheTypeV    *string  `json:"cacheTypeV,omitempty"`      // -ctv
	Mlock         *bool    `json:"mlock,omitempty"`           // --mlock
	NoMmap        *bool    `json:"noMmap,omitempty"`          // --no-mmap
	Jinja         *bool    `json:"jinja,omitempty"`           // --jinja
	Temp          *float64 `json:"temp,omitempty"`            // --temp
	TopK          *int     `json:"topK,omitempty"`            // --top-k
	TopP          *float64 `json:"topP,omitempty"`            // --top-p
	MinP          *float64 `json:"minP,omitempty"`            // --min-p
	RepeatPenalty *float64 `json:"repeatPenalty,omitempty"`   // --repeat-penalty
	PresencePen   *float64 `json:"presencePenalty,omitempty"` // --presence-penalty
	Host          *string  `json:"host,omitempty"`            // --host
	Port          *int     `json:"port,omitempty"`            // --port，0 表示自动分配
	APIKey        *string  `json:"apiKey,omitempty"`          // --api-key
	Alias         *string  `json:"alias,omitempty"`           // -a（仅模型参数有意义）
	ExtraArgs     *string  `json:"extraArgs,omitempty"`       // 附加参数
}

// LastRun 记录模型组上次的启动选择，作为下次启动的默认值。
type LastRun struct {
	Variant string `json:"variant"` // 模型文件路径
	Vision  bool   `json:"vision"`
	Mmproj  string `json:"mmproj"`
	Extra   string `json:"extra"`
}

type Config struct {
	LlamaServerPath string             `json:"llamaServerPath,omitempty"` // 旧版字段，读取时迁移到 ServerPathGPU
	ServerPathGPU   string             `json:"serverPathGpu"`             // GPU 版（CUDA/Vulkan/ROCm…），为空时从 PATH 查找
	ServerPathCPU   string             `json:"serverPathCpu"`             // CPU 版，为空时使用 GPU 版并禁用 GPU
	Proxy           string             `json:"proxy"`                     // http://、https:// 或 socks5:// 代理
	ProxyAll        bool               `json:"proxyAll"`                  // false：仅 HuggingFace 走代理
	LastRun         map[string]LastRun `json:"lastRun"`
	ModelDirs       []string           `json:"modelDirs"`
	AllowMulti      bool               `json:"allowMulti"`  // 是否允许同时运行多个模型
	MaxLogLines     int                `json:"maxLogLines"` // 每个实例保留/展示的日志行数
	BasePort        int                `json:"basePort"`    // 自动分配端口的起始值
	HFEndpoint      string             `json:"hfEndpoint"`  // 可改为 https://hf-mirror.com
	Global          Params             `json:"global"`
	Models          map[string]Params  `json:"models"` // key 为模型组 key
}

func defaultParams() Params {
	return Params{
		Mode:       ptr("gpu"),
		NGpuLayers: ptr(FlexStr("999")),
		CtxSize:    ptr(4096),
		Host:       ptr("127.0.0.1"),
		Port:       ptr(0),
		FlashAttn:  ptr("auto"),
		Jinja:      ptr(true),
	}
}

func ptr[T any](v T) *T { return &v }

// FlexStr 可从 JSON 数字或字符串解析。
type FlexStr string

func (f *FlexStr) UnmarshalJSON(b []byte) error {
	var s string
	if json.Unmarshal(b, &s) == nil {
		*f = FlexStr(s)
		return nil
	}
	var n json.Number
	if err := json.Unmarshal(b, &n); err != nil {
		return err
	}
	*f = FlexStr(n.String())
	return nil
}

// merge 用 over 中已设置的字段覆盖 base。
func merge(base, over Params) Params {
	b, _ := json.Marshal(base)
	o, _ := json.Marshal(over)
	m := map[string]json.RawMessage{}
	_ = json.Unmarshal(b, &m)
	om := map[string]json.RawMessage{}
	_ = json.Unmarshal(o, &om)
	for k, v := range om {
		m[k] = v
	}
	var out Params
	nb, _ := json.Marshal(m)
	_ = json.Unmarshal(nb, &out)
	return out
}

type ConfigStore struct {
	mu   sync.RWMutex
	path string
	cfg  Config
}

func LoadConfig(path string) (*ConfigStore, error) {
	s := &ConfigStore{path: path}
	s.cfg = Config{MaxLogLines: 1000, BasePort: 8080, HFEndpoint: "https://huggingface.co", Models: map[string]Params{}, LastRun: map[string]LastRun{}}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return s, s.save()
		}
		return nil, err
	}
	if err := json.Unmarshal(data, &s.cfg); err != nil {
		return nil, err
	}
	s.normalize()
	return s, nil
}

func (s *ConfigStore) normalize() {
	if s.cfg.MaxLogLines <= 0 {
		s.cfg.MaxLogLines = 1000
	}
	if s.cfg.BasePort <= 0 {
		s.cfg.BasePort = 8080
	}
	if s.cfg.HFEndpoint == "" {
		s.cfg.HFEndpoint = "https://huggingface.co"
	}
	if s.cfg.Models == nil {
		s.cfg.Models = map[string]Params{}
	}
	if s.cfg.LastRun == nil {
		s.cfg.LastRun = map[string]LastRun{}
	}
	if s.cfg.LlamaServerPath != "" && s.cfg.ServerPathGPU == "" {
		s.cfg.ServerPathGPU = s.cfg.LlamaServerPath
	}
	s.cfg.LlamaServerPath = ""
}

func (s *ConfigStore) save() error {
	data, err := json.MarshalIndent(s.cfg, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(s.path); dir != "" {
		_ = os.MkdirAll(dir, 0o755)
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

func (s *ConfigStore) Get() Config {
	s.mu.RLock()
	defer s.mu.RUnlock()
	c := s.cfg
	c.ModelDirs = append([]string(nil), s.cfg.ModelDirs...)
	c.Models = make(map[string]Params, len(s.cfg.Models))
	for k, v := range s.cfg.Models {
		c.Models[k] = v
	}
	c.LastRun = make(map[string]LastRun, len(s.cfg.LastRun))
	for k, v := range s.cfg.LastRun {
		c.LastRun[k] = v
	}
	return c
}

// UpdateSettings 更新除模型参数以外的配置。
func (s *ConfigStore) UpdateSettings(c Config) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	models, last := s.cfg.Models, s.cfg.LastRun
	s.cfg = c
	s.cfg.Models, s.cfg.LastRun = models, last
	s.normalize()
	return s.save()
}

func (s *ConfigStore) SetModelParams(path string, p Params) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if p == (Params{}) {
		delete(s.cfg.Models, path)
	} else {
		s.cfg.Models[path] = p
	}
	return s.save()
}

func (s *ConfigStore) SetLastRun(key string, l LastRun) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cfg.LastRun[key] = l
	return s.save()
}

// Effective 返回模型最终生效的参数：内置默认 -> 全局 -> 模型。
func (s *ConfigStore) Effective(path string) Params {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return merge(merge(defaultParams(), s.cfg.Global), s.cfg.Models[path])
}

func (s *ConfigStore) Dir() string { return filepath.Dir(s.path) }
