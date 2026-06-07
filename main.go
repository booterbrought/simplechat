package main

import (
	"bufio"
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
)

//go:embed static
var static embed.FS

type Settings struct {
	Model        string  `json:"model"`
	SystemPrompt string  `json:"system_prompt"`
	Temperature  float64 `json:"temperature"`
	MaxTokens    int     `json:"max_tokens"`
}

var (
	settings     Settings
	settingsMu   sync.RWMutex
	settingsPath = "settings.json"
)

func defaultSettings() Settings {
	return Settings{
		Model:        "gpt-4o",
		SystemPrompt: "You are a helpful assistant.",
		Temperature:  0.7,
		MaxTokens:    4096,
	}
}

func loadEnvFile(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		if strings.HasPrefix(val, `"`) && strings.HasSuffix(val, `"`) {
			val = val[1 : len(val)-1]
		}
		if os.Getenv(key) == "" {
			os.Setenv(key, val)
		}
	}
}

func loadSettings() {
	settingsMu.Lock()
	defer settingsMu.Unlock()

	settings = defaultSettings()

	data, err := os.ReadFile(settingsPath)
	if err != nil {
		return
	}

	var s Settings
	if err := json.Unmarshal(data, &s); err != nil {
		return
	}

	if s.Model != "" {
		settings.Model = s.Model
	}
	if s.SystemPrompt != "" {
		settings.SystemPrompt = s.SystemPrompt
	}
	if s.Temperature > 0 {
		settings.Temperature = s.Temperature
	}
	if s.MaxTokens > 0 {
		settings.MaxTokens = s.MaxTokens
	}
}

func saveSettingsLocked() error {
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(settingsPath, data, 0600)
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Messages []ChatMessage `json:"messages"`
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"Invalid JSON"}`, http.StatusBadRequest)
		return
	}

	if len(req.Messages) == 0 {
		http.Error(w, `{"error":"No messages provided"}`, http.StatusBadRequest)
		return
	}

	apiEndpoint := strings.TrimRight(os.Getenv("GOPCHAT_API_ENDPOINT"), "/")
	if apiEndpoint == "" {
		apiEndpoint = "https://api.openai.com/v1"
	}

	apiKey := os.Getenv("GOPCHAT_API_KEY")
	if apiKey == "" {
		http.Error(w, `{"error":"GOPCHAT_API_KEY не задан. Создайте .env файл."}`, http.StatusInternalServerError)
		return
	}

	settingsMu.RLock()
	model := settings.Model
	systemPrompt := settings.SystemPrompt
	temperature := settings.Temperature
	maxTokens := settings.MaxTokens
	settingsMu.RUnlock()

	messages := []ChatMessage{
		{Role: "system", Content: systemPrompt},
	}
	messages = append(messages, req.Messages...)

	openaiReq := map[string]interface{}{
		"model":       model,
		"messages":    messages,
		"temperature": temperature,
		"max_tokens":  maxTokens,
		"stream":      true,
	}

	body, err := json.Marshal(openaiReq)
	if err != nil {
		http.Error(w, `{"error":"Failed to encode request"}`, http.StatusInternalServerError)
		return
	}

	url := apiEndpoint + "/chat/completions"

	proxyReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"Failed to create request: %s"}`, err.Error()), http.StatusInternalServerError)
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")
	proxyReq.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(proxyReq)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"Failed to reach API: %s"}`, err.Error()), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		fmt.Fprintf(w, `{"error":"API returned status %d"}`, resp.StatusCode)
		if len(errBody) > 0 {
			fmt.Fprintf(w, "\n")
		}
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error":"Streaming not supported"}`, http.StatusInternalServerError)
		return
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		fmt.Fprintf(w, "%s\n", line)
		flusher.Flush()
	}
}

func main() {
	loadEnvFile(".env")

	if _, err := os.Stat(settingsPath); os.IsNotExist(err) {
		data, _ := json.MarshalIndent(defaultSettings(), "", "  ")
		os.WriteFile(settingsPath, data, 0600)
	}
	loadSettings()

	staticFS, err := fs.Sub(static, "static")
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/chat", handleChat)
	mux.Handle("/", http.FileServer(http.FS(staticFS)))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Printf("ГОПчат запущен на http://localhost:%s\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
