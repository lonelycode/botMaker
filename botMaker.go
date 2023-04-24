package botMaker

import (
	"bytes"
	"fmt"
	"github.com/pkoukk/tiktoken-go"
	"github.com/sashabaranov/go-openai"
	"log"
	"os"
	"strings"
	"text/template"
)

var DEFAULT_TEMPLATE = `
{{if .Instructions}}{{.Instructions}}{{end}}
{{ if .ContextToRender }}Use the following context to help with your response:
{{ range $ctx := .ContextToRender }}
Context: {{$ctx}}
{{ end }}{{ end }}
Human: {{.Body}}
{{ if .DesiredFormat }}Provide your output using the following format:
{{.DesiredFormat}}{{ end }}
`

// BotSettings holds configs for OpenAI APIs
type BotSettings struct {
	ID               string // Used when retrieving contexts
	Model            string
	Temp             float32
	TopP             float32
	FrequencyPenalty float32
	PresencePenalty  float32
	MaxTokens        int // Max to receive
	TokenLimit       int // Max to send
	EmbeddingModel   openai.EmbeddingModel
	Memory           Storage
}

// NewBotSettings Returns settings for OpenAI with sane defaults
func NewBotSettings() *BotSettings {
	return &BotSettings{
		Temp:             0.6,
		TopP:             0.6,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.6,
		Model:            openai.GPT3TextDavinci003,
		MaxTokens:        4096,
		TokenLimit:       4096,
		EmbeddingModel:   openai.AdaEmbeddingV2,
	}
}

// BotPrompt has the components to make a call to OpenAPI
type BotPrompt struct {
	OAIClient       *OAIClient
	Instructions    string   // You are an AI assistant that is happy, helpful and tries to offer insightful answers
	Body            string   // The actual prompt
	DesiredFormat   string   // Provide your answer using the following output
	ContextToRender []string // Rendered context (within token limit)
	Stop            []string // Human: AI:
	Template        string
	RenderedPrompt  string
	PromptLength    int
	tpl             *template.Template
}

type Context struct {
	Text  string `json:"text"`
	Title string `json:"title"`
}

func NewBotPrompt(promptTemplate string, withClient *OAIClient) *BotPrompt {
	b := &BotPrompt{
		OAIClient: withClient,
	}

	b.Template = promptTemplate

	if b.Template == "" {
		b.Template = DEFAULT_TEMPLATE
	}

	// load from a file if it's a file
	pf := strings.HasPrefix(promptTemplate, "file://")
	if pf {
		c, err := os.ReadFile(promptTemplate)
		if err != nil {
			log.Fatal(err)
		}

		b.Template = string(c)
	}

	return b
}

func (b *BotPrompt) renderPrompt() (string, error) {
	var out bytes.Buffer
	err := b.tpl.Execute(&out, b)
	if err != nil {
		return "", err
	}

	return out.String(), nil
}

// Prompt renders the prompt to the prompt template
func (b *BotPrompt) Prompt(settings *BotSettings) (string, error) {
	var err error
	if b.tpl == nil {
		b.tpl = template.New("prompt-tpl")
		b.tpl, err = b.tpl.Parse(b.Template)
		if err != nil {
			return "", err
		}
	}

	// check for context or memory to embed
	if settings.Memory != nil {
		_, err := GetContexts(b, settings, settings.Memory, b.OAIClient)
		if err != nil {
			return "", err
		}
	}

	// render it again
	finalPrompt, err := b.renderPrompt()
	if err != nil {
		return "", err
	}

	if !CheckTokenLimit(finalPrompt, settings.Model, settings.TokenLimit) {
		return "", fmt.Errorf("prompt is longer than token limit, please shorten your prompt")
	}

	// save this
	b.RenderedPrompt = finalPrompt
	b.PromptLength, _ = CountTokens(finalPrompt, settings.Model)

	return finalPrompt, nil
}

func CountTokens(text, model string) (int, error) {
	// Get tiktoken encoding for the model
	tke, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return 0, err
	}

	// Count tokens for the question
	questionTokens := tke.Encode(text, nil, nil)
	return len(questionTokens), nil
}

func CheckTokenLimit(text, model string, tokenLimit int) bool {
	// Get tiktoken encoding for the model
	tke, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return false
	}

	// Count tokens for the question
	questionTokens := tke.Encode(text, nil, nil)
	currentTokenCount := len(questionTokens)

	//fmt.Printf("TOKENS: %d", len(questionTokens))

	if currentTokenCount >= tokenLimit {
		return false
	}

	return true
}

func (b *BotPrompt) AsCompletionRequest(s *BotSettings) (*openai.CompletionRequest, error) {
	p, err := b.Prompt(s)
	if err != nil {
		return nil, err
	}

	return &openai.CompletionRequest{
		Model:            s.Model,
		Prompt:           p,
		Temperature:      s.Temp,
		MaxTokens:        s.MaxTokens - b.PromptLength,
		TopP:             s.TopP,
		FrequencyPenalty: s.FrequencyPenalty,
		PresencePenalty:  s.PresencePenalty,
		Stop:             b.Stop,
	}, nil
}

func (b *BotPrompt) AsChatCompletionRequest(s *BotSettings) (*openai.ChatCompletionRequest, error) {
	p, err := b.Prompt(s)
	if err != nil {
		return nil, err
	}

	messages := []openai.ChatCompletionMessage{
		{
			Role:    "system",
			Content: b.Instructions,
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: p,
		},
	}

	return &openai.ChatCompletionRequest{
		Model:            s.Model,
		Messages:         messages,
		Temperature:      s.Temp,
		MaxTokens:        s.MaxTokens,
		TopP:             s.TopP,
		FrequencyPenalty: s.FrequencyPenalty,
		PresencePenalty:  s.PresencePenalty,
		Stop:             b.Stop,
	}, nil
}
