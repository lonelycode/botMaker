package botMaker

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"strings"
	"text/template"

	"github.com/pkoukk/tiktoken-go"
	"github.com/sashabaranov/go-openai"
)

var DEFAULT_TEMPLATE = `
{{ if .ContextToRender }}Use the following context to help with your response:
{{ range $ctx := .ContextToRender }}
{{$ctx}}
{{ end }}
===={{ end }}

user: {{.Body}}
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

type RenderContext struct {
	Role    string
	Content string
}

// BotPrompt has the components to make a call to OpenAPI
type BotPrompt struct {
	OAIClient       LLMAPIClient
	Instructions    string   // You are an AI assistant that is happy, helpful and tries to offer insightful answers
	Body            string   // The actual prompt
	DesiredFormat   string   // Provide your answer using the following output
	ContextToRender []string // Rendered context (within token limit)
	ContextTitles   []string // titles and references to the content
	Stop            []string // Human: AI:
	History         []*RenderContext
	Template        string
	RenderedPrompt  string
	PromptLength    int
	tpl             *template.Template
}

type Context struct {
	Text  string `json:"text"`
	Title string `json:"title"`
}

func NewBotPrompt(promptTemplate string, withClient LLMAPIClient) *BotPrompt {
	b := &BotPrompt{
		OAIClient:       withClient,
		ContextToRender: make([]string, 0),
		ContextTitles:   make([]string, 0),
		Stop:            make([]string, 0),
		History:         make([]*RenderContext, 0),
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

func (b *BotPrompt) GetContextsForLastPrompt() []string {
	if len(b.ContextToRender) > 0 {
		return b.ContextToRender
	}

	return make([]string, 0)
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

	if !b.OAIClient.CheckTokenLimit(finalPrompt, settings.Model, settings.TokenLimit) {
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

	var messages = make([]openai.ChatCompletionMessage, 0)
	// Instructions
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: b.Instructions,
	})

	// Context
	for i, _ := range b.History {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    b.History[i].Role,
			Content: b.History[i].Content,
		})
	}

	// Prompt
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: p,
	})

	numTokens := 1 // odd, but necessary
	for _, m := range messages {
		numTokens += 4
		rC, _ := CountTokens(m.Role, s.Model)
		numTokens += rC
		cC, _ := CountTokens(m.Content, s.Model)
		numTokens += cC
	}
	numTokens += 2

	// can't be 0
	mtokens := s.MaxTokens - numTokens
	if mtokens < 1 {
		return nil, fmt.Errorf("prompt is too long")
	}

	return &openai.ChatCompletionRequest{
		Model:            s.Model,
		Messages:         messages,
		Temperature:      s.Temp,
		MaxTokens:        mtokens,
		TopP:             s.TopP,
		FrequencyPenalty: s.FrequencyPenalty,
		PresencePenalty:  s.PresencePenalty,
		Stop:             b.Stop,
	}, nil
}
