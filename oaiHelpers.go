package botMaker

import (
	"context"
	"github.com/pkoukk/tiktoken-go"
	"log"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

type OpenAIResponse struct {
	Response string `json:"response"`
	Tokens   int    `json:"tokens"`
}

// GetContexts will use OpenAI to get vectors for the prompt, then use Memory to retrieve relevant
// contexts to include in the query prompt
func GetContexts(b *BotPrompt, s *BotSettings, m Storage, c *OAIClient) ([]string, error) {
	if b.ContextToRender == nil {
		b.ContextToRender = make([]string, 0)
	}

	promptNoContext, err := b.renderPrompt(s)
	if err != nil {
		return nil, err
	}

	questionEmbedding, err := c.GetEmbeddingsForPrompt(b.Body, openai.AdaEmbeddingV2)
	if err != nil {
		return nil, err
	}

	//log.Println("[GetContexts] Question Embedding Length:", len(questionEmbedding))

	// step 2: Query Pinecone using questionEmbedding to get context matches
	matches, err := m.Retrieve(questionEmbedding, 3, s.ID)
	if err != nil {
		//log.Println("[QuestionHandler ERR] Pinecone query error\n", err.Error())
		return nil, err
	}

	//log.Println("[GetContexts] Got matches from Pinecone:", matches)

	// Extract context text and titles from the matches
	contexts := make([]Context, len(matches))
	for i, match := range matches {
		contexts[i].Text = match.Metadata["text"]
		contexts[i].Title = match.Metadata["title"]
	}
	//log.Println("[QuestionHandler] Retrieved context from Pinecone:\n", contexts)

	// step 3: Structure the prompt with a context section + question,
	//using top x results from pinecone as the context
	contextTexts := make([]string, len(contexts))
	for i, ctx := range contexts {
		contextTexts[i] = ctx.Text
	}

	// Count tokens for the question without context
	tke, err := tiktoken.EncodingForModel(s.Model)
	questionTokens := tke.Encode(promptNoContext, nil, nil)
	currentTokenCount := len(questionTokens)

	for i, _ := range contextTexts {
		// Count tokens for the current context
		contextTokens := tke.Encode(contextTexts[i], nil, nil)
		currentTokenCount += len(contextTokens)

		if currentTokenCount >= s.TokenLimit {
			break
		} else if i == len(contextTexts)-1 {
			b.ContextToRender = append(b.ContextToRender,
				strings.Trim(strings.ReplaceAll(
					contextTexts[i], "\n", " "), " "))
		}
	}

	return b.ContextToRender, nil
}

type OAIClient struct {
	Client *openai.Client
}

func NewOAIClient(key string) *OAIClient {
	return &OAIClient{
		Client: openai.NewClient(key),
	}
}

func (c *OAIClient) CallUnifiedCompletionAPI(settings *BotSettings, prompt *BotPrompt) (string, int, error) {
	var assistantMessage string
	var tokens int
	var err error

	if settings.Model == openai.GPT3TextDavinci003 {
		assistantMessage, tokens, err = c.useCompletionAPI(prompt, settings)
	} else {
		assistantMessage, tokens, err = c.useChatCompletionAPI(prompt, settings)
	}

	return assistantMessage, tokens, err
}

func (c *OAIClient) useChatCompletionAPI(prompt *BotPrompt, s *BotSettings) (string, int, error) {
	cp, err := prompt.AsChatCompletionRequest(s)
	if err != nil {
		return "", 0, nil
	}

	resp, err := c.Client.CreateChatCompletion(
		context.Background(),
		*cp,
	)

	if err != nil {
		return "", 0, err
	}

	return resp.Choices[0].Message.Content, resp.Usage.TotalTokens, nil
}

func (c *OAIClient) useCompletionAPI(prompt *BotPrompt, s *BotSettings) (string, int, error) {
	comp, err := prompt.AsCompletionRequest(s)
	if err != nil {
		return "", 0, nil
	}
	resp, err := c.Client.CreateCompletion(
		context.Background(),
		*comp,
	)

	if err != nil {
		return "", 0, err
	}

	// TODO: Check this
	return resp.Choices[0].Text, resp.Usage.TotalTokens, nil
}

func (c *OAIClient) callEmbeddingAPIWithRetry(texts []string, embedModel openai.EmbeddingModel,
	maxRetries int) (*openai.EmbeddingResponse, error) {
	var err error
	var res openai.EmbeddingResponse

	for i := 0; i < maxRetries; i++ {
		res, err = c.Client.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
			Input: texts,
			Model: embedModel,
		})

		if err == nil {
			return &res, nil
		}

		time.Sleep(5 * time.Second)
	}

	return nil, err
}

// getEmbeddingsForData gets embedding vectors for data to be ingested and used for context in queries
func (c *OAIClient) getEmbeddingsForData(chunks []Chunk, batchSize int,
	embedModel openai.EmbeddingModel) ([][]float32, error) {
	embeddings := make([][]float32, 0, len(chunks))

	for i := 0; i < len(chunks); i += batchSize {
		iEnd := min(len(chunks), i+batchSize)

		texts := make([]string, 0, iEnd-i)
		for _, chunk := range chunks[i:iEnd] {
			texts = append(texts, chunk.Text)
		}

		log.Printf("[oaiclient] getting embeddings for chunk %d -> %d (of %d)", i, iEnd, len(chunks))

		res, err := c.callEmbeddingAPIWithRetry(texts, embedModel, 3)
		if err != nil {
			return nil, err
		}

		embeds := make([][]float32, len(res.Data))
		for i, record := range res.Data {
			embeds[i] = record.Embedding
		}

		embeddings = append(embeddings, embeds...)
	}

	return embeddings, nil
}

// GetEmbeddingsForPrompt will return embedding vectors for the prompt
func (c *OAIClient) GetEmbeddingsForPrompt(text string, embedModel openai.EmbeddingModel) ([]float32, error) {
	res, err := c.callEmbeddingAPIWithRetry([]string{text}, embedModel, 3)
	if err != nil {
		return nil, err
	}

	return res.Data[0].Embedding, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}