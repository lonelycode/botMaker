package main

import (
	"bufio"
	"fmt"
	"github.com/lonelycode/botMaker"
	"github.com/sashabaranov/go-openai"
	"math/rand"
	"os"
	"time"
)

func main() {
	// Get a namespace
	if len(os.Args) < 2 {
		fmt.Println("please provide a namespace for the bot to use as an argument")
		return
	}
	namespace := os.Args[1]

	// For the typewriter, not important
	var min int64 = 5
	var max int64 = 30

	// Get the system config (API keys and Pinecone endpoint)
	cfg := botMaker.NewConfigFromEnv()

	// Set up the OAI API client
	oai := botMaker.NewOAIClient(cfg.OpenAPIKey)

	// Get the tuning for the bot, we'll use some defaults
	settings := botMaker.NewBotSettings()

	// We set the ID for the bot as this will be used when querying
	// pinecone for context embeddings specifically for this bot -
	// use different IDs for difference PC namespaces to create
	// different context-flavours for bots
	settings.ID = namespace
	settings.Model = openai.GPT3TextDavinci003
	settings.Temp = 0.9
	settings.TopP = 0.9
	settings.MaxTokens = 4096 // need to set this for 3.5 turbo

	// If adding context (additional data outside of GPTs training data), y
	// you can attach a memory store to query
	settings.Memory = &botMaker.Pinecone{
		APIEndpoint: cfg.PineconeEndpoint,
		APIKey:      cfg.PineconeKey,
	}

	// the Prompt holds all the information and logic needed to make a query to OpenAI,
	// to change the way the prompt is presented to the AI, provide a text/template
	// (see DEFAULT_TEMPLATE in botMaker.go for the default). this can also be a file
	// path to the template
	prompt := botMaker.NewBotPrompt("", oai)

	// Set an initial instruction to the bot
	prompt.Instructions = "You are an AI chatbot that is funny and helpful"

	Typewriter("Hi, I'm Globutron, your friendly neighborhood chatbot powered by GPT3. Let's chat!", min, max)

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nInput: ")
		text, _ := reader.ReadString('\n')

		// We populate the Body with the query from the user
		prompt.Body = text
		if text == "quit\n" {
			break
		}

		// make the OpenAI query, the prompt object will render the query
		// according to its template with the context embeddings pulled from Pinecone
		resp, _, err := oai.CallUnifiedCompletionAPI(settings, prompt)
		if err != nil {
			fmt.Println(err)
		}

		// Show the response
		Typewriter("\n"+resp, min, max)
		Typewriter(fmt.Sprintf("(Contexts: %d)", len(prompt.GetContextsForLastPrompt())), min, max)

		oldBody := prompt.Body
		if settings.Model != openai.GPT3Dot5Turbo {
			withRole := "user: " + oldBody
			prompt.ContextToRender = append(prompt.ContextToRender, withRole)

			// Next make sure the AI's response is added too
			prompt.ContextToRender = append(prompt.ContextToRender, resp)
		} else {
			prompt.History = append(prompt.History,
				&botMaker.RenderContext{
					Role:    openai.ChatMessageRoleUser,
					Content: oldBody,
				},
				&botMaker.RenderContext{
					Role:    openai.ChatMessageRoleAssistant,
					Content: resp,
				})
		}

	}

}

var rng = rand.New(rand.NewSource(time.Now().Unix()))

func Typewriter(data string, min, max int64) {
	for _, c := range data {
		fmt.Printf("%c", c)
		d := rng.Int63n(max - min)
		time.Sleep(time.Millisecond*time.Duration(min) + time.Millisecond*time.Duration(d))
	}
	fmt.Print("\n")
}
