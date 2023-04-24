package main

import (
	"bufio"
	"fmt"
	"github.com/lonelycode/botMaker"
	"math/rand"
	"os"
	"time"
)

func main() {
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
	settings.ID = "a45dbe63-4207-419c-bca7-5d940bf3d908"

	// If adding context (additional data outside of GPTs training data), y
	// you can attach a memory store to query
	settings.Memory = &botMaker.Pinecone{
		APIEndpoint: cfg.PineconeEndpoint,
		APIKey:      cfg.PineconeKey,
	}

	// the Prompt holds all the information and logic needed to make a query to OpenAI,
	// to change the way the prompt is presented to the AI, provide a text/template
	// (see DEFAULT_TEMPLATE in botMaker.go for the default).
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

		oldBody := "Human: " + prompt.Body
		prompt.ContextToRender = append(prompt.ContextToRender, oldBody)

		// Next make sure the AI's response is added too
		prompt.ContextToRender = append(prompt.ContextToRender, resp)
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
