package main

import (
	"bufio"
	"fmt"
	"github.com/lonelycode/botMaker"
	"os"
)

func main() {
	cfg := botMaker.NewConfigFromEnv()
	oai := botMaker.NewOAIClient(cfg.OpenAPIKey)
	settings := botMaker.NewBotSettings()
	settings.ID = "a45dbe63-4207-419c-bca7-5d940bf3d908"

	// For adding context, you can attach a memory store to query
	settings.Memory = &botMaker.Pinecone{
		APIEndpoint: cfg.PineconeEndpoint,
		APIKey:      cfg.PineconeKey,
		UUID:        "a45dbe63-4207-419c-bca7-5d940bf3d908",
	}

	prompt := botMaker.NewBotPrompt("", oai)
	prompt.Instructions = "You are an AI chatbot that is funny and helpful"

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("Chat: ")
		text, _ := reader.ReadString('\n')

		prompt.Body = text
		if text == "quit\n" {
			break
		}

		resp, _, err := oai.CallUnifiedCompletionAPI(settings, prompt)
		if err != nil {
			fmt.Println(err)
		}

		fmt.Println(resp)

		oldBody := "Human: " + prompt.Body
		prompt.ContextToRender = append(prompt.ContextToRender, oldBody)

		// Next make sure the AI's response is added too
		prompt.ContextToRender = append(prompt.ContextToRender, resp)
	}

}
