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
	var min int64 = 5
	var max int64 = 30

	cfg := botMaker.NewConfigFromEnv()
	oai := botMaker.NewOAIClient(cfg.OpenAPIKey)
	settings := botMaker.NewBotSettings()
	settings.ID = "a45dbe63-4207-419c-bca7-5d940bf3d908"

	// For adding context, you can attach a memory store to query
	settings.Memory = &botMaker.Pinecone{
		APIEndpoint: cfg.PineconeEndpoint,
		APIKey:      cfg.PineconeKey,
	}

	prompt := botMaker.NewBotPrompt("", oai)
	prompt.Instructions = "You are an AI chatbot that is funny and helpful"

	Typewriter("Hi, I'm Globutron, your friendly neighborhood chatbot powered by GPT3. Let's chat!", min, max)

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nInput: ")
		text, _ := reader.ReadString('\n')

		prompt.Body = text
		if text == "quit\n" {
			break
		}

		resp, _, err := oai.CallUnifiedCompletionAPI(settings, prompt)
		if err != nil {
			fmt.Println(err)
		}

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
