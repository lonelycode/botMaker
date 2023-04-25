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

// CODE_TEMPLATE is much shorter than the default (chatbot) template as we do not have any embeddings
var CODE_TEMPLATE = `
Human: {{.Body}}
{{ if .DesiredFormat }}Provide your output using the following format:
{{.DesiredFormat}}{{ end }}
`

func main() {
	// For the typewriter, not important
	var min int64 = 5
	var max int64 = 15

	// Get the system config (API keys and Pinecone endpoint)
	cfg := botMaker.NewConfigFromEnv()

	// Set up the OAI API client
	oai := botMaker.NewOAIClient(cfg.OpenAPIKey)

	// Get the tuning for the bot, we'll use specialist code one and up the temp to make answers stricter
	settings := botMaker.NewBotSettings()
	settings.Model = openai.GPT3Dot5Turbo
	settings.Temp = 0.9
	settings.TopP = 0.9
	settings.MaxTokens = 3500 // need to set this for 3.5 turbo

	// the Prompt holds all the information and logic needed to make a query to OpenAI,
	// to change the way the prompt is presented to the AI, provide a text/template
	// (see DEFAULT_TEMPLATE in botMaker.go for the default). this can also be a file
	// path to the template
	prompt := botMaker.NewBotPrompt(CODE_TEMPLATE, oai)

	// Set an initial instruction to the bot
	prompt.Instructions = "You are an AI coding assistant that provides concise and helpful answers to users"

	Typewriter("Hi, I'm Codurama!, your friendly neighborhood code assistant powered by GPT3. Let's code!!", min, max)

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
		//Typewriter(fmt.Sprintf("(Contexts: %d)", len(prompt.GetContextsForLastPrompt())), min, max)

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
