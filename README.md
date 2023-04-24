# botMaker

A Go library to help create and train AI bots quickly with OpenA and PineCone. It is heavily based on code from the excellent [vault-ai project](https://github.com/pashpashpash/vault-ai).

## Sample Usage

### A simple chatbot:

```go
func Chat() {
    cfg := Config{
        OpenAPIKey:       "xxxx",
        PineconeKey:      "xxxx",
        PineconeEndpoint: "xxxx",
    }
    
    // Client
    cl := NewOAIClient(cfg.OpenAPIKey)
    
    // Settings for the AI
    bs := NewBotSettings()
    bs.ID = "a-UUID-here"
    
    // Build a prompt using the default template
    pr := NewBotPrompt("", cl)
    pr.Instructions = "You are an AI assistant that provides answers that are helpful in a friendly and cheerful way."
    pr.Body = "What is the best way to scale a redis database?"
    
    // Create some storage
    pc := &Pinecone{
        APIEndpoint: cfg.PineconeEndpoint,
        APIKey:      cfg.PineconeKey,
        UUID:        bs.ID,
    }
    
    // attach memory
    bs.Memory = pc
    
    oaiResponse, _, err := cl.CallUnifiedCompletionAPI(bs, pr)
        if err != nil {
       fatal("query send fail: %v", err)
    }
    
    fmt.Println("FIRST PROMPT:")
    fmt.Println(pr.RenderedPrompt)
    
    fmt.Println("GOT FIRST RESPONSE: ")
    fmt.Println(oaiResponse)
    time.Sleep(5 * time.Second)
    
    // we do some string shenanigans to make a chatbot
    // First: Update the context
    oldBody := "Human: " + pr.Body
    pr.ContextToRender = append(pr.ContextToRender, oldBody)
    
    // Next make sure the AI's response is added too
    pr.ContextToRender = append(pr.ContextToRender, oaiResponse)
    
    // Replace the main query with a new one
    pr.Body = "How is a cluster different from sentinel?"
    
    // Make the call!
    secondResponse, _, err := cl.CallUnifiedCompletionAPI(bs, pr)
    if err != nil {
        fatal("prompt2 fail: %v", err)
    }
    
    fmt.Println("SECOND RESPONSE")
    fmt.Println(secondResponse)
}
```

### Learning from a PDF

```go
func TestLearning() {
	cfg := Config{
		OpenAPIKey:       "xxx",
		PineconeKey:      "xxx",
		PineconeEndpoint: "xxx",
	}

	// Client
	cl := NewOAIClient(cfg.OpenAPIKey)

	// Create some storage
	pc := &Pinecone{
		APIEndpoint: cfg.PineconeEndpoint,
		APIKey:      cfg.PineconeKey,
		UUID:        "a45dbe63-4207-419c-bca7-5d940bf3d908",
	}

	l := Learn{
		Model:      openai.GPT3TextDavinci003,
		TokenLimit: 8191,
		ChunkSize:  20,
		Memory:     pc,
		Client:     cl,
	}

	_, err := l.FromFile("/data/socrates.pdf")
	if err != nil {
		fatal(err)
	}
}
```