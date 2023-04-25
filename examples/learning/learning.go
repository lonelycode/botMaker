package main

import (
	"fmt"
	"github.com/lonelycode/botMaker"
	"github.com/sashabaranov/go-openai"
	"log"
	"os"
	"path/filepath"
)

func main() {
	// Check if there are enough arguments
	if len(os.Args) < 3 {
		fmt.Println("please provide at least two arguments: namespace and filename (can be a directory)")
		return
	}

	// Get the first two arguments
	namespace := os.Args[1]
	fileOrDir := os.Args[2]

	fileInfo, err := os.Stat(fileOrDir)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Check if it is a directory
	if fileInfo.IsDir() {
		files, err := os.ReadDir(fileOrDir)
		if err != nil {
			log.Println(err)
			return
		}

		for _, file := range files {
			if !file.IsDir() && !isHidden(file.Name()) {
				fPath := filepath.Join(fileOrDir, file.Name())
				err := learnFile(namespace, fPath)
				if err != nil {
					log.Printf("failed to learn file: %v", err)
					continue
				}
			}
		}
	} else {
		err := learnFile(namespace, fileOrDir)
		if err != nil {
			log.Println(err)
			return
		}
	}
}

func isHidden(name string) bool {
	return len(name) > 0 && name[0] == '.'
}

func learnFile(ns, filename string) error {
	log.Printf("starting %v\n", filename)
	cfg := botMaker.NewConfigFromEnv()

	// Client
	cl := botMaker.NewOAIClient(cfg.OpenAPIKey)

	// Create some storage
	pc := &botMaker.Pinecone{
		APIEndpoint: cfg.PineconeEndpoint,
		APIKey:      cfg.PineconeKey,
		UUID:        ns,
	}

	l := botMaker.Learn{
		Model:      openai.GPT3TextDavinci003,
		TokenLimit: 8191,
		ChunkSize:  20,
		Memory:     pc,
		Client:     cl,
	}

	count, err := l.FromFile(filename)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("processed %v with %v tokens into %v\n", filename, count, pc.UUID)
	return nil
}
