package botMaker

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
)

type Storage interface {
	Retrieve(questionEmbedding []float32, topK int, uuid string) ([]QueryMatch, error)
	UploadEmbeddings(embeddings [][]float32, chunks []Chunk) error
}

type Pinecone struct {
	APIEndpoint string
	APIKey      string
	UUID        string // Used when ingesting data
}

type PineconeVector struct {
	ID       string            `json:"id"`
	Values   []float32         `json:"values"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

func (p *Pinecone) UploadEmbeddings(embeddings [][]float32, chunks []Chunk) error {
	// Prepare URL
	url := p.APIEndpoint + "/vectors/upsert"

	// Prepare the vectors
	vectors := make([]PineconeVector, len(embeddings))
	for i, embedding := range embeddings {
		chunk := chunks[i]
		vectors[i] = PineconeVector{
			ID:     fmt.Sprintf("id-%d", i),
			Values: embedding,
			Metadata: map[string]string{
				"file_name": chunk.Title,
				"start":     strconv.Itoa(chunk.Start),
				"end":       strconv.Itoa(chunk.End),
				"title":     chunk.Title,
				"text":      chunk.Text,
			},
		}
	}

	maxVectorsPerRequest := 100

	// Split vectors into smaller chunks and make multiple upsert requests
	for i := 0; i < len(vectors); i += maxVectorsPerRequest {
		end := i + maxVectorsPerRequest
		if end > len(vectors) {
			end = len(vectors)
		}

		requestBody, err := json.Marshal(struct {
			Vectors   []PineconeVector `json:"vectors"`
			Namespace string           `json:"namespace"`
		}{
			Vectors:   vectors[i:end],
			Namespace: p.UUID,
		})
		if err != nil {
			return err
		}
		log.Printf("[pinecone] created upsert with ns (%d -> %d) ns=%v", i, end, p.UUID)
		// Create HTTP request
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestBody))
		if err != nil {
			return err
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Api-Key", p.APIKey)

		// Send HTTP request
		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := ioutil.ReadAll(resp.Body)
			return errors.New(string(body))
		}
	}

	return nil
}

type PineconeQueryRequest struct {
	TopK            int                 `json:"topK"`
	IncludeMetadata bool                `json:"includeMetadata"`
	Namespace       string              `json:"namespace"`
	Queries         []PineconeQueryItem `json:"queries"`
}

type PineconeQueryItem struct {
	Values []float32 `json:"values"`
}

type QueryMatch struct {
	ID       string            `json:"id"`
	Score    float32           `json:"score"` // Use "score" instead of "distance"
	Metadata map[string]string `json:"metadata"`
}

type PineconeQueryResponseResult struct {
	Matches []QueryMatch `json:"matches"`
}

type PineconeQueryResponse struct {
	Results []PineconeQueryResponseResult `json:"results"`
}

func (p *Pinecone) Retrieve(questionEmbedding []float32, topK int, uuid string) ([]QueryMatch, error) {
	// Prepare the Pinecone query request
	requestBody, _ := json.Marshal(PineconeQueryRequest{
		TopK:            topK,
		IncludeMetadata: true,
		Namespace:       uuid,
		Queries: []PineconeQueryItem{
			{
				Values: questionEmbedding,
			},
		},
	})

	// log.Println("[retrieve] Querying pinecone namespace:", uuid)
	// Send the Pinecone query request
	pineconeIndexURL := p.APIEndpoint + "/query"
	req, _ := http.NewRequest("POST", pineconeIndexURL, bytes.NewBuffer(requestBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Api-Key", p.APIKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Parse the Pinecone query response
	body, _ := ioutil.ReadAll(resp.Body)
	var pineconeQueryResponse PineconeQueryResponse
	json.Unmarshal(body, &pineconeQueryResponse)

	// Check if there are any results and return the matches
	if len(pineconeQueryResponse.Results) > 0 {
		return pineconeQueryResponse.Results[0].Matches, nil
	}

	return nil, nil
}
