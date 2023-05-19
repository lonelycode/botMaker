package botMaker

import (
	"log"

	"github.com/caarlos0/env/v8"
)

type Config struct {
	LLMAPIKey string `env:"LLM_API_KEY,required"`

	PineconeKey      string `env:"PINECONE_KEY"`
	PineconeEndpoint string `env:"PINECONE_URL"`
}

func NewConfig() *Config {
	return &Config{}
}

func NewConfigFromEnv() *Config {
	cfg := Config{}
	if err := env.Parse(&cfg); err != nil {
		log.Fatalf("%+v\n", err)
	}

	return &cfg
}
