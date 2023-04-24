package botMaker

import (
	"fmt"
	"github.com/caarlos0/env/v8"
)

type Config struct {
	OpenAPIKey string `env:"OPEN_API_KEY"`

	PineconeKey      string `env:"PINECONE_KEY"`
	PineconeEndpoint string `env:"PINECONE_URL"`
}

func NewConfig() *Config {
	return &Config{}
}

func NewConfigFromEnv() *Config {
	cfg := Config{}
	if err := env.Parse(&cfg); err != nil {
		fmt.Printf("%+v\n", err)
	}

	return &cfg
}
