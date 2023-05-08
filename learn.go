package botMaker

import (
	"bufio"
	"code.sajari.com/docconv"
	"fmt"
	"github.com/jdkato/prose/v2"
	"github.com/sashabaranov/go-openai"
	"log"
	"os"
	"path/filepath"
	"strings"
)

type Chunk struct {
	Start int
	End   int
	Title string
	Text  string
}

type Learn struct {
	Model           string
	TokenLimit      int
	ChunkSize       int
	Overlap         int
	Memory          Storage
	Client          *OAIClient
	GetTitle        TitleGetter
	PreProcessBody  PreProcessor
	PreProcessChunk PreProcessor
}

// ExtensionSupported checks if the extension for a given file path is supported by the library, it returns the
// file extension and a bool whether it is supported or not, supported file types are txt, pdf, and md.
func (l *Learn) ExtensionSupported(path string) (string, bool) {
	ext := filepath.Ext(path)
	supported := false

	switch ext {
	case ".txt", ".pdf", ".md", "json", ".yml":
		supported = true
		// add new extensions here
	}

	return ext, supported
}

type PreProcessor func(string) (string, error)

type TitleGetter func(string) (string, error)

func PathTitleGetter(path string) (string, error) {
	file, err := os.Stat(path)
	if err != nil {
		return "", err
	}

	return file.Name(), nil
}

// ProcessTextFile opens and fully reads the file in 'path', it treats the first line that contains text as a title,
// and reads the remaining file into another variable, it then returns the title and the file contents. It returns an
// error if there is a problem opening or reading the file.
func (l *Learn) ProcessTextFile(path string) (string, string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", "", err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var contents string
	for scanner.Scan() {
		contents += scanner.Text() + "\n"
	}

	if err := scanner.Err(); err != nil {
		return "", "", err
	}

	if l.GetTitle == nil {
		l.GetTitle = PathTitleGetter
	}

	title, err := l.GetTitle(path)
	if err != nil {
		return "", "", err
	}

	if title == "" {
		title = file.Name()
	}

	return title, contents, nil
}

// ProcessPDFFile reads a PDF file from the path and extracts the human-readable text, it will also attempt to extract
// the title of the PDF. It returns the title, the human-readable content, and an optional error if there is a problem
// reading or parsing the file.
func (l *Learn) ProcessPDFFile(path string) (string, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", "", err
	}

	// Convert the uploaded file to a human-readable text
	bodyResult, _, err := docconv.ConvertPDF(f)
	if err != nil {
		return "", "", err
	}

	// Remove extra whitespace and newlines
	text := strings.TrimSpace(bodyResult)

	// TODO: Make the title getter work against file extensions

	return f.Name(), text, nil
}

func (l *Learn) Learn(contents, title string, sentences bool) (int, error) {
	var chunks []Chunk

	if l.PreProcessBody != nil {
		preProcessed, err := l.PreProcessBody(contents)
		if err != nil {
			return 0, err
		}

		contents = preProcessed
	}

	if sentences {
		chunks = l.CreateChunks(contents, title)
	} else {
		chunks = l.CreateChunksCharacterBased(contents, title)
	}

	embeddings, err := l.Client.getEmbeddingsForData(chunks, 100, openai.AdaEmbeddingV2)
	if err != nil {
		return 0, fmt.Errorf("error getting embeddings: %v", err)
	}

	log.Printf("[learn] title: %s", title)
	log.Printf("[learn] total chunks: %d", len(chunks))
	log.Printf("[learn] total embeddings: %d", len(embeddings))
	if len(embeddings) == 0 {
		log.Println("no embeddings in this data, skipping")
		return 0, nil
	}

	// Send the embeddings to memory
	err = l.Memory.UploadEmbeddings(embeddings, chunks)
	if err != nil {
		return 0, fmt.Errorf("error upserting embeddings to memory: %v", err)
	}

	return len(embeddings), nil
}

// FromFile processes a file to learn into an OpenAI memory store, returns number of embeddings
// created and an error if failed
func (l *Learn) FromFile(path string, sentences bool) (int, error) {
	ext, supported := l.ExtensionSupported(path)
	if !supported {
		return 0, fmt.Errorf("file format is not supported")
	}

	var contents, title string
	var err error

	switch ext {
	case ".txt", ".md", ".text":
		title, contents, err = l.ProcessTextFile(path)
	case ".pdf":
		title, contents, err = l.ProcessPDFFile(path)
	}

	if err != nil {
		return 0, err
	}

	return l.Learn(contents, title, sentences)
}

func (l *Learn) CreateChunksCharacterBased(fileContent, title string) []Chunk {
	log.Println("starting character-based chunk generator")
	newData := make([]Chunk, 0)

	c := 0
	text := ""
	start := 0
	end := 0
	overlap := l.Overlap

	//overlap := int(float32(l.ChunkSize) * 0.03) // 3% overlap between chunks

	tail := ""
	for si, _ := range fileContent {
		text += string(fileContent[si])
		end = start + len(text)

		if c == l.ChunkSize || (c < l.ChunkSize && si == len(fileContent)-1) {
			//TODO: Should the chunks be a forced size of the max limit?
			if CheckTokenLimit(text, l.Model, l.TokenLimit) {
				// only write chunks that are ok
				newData = append(newData, Chunk{
					Start: start,
					End:   end,
					Title: title,
					Text:  tail + text, // include the overlap
				})

				//fmt.Println("CHUNK CREATED")
				//fmt.Println("=============")
				//fmt.Println(tail + text)
				//fmt.Println("=============")

				if len(text) > overlap {
					if overlap != 0 {
						tail = text[len(text)-(overlap)-1 : len(text)]
					}
				}
			} else {
				log.Println("chunk size too large, skipping chunk")
			}

			text = ""
			c = 0
		}

		c++
		start = end + 1
	}

	return newData
}

// CreateChunks generates uploadable chunks to send to a memory store
func (l *Learn) CreateChunks(fileContent, title string) []Chunk {
	log.Println("starting sentence-based chunk generator")
	doc, err := prose.NewDocument(fileContent)
	if err != nil {
		log.Fatal(err)
	}

	sentences := doc.Sentences()
	newData := make([]Chunk, 0)

	c := 0
	text := ""
	start := 0
	end := 0
	tailTxt := ""
	for si, _ := range sentences {
		//fixedText := strings.ReplaceAll(sentences[si].Text, "  ", " ")
		text += " " + sentences[si].Text
		end = start + len(text)

		if c == l.ChunkSize || (c < l.ChunkSize && si == len(sentences)-1) {
			toWrite := tailTxt + text
			// preprocess the chunk
			if l.PreProcessChunk != nil {
				preProcessed, err := l.PreProcessChunk(toWrite)
				if err != nil {
					log.Printf("[createChunks] failed to preprocess: %v", err)
					preProcessed = toWrite
				}

				toWrite = preProcessed
			}
			if CheckTokenLimit(toWrite, l.Model, l.TokenLimit) {
				// only write chunks that are ok
				newData = append(newData, Chunk{
					Start: start,
					End:   end,
					Title: title,
					Text:  toWrite,
				})

				if l.Overlap != 0 {
					dist := si + 1 - (si + 1) - l.Overlap
					if dist > len(sentences) {
						tailTxt = ""
						tail := sentences[(si+1)-l.Overlap : si+1]
						for x, _ := range tail {
							tailTxt += tail[x].Text
						}
					}

					//tailTxt = "++++BEGIN++++" + tailTxt + "++++END++++"
				}
			} else {
				log.Println("chunk size too large, skipping chunk")
			}

			text = ""
			c = 0
		}

		c++
		start = end + 1
	}

	return newData
}
