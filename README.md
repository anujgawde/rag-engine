
# PDF RAG API

A powerful Retrieval-Augmented Generation (RAG) system for PDF documents using LlamaParse, LangChain, and Groq. This API allows you to upload PDF files, process them into a searchable knowledge base, and query them using natural language.

## Features

- 📄 **Multi-PDF Processing**: Upload and process multiple PDF files simultaneously
- 🔍 **Intelligent Document Parsing**: Uses LlamaParse for high-quality PDF text extraction
- 🧠 **Smart Chunking**: Automatically splits documents into optimal chunks for retrieval
- 🔎 **Semantic Search**: Vector-based similarity search using HuggingFace embeddings
- 💬 **Natural Language Queries**: Ask questions in plain English and get relevant answers
- 📚 **Source Attribution**: Get references to the source documents for each answer
- 💾 **Persistent Storage**: Vector database persists between sessions using ChromaDB
- ⚡ **Fast Inference**: Powered by Groq for quick response times

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│   LlamaParse     │───▶│  Text Chunking  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Queries   │◀───│   Groq LLM       │◀───│ Vector Database │
└─────────────────┘    └──────────────────┘    │   (ChromaDB)    │
                                                └─────────────────┘
                                                         ▲
                                                         │
                                                ┌─────────────────┐
                                                │ HuggingFace     │
                                                │ Embeddings      │
                                                └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- LlamaCloud API Key (from [LlamaIndex](https://cloud.llamaindex.ai/))
- Groq API Key (from [Groq](https://console.groq.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anujgawde/rag-engine.git
   cd rag-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
   GROQ_API_KEY=your_groq_api_key
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   PERSIST_DIR=./chroma_db
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## API Documentation

### Upload PDFs

Upload and process PDF files into the knowledge base.

**Endpoint:** `POST /upload-pdfs`

**Request:**
```bash
curl -X POST "http://localhost:8000/upload-pdfs" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully ingested 2 PDF files",
  "files_processed": 2,
  "file_names": ["document1.pdf", "document2.pdf"]
}
```

### Query Documents

Ask questions about your uploaded documents.

**Endpoint:** `POST /query`

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=What are the main topics discussed in the documents?"
```

**Response:**
```json
{
  "status": "success",
  "message": "Query processed successfully",
  "query": "What are the main topics discussed in the documents?",
  "answer": "Based on the documents, the main topics include...",
  "sources": [
    {
      "content": "Excerpt from the relevant document...",
      "metadata": {
        "source": "document1.pdf",
        "page": 1,
        "file_name": "document1.pdf"
      }
    }
  ]
}
```

## Interactive API Documentation

Once the server is running, you can access the interactive API documentation at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_CLOUD_API_KEY` | API key for LlamaParse service | Required |
| `GROQ_API_KEY` | API key for Groq LLM service | Required |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `PERSIST_DIR` | Directory to store vector database | `./chroma_db` |

### Customization

You can customize various parameters in the `PDFRAG` class:

- **Chunk Size**: Modify `chunk_size` in `RecursiveCharacterTextSplitter`
- **Chunk Overlap**: Adjust `chunk_overlap` for better context preservation
- **Retrieval Count**: Change `k` parameter in retriever setup
- **LLM Model**: Switch between different Groq models
- **Temperature**: Adjust creativity vs consistency in responses

## Examples

### Example 1: Processing Research Papers

```python
# Upload research papers
files = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
# Query: "What are the main findings across these research papers?"
```

### Example 2: Company Documentation

```python
# Upload company manuals, policies, and procedures
files = ["employee_handbook.pdf", "safety_procedures.pdf", "it_policies.pdf"]
# Query: "What is the company's remote work policy?"
```

### Example 3: Legal Documents

```python
# Upload contracts and legal documents
files = ["contract1.pdf", "terms_of_service.pdf", "privacy_policy.pdf"]
# Query: "What are the termination clauses in these contracts?"
```

## Troubleshooting

### Common Issues

1. **"No documents have been ingested yet"**
   - Make sure you've uploaded PDF files using the `/upload-pdfs` endpoint
   - Check if the vector database directory exists and contains data

2. **"Error parsing PDF"**
   - Ensure your PDF files are not corrupted
   - Check if your LlamaCloud API key is valid and has sufficient credits

3. **Slow response times**
   - Consider using a smaller embedding model for faster processing
   - Reduce chunk size or retrieval count for quicker responses

4. **Memory issues**
   - For large documents, consider increasing chunk size to reduce total chunks
   - Monitor system memory usage during processing

### API Key Issues

- Ensure your API keys are correctly set in the `.env` file
- Check API key validity and quota limits
- Verify network connectivity to external services

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for excellent PDF parsing capabilities
- [LangChain](https://www.langchain.com/) for the RAG framework
- [Groq](https://groq.com/) for fast LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [HuggingFace](https://huggingface.co/) for embedding models

## Support

If you encounter any issues or have questions, please:
1. Check the [troubleshooting section](#troubleshooting)
2. Search existing [GitHub issues](https://github.com/anujgawde/rag-engine/issues)
3. Create a new issue with detailed information about your problem

---

⭐ If you find this project helpful, please consider giving it a star!
