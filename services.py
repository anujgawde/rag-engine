import os
from typing import List
from pathlib import Path

# Core libraries
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from groq import Groq

# Load Environment variable
from dotenv import load_dotenv
load_dotenv()

class PDFRAG:
    def __init__(self,
                 llama_cloud_api_key: str,
                 groq_api_key: str,
                 embedding_model: str = os.environ.get("EMBEDDING_MODEL")):
        """
        Initializing RAG System with necessary config

        Constructor Args:
            llama_cloud_api_key: LlamaParse api key
            groq_api_key: Groq api key
            embedding_model: Hugging Face embedding model name
        """
        self.llama_cloud_api_key = llama_cloud_api_key
        self.groq_api_key = groq_api_key

        # LlamaParse parser
        self.parser = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="markdown",  # "markdown" or "text"
            verbose=True,
            language="en"
        )

        # HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # Groq
        self.groq_client = Groq(api_key=groq_api_key)

        # Text splitter (for chunking)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # To be initialized while ingesting.
        # Vector store to store embeddings
        self.vectorstore = None

    async def parse_pdf(self, pdf_path: str) -> List[Document]:
        """
        Parsing a PDF using LlamaParse.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of LangChain Document objects
        """
        print(f"Parsing PDF: {pdf_path}")

        # Parsing the PDF using llamaparse parser
        documents = await self.parser.aload_data(pdf_path)

        # Converting to LangChain Documents
        langchain_docs = []
        for doc in documents:
            langchain_doc = Document(
                page_content=doc.text,
                metadata={
                    "source": pdf_path,
                    "page": getattr(doc, 'page_number', None),
                    "file_name": Path(pdf_path).name
                }
            )
            langchain_docs.append(langchain_doc)

        print(f"Successfully parsed {len(langchain_docs)} pages")
        return langchain_docs

    async def parse_multiple_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        Parse multiple PDFs using LlamaParse.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            List of LangChain Document objects from all PDFs
        """
        print(f"Parsing {len(pdf_paths)} PDFs...")
        all_documents = []

        for pdf_path in pdf_paths:
            try:
                documents = await self.parse_pdf(pdf_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error parsing {pdf_path}: {str(e)}")
                continue

        print(f"Successfully parsed {len(all_documents)} total pages from {len(pdf_paths)} PDFs")
        return all_documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        print("Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document], persist_directory: str = "./chroma_db"):
        """
        Create vector store from document chunks.

        Args:
            chunks: List of document chunks
            persist_directory: Directory to persist the vector database
        """
        print("Creating vector store...")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            # Todo: Add a collection name to store files. Currently a random hash is used for naming collections.
            # collection_name="knowledge_base"
        )

        # Saving to disk
        self.vectorstore.persist()
        print(f"Vector store created with {len(chunks)} chunks")

    def load_vectorstore(self, persist_directory: str = "./chroma_db"):
        """
        Load existing vector store from disk.

        Args:
            persist_directory: Directory where vector database is stored
        """
        print("Loading existing vector store...")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        print("Vector store loaded successfully")

    def setup_retriever(self, model_name: str = "llama3-8b-8192", temperature: float = 0.1):
        """
        Set up the question-answering chain using Groq.

        Args:
            model_name: Groq model name
            temperature: Temperature for text generation
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please create or load a vector store first.")

        print("Setting up retriever chain...")

        # Custom LLM wrapper for Groq
        class GroqLLM:
            def __init__(self, client, model_name, temperature):
                self.client = client
                self.model_name = model_name
                self.temperature = temperature

            def __call__(self, prompt: str) -> str:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=1024
                )
                return response.choices[0].message.content

        # Initialize Groq LLM
        groq_llm = GroqLLM(self.groq_client, model_name, temperature)

        # Creating retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Top 4 most similar chunks
        )

        # Simple retriever function
        self.retriever = retriever
        self.llm = groq_llm

        print("Retriever chain setup complete")

    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.

        Args:
            question: User question

        Returns:
            Dictionary containing answer and source documents
        """
        if self.retriever is None or self.llm is None:
            raise ValueError("Retriever chain not set up. Please run setup_retriever() first.")

        print(f"Processing query: {question}")

        # Relevant documents to query
        relevant_docs = self.retriever.get_relevant_documents(question)

        # Context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # LLM Prompt
        prompt = f"""
        Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        # Answer from LLM
        answer = self.llm(prompt)

        # Sources of truth for the answer retrieved by RAG system
        sources = []
        for doc in relevant_docs:
            source_info = {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            sources.append(source_info)

        return {
            "answer": answer,
            "sources": sources,
            "query": question
        }

    async def process_pdfs_and_setup(self, pdf_paths: List[str], persist_directory: str = "./chroma_db"):
        """
        Complete pipeline to process multiple PDFs RAG system setup.

        Args:
            pdf_paths: List of paths to PDF files
            persist_directory: Directory to persist vector database
        """
        # Parsing Multiple PDFs
        documents = await self.parse_multiple_pdfs(pdf_paths)

        # Document chunking
        chunks = self.chunk_documents(documents)

        # Vector store
        self.create_vectorstore(chunks, persist_directory)

        # Retriever chain
        self.setup_retriever()

        print("RAG system ready for queries!")


# Service class to be exposed for FastAPI
class RAGService:
    def __init__(self):
        """Initialize the RAG service with environment variables."""

        # API Keys
        self.llama_cloud_api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        self.groq_api_key = os.environ.get("GROQ_API_KEY")

        if not self.llama_cloud_api_key or not self.groq_api_key:
            raise ValueError("Please set LLAMA_CLOUD_API_KEY and GROQ_API_KEY environment variables")

        # Initializing RAG System
        self.rag_system = PDFRAG(
            llama_cloud_api_key=self.llama_cloud_api_key,
            groq_api_key=self.groq_api_key
        )
        self.persist_dir = os.environ.get("PERSIST_DIR")

        # Loading existing vectorstore on initialization
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initialize vectorstore if it exists."""
        if os.path.exists(self.persist_dir):
            try:
                self.rag_system.load_vectorstore(self.persist_dir)
                self.rag_system.setup_retriever()
                print("Loaded existing vectorstore on service initialization")
            except Exception as e:
                print(f"Could not load existing vectorstore: {str(e)}")

    async def ingest_pdfs(self, pdf_paths: List[str]) -> dict:
        """
        Ingest multiple PDF files into the RAG system.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            Dictionary with ingestion status
        """
        try:
            await self.rag_system.process_pdfs_and_setup(pdf_paths, self.persist_dir)
            return {
                "status": "success",
                "message": f"Successfully ingested {len(pdf_paths)} PDF files",
                "files_processed": len(pdf_paths)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error ingesting PDFs: {str(e)}",
                "files_processed": 0
            }

    def ask_question(self, question: str) -> dict:
        """
        Ask a question to the RAG system.

        Args:
            question: User question

        Returns:
            Dictionary containing answer and source information
        """
        try:
            # Checking System status.
            if self.rag_system.retriever is None or self.rag_system.llm is None:
                # Try to initialize if vectorstore exists
                if os.path.exists(self.persist_dir):
                    self.rag_system.load_vectorstore(self.persist_dir)
                    self.rag_system.setup_retriever()
                else:
                    return {
                        "status": "error",
                        "message": "No documents have been ingested yet. Please upload and ingest PDFs first.",
                        "answer": None,
                        "sources": []
                    }
            # Leverages query from PDFRAG class to answer the question.
            result = self.rag_system.query(question)
            return {
                "status": "success",
                "message": "Query processed successfully",
                "answer": result["answer"],
                "sources": result["sources"],
                "query": result["query"]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}",
                "answer": None,
                "sources": []
            }