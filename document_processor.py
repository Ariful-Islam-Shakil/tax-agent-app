import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


class DocumentProcessor:
    """Process and index documents from the specified folder."""
    
    def __init__(self, documents_path: str, persist_directory: str = "./chroma_db"):
        self.documents_path = documents_path
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def load_documents(self):
        """Load all text and markdown files from the documents folder."""
        print(f"Loading documents from {self.documents_path}...")
        
        documents = []
        
        # Load .txt files
        txt_loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'},
            show_progress=True
        )
        documents.extend(txt_loader.load())
        
        # Load .md files
        md_loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'},
            show_progress=True
        )
        documents.extend(md_loader.load())
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents):
        """Split documents into chunks for better retrieval."""
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks: List, batch_size: int = 50):
        """Create and persist vector store with batching to avoid rate limits."""
        print(f"Creating vector store for {len(chunks)} chunks...")
        
        # Initialize vector store with the first batch
        vectorstore = Chroma.from_documents(
            documents=chunks[:batch_size],
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Add remaining chunks in batches
        for i in range(batch_size, len(chunks), batch_size):
            import time
            batch = chunks[i:i + batch_size]
            print(f"Adding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
            vectorstore.add_documents(batch)
            # Small delay to respect rate limits
            time.sleep(1)
            
        print(f"Vector store created and persisted to {self.persist_directory}")
        return vectorstore
    
    def load_vector_store(self):
        """Load existing vector store."""
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return vectorstore
    
    def process_and_index(self):
        """Main method to process and index all documents."""
        # Check if vector store already exists
        if os.path.exists(self.persist_directory):
            print("Vector store already exists. Loading...")
            return self.load_vector_store()
        
        # Load and process documents
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        vectorstore = self.create_vector_store(chunks)
        
        return vectorstore


if __name__ == "__main__":
    # Test the document processor
    docs_path = os.getenv("DOCUMENTS_PATH")
    processor = DocumentProcessor(docs_path)
    vectorstore = processor.process_and_index()
    
    # Test retrieval
    query = "What is income tax?"
    results = vectorstore.similarity_search(query, k=3)
    print(f"\nTest query: {query}")
    print(f"Found {len(results)} relevant chunks")
    for i, doc in enumerate(results):
        print(f"\nChunk {i+1}:")
        print(doc.page_content[:200] + "...")
