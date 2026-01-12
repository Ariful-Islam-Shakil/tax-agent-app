import os
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv

load_dotenv()


class DocumentProcessor:
    """Process and index documents from the specified folder."""

    def __init__(self, documents_path: str, index_name: str = "TaxDocument"):
        if not documents_path:
            raise ValueError("DOCUMENTS_PATH is not set")

        self.documents_path = documents_path
        self.index_name = index_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        if not weaviate_url or not weaviate_api_key:
            raise ValueError("WEAVIATE_URL or WEAVIATE_API_KEY is missing")

        auth_config = Auth.api_key(weaviate_api_key)
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=auth_config,
        )

    def load_documents(self):
        """Load all text and markdown files."""
        print(f"Loading documents from {self.documents_path}...")

        documents = []

        loaders = [
            DirectoryLoader(
                self.documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
            ),
            DirectoryLoader(
                self.documents_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
            ),
        ]

        for loader in loaders:
            documents.extend(loader.load())

        if not documents:
            raise ValueError("No documents found to index")

        print(f"Loaded {len(documents)} documents")
        return documents

    def split_documents(self, documents):
        """Split documents into chunks."""
        print("Splitting documents into chunks...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Document splitting produced no chunks")

        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vector_store(self, chunks: List):
        """Create and index documents in Weaviate."""
        print(f"Indexing {len(chunks)} chunks into Weaviate ({self.index_name})...")

        vectorstore = WeaviateVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=self.client,
            index_name=self.index_name,
            text_key="text",
        )

        print("Indexing completed")
        return vectorstore

    def load_vector_store(self):
        """Load existing vector store."""
        print(f"Loading existing Weaviate index: {self.index_name}")

        return WeaviateVectorStore(
            client=self.client,
            index_name=self.index_name,
            embedding=self.embeddings,
            text_key="text",
        )

    def process_and_index(self):
        """Main pipeline."""
        try:
            if self.client.collections.exists(self.index_name):
                print("Index already exists. Using existing index.")
                return self.load_vector_store()

            documents = self.load_documents()
            chunks = self.split_documents(documents)
            return self.create_vector_store(chunks)

        except Exception as e:
            print(f"Error during processing: {e}")
            raise

    def close(self):
        """Close Weaviate client."""
        self.client.close()


if __name__ == "__main__":
    processor = DocumentProcessor(os.getenv("DOCUMENTS_PATH"))

    try:
        vectorstore = processor.process_and_index()

        query = "What is income tax?"
        results = vectorstore.similarity_search(query, k=3)

        print(f"\nQuery: {query}")
        for i, doc in enumerate(results, 1):
            print(f"\nChunk {i}:\n{doc.page_content[:200]}...")

    finally:
        processor.close()
