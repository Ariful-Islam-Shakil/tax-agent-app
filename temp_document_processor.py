import os
import shutil
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


class TempDocumentProcessor:
    """Process and index documents from the specified folder using ChromaDB."""

    def __init__(self, documents_path: str, index_name: str = "TaxDocument"):
        if not documents_path:
            raise ValueError("DOCUMENTS_PATH is not set")

        self.documents_path = documents_path
        self.index_name = index_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ChromaDB persistence directory
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

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
        """Create and index documents in ChromaDB."""
        print(f"Indexing {len(chunks)} chunks into ChromaDB ({self.index_name})...")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.index_name,
            persist_directory=self.persist_directory,
        )

        print("Indexing completed")
        return vectorstore

    def load_vector_store(self):
        """Load existing vector store."""
        print(f"Loading existing ChromaDB index: {self.index_name}")

        return Chroma(
            collection_name=self.index_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def process_and_index(self):
        """Main pipeline."""
        try:
            # Check if collection exists by trying to load it
            try:
                vectorstore = self.load_vector_store()
                if vectorstore._collection.count() > 0:
                    print("Index already exists. Using existing index.")
                    return vectorstore
            except Exception:
                pass

            documents = self.load_documents()
            chunks = self.split_documents(documents)
            return self.create_vector_store(chunks)

        except Exception as e:
            print(f"Error during processing: {e}")
            raise

    def delete_vector_store(self):
        """Delete the existing vector store collection and persistence directory."""
        print(f"Deleting vector store index: {self.index_name}...")
        try:
            # Initialize we can use the load_vector_store to get the object
            vectorstore = self.load_vector_store()
            vectorstore.delete_collection()
            
            # Also remove the persistence directory to clean up files
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                print(f"Removed directory: {self.persist_directory}")
                
            print(f"Successfully deleted vector store: {self.index_name}")
        except Exception as e:
            print(f"Error deleting vector store: {e}")

    def close(self):
        """Close ChromaDB client (cleanup if needed)."""
        pass


if __name__ == "__main__":
    processor = TempDocumentProcessor(os.getenv("TEMP_DOCUMENTS_PATH"))

    try:
        vectorstore = processor.process_and_index()

        query = "What is income tax?"
        results = vectorstore.similarity_search(query, k=3)

        print(f"\nQuery: {query}")
        for i, doc in enumerate(results, 1):
            print(f"\nChunk {i}:\n{doc.page_content[:200]}...")
            print("######\n")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print("######\n")


        # print("\n#######\nDeleting vector store...")
        # processor.delete_vector_store()
        # print("\n#######\nVector store deleted successfully.")


    finally:
        processor.close()