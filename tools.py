# tools.py

from typing import Any, List
from crewai.tools import BaseTool
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import weaviate
from weaviate.classes.init import Auth
import os
from dotenv import load_dotenv

load_dotenv()


class DocumentSearchTool(BaseTool):
    name: str = "Document Search Tool"
    description: str = (
        "Searches tax and income tax documents to retrieve relevant sections. "
        "Use this tool ONLY for tax-related questions."
    )

    embeddings: Any = None
    vectorstore: Any = None
    client: Any = None
    index_name: str = "TaxDocument"

    def __init__(self):
        super().__init__()

        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        if not weaviate_url or not weaviate_api_key:
            raise RuntimeError("❌ WEAVIATE_URL or WEAVIATE_API_KEY missing")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        auth_config = Auth.api_key(weaviate_api_key)
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=auth_config,
        )

        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name=self.index_name,
            embedding=self.embeddings,
            text_key="text",
        )

    def _format_results(self, docs: List) -> str:
        """Reduce token usage and structure context."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()[:800]  # limit length
            formatted.append(
                f"[Document {i}]\nSource: {source}\nContent:\n{content}"
            )
        return "\n\n---\n\n".join(formatted)

    def _run(self, query: str) -> str:
        try:
            docs = self.vectorstore.similarity_search(query, k=3)

            if not docs:
                return "NO_RESULTS"

            return self._format_results(docs)

        except Exception as e:
            return f"ERROR: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)

    def close(self):
        """Close Weaviate connection safely."""
        if self.client:
            self.client.close()
