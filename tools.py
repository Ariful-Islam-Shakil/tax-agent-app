# tools.py

from typing import Any
from crewai.tools import BaseTool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class DocumentSearchTool(BaseTool):
    name: str = "Document Search Tool"
    description: str = (
        "Searches through tax and income tax related documents to find relevant information. "
        "Use this tool to answer questions about income tax, taxation laws, tax rules, "
        "and related topics based on the indexed documents. "
        "Input should be a clear question or search query."
    )

    embeddings: Any = None
    vectorstore: Any = None

    def __init__(self):
        super().__init__()

        # ✅ Explicit local embedding model (NO OpenAI, NO Gemini)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ✅ Updated Chroma import
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    def _run(self, query: str) -> str:
        """Search for relevant documents based on the query."""
        try:
            results = self.vectorstore.similarity_search(query, k=5)

            if not results:
                return "No relevant information found in the documents."

            context = "\n\n---\n\n".join(
                [
                    f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                    for doc in results
                ]
            )

            return context

        except Exception as e:
            return f"Error searching documents: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async wrapper (required by CrewAI best practice)."""
        return self._run(query)
