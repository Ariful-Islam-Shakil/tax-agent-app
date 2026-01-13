# Tax Agent - Document-based Q&A System

A CrewAI-powered agent system that answers questions based on tax and income tax documents using local embeddings and efficient LLMs.

## Features

- **Document Processing**: Automatically loads and indexes all `.txt` and `.md` files from the specified documents folder.
- **Local Embeddings**: Uses HuggingFace (`all-MiniLM-L6-v2`) for local vector generation, eliminating quota issues for embeddings.
- **Query Triage & Rewriting**: Automatically classifies queries to ensure they are tax-related and rewrites them for optimized search retrieval.
- **Vector Search**: Uses Weaviate Cloud for enterprise-grade vector search and document retrieval.
- **Multi-Agent System**: 
  - **Router Agent**: Analyzes intent, filters out non-tax queries, and rewrites valid queries for better search.
  - **Researcher Agent**: Searches and retrieves relevant information from documents using rewritten queries.
  - **Advisor Agent**: Synthesizes researched information into clear, accurate answers.
- **Groq Integration**: Powered by Groq (Llama 3.1) for lightning-fast, high-quality responses.
- **Interactive CLI**: Easy-to-use command-line interface for asking questions.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

1. **Groq API Key**: Get it from [Groq Console](https://console.groq.com/keys). This is used for the LLM agents.
2. **Google API Key (Optional)**: Used for fallback or supplementary tasks. Get it from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 3. Configure Environment

Edit the `.env` file and add your keys:

```text
DOCUMENTS_PATH=/path/to/your/tax_documents
GROQ_API_KEY=your_groq_api_key
WEAVIATE_URL=your_weaviate_cluster_url
WEAVIATE_API_KEY=your_weaviate_api_key
```

### 4. Index Documents

Run the document processor to create the local vector database:

```bash
python document_processor.py
```

This will:
- Load all `.txt` and `.md` files from the documents folder.
- Split them into chunks.
- Create embeddings **locally** using the HuggingFace `all-MiniLM-L6-v2` model.
- Store them in a Weaviate Cloud index (`TaxDocument`).

**Note**: You only need to run this once, unless you add new documents.

### 5. Run the Agent

```bash
python main.py
```

## Usage

Once the agent is running, you can ask questions about the tax documents:

```text
Your question: What is the income tax rate for individuals?

Your question: What are the tax exemptions available?

Your question: How is capital gains tax calculated?
```

Type `exit`, `quit`, or `q` to end the session.

## How It Works

1. **Document Processing**: 
   - All text files are loaded and split into manageable chunks.
   - Each chunk is converted to embeddings using a **local HuggingFace model**, meaning no API costs or rate limits for indexing.
   - Embeddings are stored in Weaviate Cloud for fast, scalable retrieval.

2. **Query Processing**:
   - User asks a question.
   - **Router Agent**: Analyzes if the query is related to tax law. If not, it provides a polite rejection. If yes, it rewrites the query into optimized keywords for a vector search.
   - **Researcher Agent**: Uses the **rewritten query** to scan the Weaviate index for relevant excerpts (e.g., specific Sections or Rules).
   - **Advisor Agent**: Takes the raw excerpts and the original question to synthesize a final, human-readable answer.

3. **Multi-Agent Collaboration**:
   - The **Router** acts as a gatekeeper and optimizer.
   - The **Researcher** acts as a document excavator.
   - The **Advisor** acts as the final legal interpreter.

## Documents Indexed

The system indexes all `.txt` and `.md` files in the configured documents path, including:
- Income Tax Acts and Ordinances
- Tax Rules and Regulations
- Taxation Handbooks
- Tax Analysis Documents

## Technologies Used

- **CrewAI**: Multi-agent orchestration framework.
- **LangChain & langchain-weaviate**: Document processing and vector store integration.
- **Groq**: High-performance LLM hosting (Llama 3.1).
- **HuggingFace**: Local embedding models (`all-MiniLM-L6-v2`).
- **Weaviate**: Managed vector database.

## Troubleshooting

**Error: "ImportError: cannot import name 'BaseTool' from 'crewai_tools'"**
- This is fixed in the current code (imported from `crewai.tools` instead).

**Error: "ValueError: ... object has no field 'embeddings'"**
- This is fixed in the current code by defining fields explicitly for Pydantic.

**Error: "GROQ_API_KEY not set"**
- Ensure you have added your Groq key to the `.env` file.

**Error: "No relevant information found"**
- Make sure you've run `document_processor.py` to index the documents first.
- Check that your Weaviate cluster is accessible and the index contains data.