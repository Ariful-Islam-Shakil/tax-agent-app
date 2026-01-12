import os
from crewai import Agent, Task, Crew, Process, LLM
from tools import DocumentSearchTool
from dotenv import load_dotenv
from litellm import RateLimitError

load_dotenv()


class TaxAgentCrew:
    """
    Tax Agent Crew for answering questions based on tax documents.
    """

    def __init__(self):
        # --- LLM initialization (Groq primary, Gemini fallback) ---
        self.llm = self._init_llm()

        # --- Tools ---
        self.search_tool = DocumentSearchTool()

    def _init_llm(self):
        """
        Initialize LLM with Groq (LiteLLM-compatible).
        """
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("❌ GROQ_API_KEY not set")

        return LLM(
            model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,
        )

    def create_agents(self):
        """
        Create agents for the crew.
        """

        # � Router: Checks relevance and rewrites query
        router = Agent(
            role="Query Router and Rewriter",
            goal="Classify if a query is tax-related and rewrite it for optimal document retrieval.",
            backstory=(
                "You are an expert at analyzing user intent. Your primary job is to determine if a "
                "question is about taxation, income tax, or tax laws. If it is, you rewrite it "
                "into a focused search query for a vector database. If it is not, you clearly "
                "mark it as irrelevant."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # 🔍 Researcher: TOOL-ONLY agent
        researcher = Agent(
            role="Tax Document Researcher",
            goal="Search and retrieve relevant information from tax documents",
            backstory=(
                "You are an expert researcher specializing in tax documentation. You retrieve "
                "exact and relevant sections from official tax documents without adding interpretations."
            ),
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # 🧠 Advisor: LLM-powered agent
        advisor = Agent(
            role="Tax Advisor",
            goal="Provide clear, accurate answers based on researched tax documents",
            backstory=(
                "You are a knowledgeable tax advisor with expertise in the income tax system. "
                "You explain tax matters clearly using official documents and cite sources when possible."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        return router, researcher, advisor

    def create_tasks(self, query: str, router: Agent, researcher: Agent, advisor: Agent):
        """
        Create tasks for answering the query.
        """

        triage_task = Task(
            description=(
                f"Analyze this query: '{query}'\n"
                "1. Determine if it is related to tax, income tax, or taxation laws.\n"
                "2. If it is NOT tax-related, the output MUST be exactly: 'IRRELEVANT: [A brief, polite explanation why]'.\n"
                "3. If it IS tax-related, rewrite the query to be optimized for a vector database search "
                "(focused on key terms like 'Section', 'Section 16', 'Tax Rate', etc.). "
                "The output MUST be exactly the rewritten query and nothing else."
            ),
            agent=router,
            expected_output="Either 'IRRELEVANT: [explanation]' or a rewritten search query."
        )

        research_task = Task(
            description=(
                "Search the tax document database for information relevant to the provided query. "
                "Retrieve the most relevant sections with source references."
            ),
            agent=researcher,
            context=[triage_task],
            expected_output=(
                "Relevant excerpts from tax documents with source references."
            )
        )

        advisory_task = Task(
            description=(
                f"Using the researched information, answer the original user question: '{query}'. "
                "Explain clearly and accurately. If information is missing, state it."
            ),
            agent=advisor,
            context=[research_task],
            expected_output=(
                "A clear, well-structured answer based strictly on tax documents."
            )
        )

        return triage_task, research_task, advisory_task

    def answer_query(self, query: str):
        """
        Run the crew to answer the query with triage and rewriting.
        """

        print("\n" + "=" * 80)
        print(f"Processing query: {query}")
        print("=" * 80 + "\n")

        router, researcher, advisor = self.create_agents()
        triage_task, research_task, advisory_task = self.create_tasks(query, router, researcher, advisor)

        # Step 1: Run triage task to check relevance and rewrite
        triage_crew = Crew(
            agents=[router],
            tasks=[triage_task],
            verbose=True
        )
        
        triage_result = str(triage_crew.kickoff())
        
        if triage_result.startswith("IRRELEVANT:"):
            return triage_result.replace("IRRELEVANT:", "").strip()

        # Step 2: If relevant, run research and advisory tasks with the rewritten query
        # The research_task will use triage_result as it's in context
        rag_crew = Crew(
            agents=[researcher, advisor],
            tasks=[research_task, advisory_task],
            process=Process.sequential,
            verbose=True
        )

        try:
            return rag_crew.kickoff()

        except RateLimitError:
            return (
                "⚠️ LLM quota exceeded. Please try again later or switch LLM provider."
            )

        except Exception as e:
            return f"❌ Error while processing query: {str(e)}"


def main():
    """
    Main CLI loop.
    """

    print("=" * 80)
    print("TAX AGENT - Document-based Q&A System")
    print("=" * 80)
    print("\nThis agent answers questions based on tax documents.")
    print("Type 'exit' or 'quit' to end the session.\n")

    tax_crew = TaxAgentCrew()

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            print("\nThank you for using Tax Agent. Goodbye!")
            break

        if not query:
            print("⚠️ Please enter a valid question.")
            continue

        answer = tax_crew.answer_query(query)

        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(answer)
        print("=" * 80)


if __name__ == "__main__":
    main()
