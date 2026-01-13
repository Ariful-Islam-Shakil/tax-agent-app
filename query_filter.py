import os
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TaxQueryFilter:
    """
    CrewAI Agent to filter queries based on whether they are tax-related or not.
    """

    def __init__(self):
        # Initialize the LLM with OpenRouter model
        # if not os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENAI_API_KEY"):
        #     os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENAI_API_KEY")

        # # Set a low max_tokens to avoid credit issues
        # self.llm = LLM(
        #     model="openrouter/meta-llama/llama-3.2-1b-instruct",
        #     api_key=os.getenv("OPENROUTER_API_KEY"),
        #     max_tokens=100,
        #     temperature=0.0 # More deterministic
        # )
        
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("❌ GROQ_API_KEY not set")

        self.llm = LLM(
            model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,
        )

    def verify_query(self, query: str) -> str:
        """
        Takes a query and returns 'yes' if it's tax-related, 'no' otherwise.
        """
        # 1. Define the Agent
        filter_agent = Agent(
            role="Tax Query Classifier",
            goal=(
                "Determine whether a user query is related to taxation. "
                "A query is considered tax-related if it involves income tax, VAT, GST, "
                "corporate tax, tax rates, tax rules, tax laws, filing returns, deductions, "
                "exemptions, tax authorities, or any form of government taxation."
            ),
            backstory=(
                "You are an expert tax-domain classifier. "
                "You carefully analyze the meaning and intent of the user's query. "
                "First, you internally decide whether the topic involves taxes or taxation. "
                "Then, you respond with a strict binary answer.\n\n"
                "IMPORTANT RULES:\n"
                "- If the query is about any type of tax, return 'yes'.\n"
                "- If the query is NOT related to tax in any way, return 'no'.\n"
                "- Do NOT explain your reasoning.\n"
                "- Do NOT add extra words, punctuation, or formatting.\n"
                "- Output MUST be exactly one word: 'yes' or 'no'."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
            cache=False
        )

        # 2. Define the Task
        filter_task = Task(
            description=(
                f"User Query:\n"
                f"\"{query}\"\n\n"
                "Task:\n"
                "Analyze the query carefully and decide whether it is related to taxation.\n\n"
                "Tax-related topics include (but are not limited to):\n"
                "- Income tax, corporate tax, VAT, GST\n"
                "- Tax rates, tax slabs\n"
                "- Filing tax returns\n"
                "- Tax deductions, exemptions, rebates\n"
                "- Tax laws, tax rules, tax authorities\n\n"
                "After analysis, respond with only one word:\n"
                "- 'yes' if the query is tax-related\n"
                "- 'no' if the query is not tax-related\n\n"
                "Do not provide explanations or additional text."
                ),
            agent=filter_agent,
            expected_output="yes or no"
        )


        # 3. Create the Crew
        crew = Crew(
            agents=[filter_agent],
            tasks=[filter_task],
            verbose=False
        )

        # Execute
        result = crew.kickoff()
        
        # Clean up the response
        final_answer = str(result).strip().lower()
        if "yes" in final_answer:
            return "yes"
        else:
            return "no"

def main():
    """
    Main function to test the TaxQueryFilter.
    """
    print("\n" + "="*50)
    print("TAX QUERY FILTER TEST")
    print("="*50)

    tester = TaxQueryFilter()
    
    test_queries = [
        "How do I calculate my income tax?",
        "What are the benefits of section 80C?",
        "Tell me about the weather in London.",
        "Can you help me with corporate tax filing?",
        "What is the best way to cook pasta?",
        "can a person legally skip the consequences of not paying tax even if he must pay tax according to bngladeshi law?",
        "how to make tea",
        "how to bake pasta. tax."
    ]
    test_queries = [
        "Do freelancers need to report their earnings to the government every year?",
        "What percentage of salary is legally deducted by the government in Bangladesh?",
        "Is money earned from YouTube or online content subject to government charges?",
        "How does the government calculate deductions before paying employees?",
        "Are businesses required to submit financial reports to revenue authorities?",
        "How does a company calculate employee salary deductions for provident funds?",
        "What fees does a bank charge for international money transfers?",
        "Why do online platforms take a percentage of sellers’ earnings?",
        "What is the cost breakdown shown on an e-commerce checkout page?",
        "Is there a penalty for submitting documents late to a private organization?"
    ]
    expected_outputs = [
        "yes",
        "yes",
        "yes",
        "yes",
        "yes",
        "no",
        "no",
        "no",
        "no",
        "no"
    ]
    test_queries = [
            "How do I file my income tax return?",
            "What are the tax brackets for this year?",
            "Can I claim deductions for my home office?",
            "I received an audit notice from the IRS.",
            "What is the penalty for late filing?",
            "Capital gains tax calculation",
            "GST and VAT regulations",
            "Corporate tax liability"
        ]


    answers = []

    for query in test_queries:
        print(f"Query: {query}")
        try:
            result = tester.verify_query(query)
            answers.append(result)
            print(f"Is Tax Related? -> {result}")
        except Exception as e:
            print(f"Error processing query: {e}")
    
    print("="*50)

    for i in range(len(test_queries)):
        print(f"Query: {test_queries[i]}")
        # print(f"Expected Output: {expected_outputs[i]}", end="\t ")
        print(f"Actual Output: {answers[i]}")
        print("-"*50)

if __name__ == "__main__":
    main()
