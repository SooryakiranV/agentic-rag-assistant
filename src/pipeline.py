import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

from src.retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def retrieve_context(self, query: str, k: int = 3) -> list[dict]:
        """
        Retrieve relevant document chunks for a user query.
        """
        return self.retriever.retrieve(query, k=k)

    def generate_answer(self, query: str, k: int = 3) -> str:
        """
        Generate an answer using retrieved document context.
        """
        results = self.retrieve_context(query, k=k)

        context = "\n\n".join(
            f"[Page {result['page_number']}]\n{result['text']}"
            for result in results
        )

        prompt = f"""
Answer the user's question using only the provided document context.

Document context:
{context}

User question:
{query}

If the answer cannot be found in the document context, say that the information
is not available in the document.
"""

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        return response.choices[0].message.content