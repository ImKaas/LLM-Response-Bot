import argparse
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain.prompts import ChatPromptTemplate

# === Configuration ===
CHROMA_PATH = "chroma"
MIN_RELEVANCE_SCORE = 0.5

# === Prompt Templates ===
CLEAN_PROMPT = """
You are a helpful assistant that improves user questions.

Rewrite the following question to be grammatically correct, clear, and professional without changing its original intent:

"{question}"
"""

RAG_PROMPT = """
If someone greets you, you reply them back nicely.

You are a knowledgeable and friendly business assistant helping users learn more about the services offered in the following company profile.

Use only the information provided in the context below. If a user's question is vague or indirect, do your best to interpret their intent and connect it to relevant offerings in the context.

The context may describe services in general terms (e.g., "website development", "SEO") without naming specific industries. You should explain how those services could apply to the user's specific case (e.g., doctors, gyms, educators, etc.) if reasonable.

If the user's question cannot be answered directly from the context, provide the closest helpful information available. Be creative, helpful, and concise.

If there truly is no relevant information to answer the question meaningfully, respond politely and say something like:
"Sorry, I couldn't find a relevant answer to your question in the available information, but I'd love to help — feel free to reach out to us at contact@bizleap.in or call +91-9876543210."

Format your entire response in plain text only. Do not use any Markdown, HTML, or special characters like asterisks, underscores, or backticks. Just simple lines and punctuation.
---

Context:
{context}

---

Answer this question using only the context above: {question}
"""

async def run_query(original_query: str):
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("Missing GOOGLE_API_KEY in environment.")

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    cleaner_prompt = ChatPromptTemplate.from_template(CLEAN_PROMPT)
    cleaned_query = (await model.ainvoke(cleaner_prompt.format(question=original_query))).content.strip()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    results = db.similarity_search_with_relevance_scores(cleaned_query, k=4)
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    prompt = rag_prompt.format(context=context, question=cleaned_query)

    response = await model.ainvoke(prompt)
    return response.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Your question")
    args = parser.parse_args()

    run_query(args.query_text)


if __name__ == "__main__":
    main()
