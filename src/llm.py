from cohere import ClientV2
import os


def generate_answer(query, combined_similar_documents_content, similar_documents_sources):
    """Generate an answer using Cohere's LLM based on retrieved documents."""
    co = ClientV2(api_key=os.environ["COHERE_API_KEY"])

    cohere_prompt = (
        f"Based on the document content: "
        f"{combined_similar_documents_content}, "
        f"answer the question: '{query}'"
    )

    cohere_response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": cohere_prompt}],
        temperature=0.3
    )

    print("\n--- Answer ---")
    print(cohere_response.message.content[0].text)
    print("\n--- Sources ---")
    print(list(set(similar_documents_sources)))
