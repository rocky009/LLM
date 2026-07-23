#pip install langchain langchain-openai langchain-community chromadb python-dotenv

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (.env file)
load_dotenv()

# Verify API key presence
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Check your .env file.")


# ==========================================
# 1. Initialize Vector Database (Knowledge Base)
# ==========================================
def create_knowledge_base():

    # Sample domain knowledge text (Replace this with real documents or text files)
    sample_knowledge = [
        """
        The Acme Cloud Platform (ACP) offers three pricing tiers:
        - Starter Tier: $10/month, includes 5GB storage and 1,000 API requests.
        - Business Tier: $49/month, includes 100GB storage and 50,000 API requests with 24/7 support.
        - Enterprise Tier: Custom pricing, unlimited storage, and a dedicated account manager.
        """,
        """
        Acme Cloud Refund Policy:
        Customers can request a full refund within 14 days of purchase if they have not exceeded 
        50% of their monthly API request limits. To request a refund, email billing@acmecloud.example.
        """,
        """
        System Maintenance Schedule:
        Routine maintenance occurs every first Sunday of the month between 02:00 UTC and 04:00 UTC. 
        Service uptime guarantee is 99.9%.
        """
    ]

    # Split documents into smaller semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    docs = text_splitter.create_documents(sample_knowledge)

    # Convert text chunks to vector embeddings using OpenAI embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Index into Chroma vector database
    vector_store = Chroma.from_documents(docs, embeddings)
    print("✅ Knowledge base built successfully!\n")
    return vector_store.as_retriever(search_kwargs={"k": 2})


# ==========================================
# 2. Build the RAG Chain
# ==========================================
def build_rag_chain():
    # Initialize the OpenAI Chat Model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    # Prompt template incorporating Context and Chat History
    system_prompt = (
        "You are a helpful assistant for Acme Cloud Platform.\n"
        "Use the following retrieved context to answer the user's question accurately.\n"
        "If you do not know the answer based on the context, state that you don't know.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Output parser to clean string output
    output_parser = StrOutputParser()

    return llm, prompt, output_parser


# ==========================================
# 3. Interactive CLI Chatbot Loop
# ==========================================
def run_chatbot():
    retriever = create_knowledge_base()
    llm, prompt, output_parser = build_rag_chain()

    # Track multi-turn conversation memory
    chat_history = []

    print("🤖 Acme Cloud RAG Chatbot is ready! Type 'exit' or 'quit' to end.\n" + "-" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        # Step 1: Retrieve relevant context chunks from the vector database
        relevant_docs = retriever.invoke(user_input)
        context_str = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Step 2: Format the prompt payload
        formatted_prompt = prompt.format_messages(
            context=context_str,
            chat_history=chat_history,
            question=user_input
        )

        # Step 3: Stream response tokens to console
        print("Bot: ", end="", flush=True)
        response_text = ""
        
        for chunk in llm.stream(formatted_prompt):
            content = chunk.content
            print(content, end="", flush=True)
            response_text += content
        print()

        # Step 4: Save interaction turn to memory
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response_text))


if __name__ == "__main__":
    run_chatbot()