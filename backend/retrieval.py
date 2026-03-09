import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

load_dotenv()

def answer_employee_question(question: str, chat_history: list = None):
    """Retrieves context from Pinecone and generates an answer with conversation memory."""
    
    if chat_history is None:
        chat_history = []

    # 1. Convert our standard dictionary history into LangChain Message objects
    lc_chat_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            lc_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_chat_history.append(AIMessage(content=msg["content"]))

    # 2. Connect to Vectorstore and LLM
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 

    # --- NEW: HISTORY-AWARE RETRIEVER ---
    # This prompt tells the LLM to rewrite follow-up questions so they make sense on their own
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
                   "formulate a standalone question which can be understood "
                   "without the chat history. Do NOT answer the question, "
                   "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. Update the QA System Prompt to accept chat history
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful HR assistant for the company. "
                   "Use the following pieces of retrieved context to answer the employee's question. "
                   "If you don't know the answer, just say 'I cannot find this in the company policies.'\n\n"
                   "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 4. Build the final chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 5. Run the chain, passing in both the input AND the formatted history
    response = rag_chain.invoke({
        "input": question, 
        "chat_history": lc_chat_history
    })
    
    # 6. Format the sources cleanly
    sources = []
    for doc in response["context"]:
        source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page_num = doc.metadata.get('page', 'Unknown')
        sources.append(f"{source_file} (Page {page_num})")

    # Remove duplicates from sources list while keeping order
    sources = list(dict.fromkeys(sources))

    return {
        "answer": response["answer"],
        "sources": sources
    }