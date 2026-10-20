from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, List
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from operator import add as add_message
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.documents import Document
import settings
import ingestion
import os

llm = ChatOllama(model="qwen3.5:9b")

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")


db = Chroma(
    persist_directory=settings.chroma_path,
    embedding_function=ingestion.get_embedding_function()
)

retriever = db.as_retriever(
    #search_type="similarity_score_threshold",
    search_kwargs={"k":5}
)


# @tool
# def retriever_tool(query:str) -> str:
#     """This tool Searches the knowledge base and return relevant evidence."""

#     docs = retriever.invoke(query)

#     if not docs:
#         return "I found no relatable information"

#     result = []
#     for i, doc in enumerate(docs):
#         print(f"Document {i+1}\n{doc.page_content}")
#         print(f"Metadata: {doc.metadata}\n\n")
#         result.append(f"Document {i+1}\n{doc.page_content}")
#     return "\n\n".join(result)


@tool
def retriever_tool(query:str) -> str:
    """This tool Searches the knowledge base and return relevant evidence."""

    results = db.similarity_search_with_relevance_scores(
        query,
        k=5
    )

    threshold = 0.40 

    print(f"\nSearch query: {query}")
    print(f"Retrieved {len(results)} documents")

    relevant = []

    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"Document {i} | Relevance score: {score:.3f}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content}")

        if score >= threshold:
            relevant.append((doc, score))
        else:
            print("Status: REJECTED (below relevance threshold)")

    print(f"\nAccepted {len(relevant)} of {len(results)} documents")

    if not relevant:
        return (
            "NO_RELEVANT_EVIDENCE: The knowledge base did not return "
            "passages above the relevance threshold. Do not answer "
            "from general knowledge. Tell the user the documents "
            "do not contain enough relevant information."
        )

    passages = []
    for i, (doc, score) in enumerate(relevant, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page")

        # PyPDFLoader usually stores zero-based page numbers.
        page_text = (
            f", page {page + 1}"
            if page is not None
            else ""
        )

        passages.append(
            f"[Source {i}: {source}{page_text}; "
            f"relevance={score:.3f}]\n"
            f"{doc.page_content}"
            )

    return "\n\n".join(passages)



tools=[retriever_tool]

llm = llm.bind_tools(tools)



class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_message]


def should_continue(state:AgentState):
    """This function checks if the last message contains tool calls"""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

NO_ANSWER = (
    "I couldn't find enough relevant information in the provided "
    "documents to answer that question."
)

system_prompt = """
You are a question-answering assistant restricted to the supplied documents.

Rules:
1. Answer using ONLY the evidence in the DOCUMENTS section.
2. Do not use your general knowledge to fill gaps.
3. If the documents do not contain enough evidence, say:
   "I couldn't find enough relevant information in the provided documents to answer that question."
4. Do not invent facts, quotations, page numbers, or sources.
5. Cite supporting evidence using the source labels provided, for example [Source 1, page 3].
6. You may use conversation history to understand pronouns and follow-up questions,
   but conversation history is not evidence. Every factual claim in your answer
   must be supported by the current retrieved documents.
7. If only part of a question is supported, answer only that part and clearly
   state what the documents do not establish.
"""

def call_llm(state:AgentState) -> AgentState:
    """Function to call llm with the current state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    messages = llm.invoke(messages)
    return {"messages":[messages]}

tools_dict = {a_tools.name: a_tools for a_tools in tools}


#Retriever Agent: executing tool
def take_action(state:AgentState):
    """Execute tool calls from the llm response."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling tool: {t['name']} with query {t['args'].get('query', 'no query provided')}")

        if not t['name'] in tools_dict: #checks if a valid tool is present
            print(f"\nTool: {t['name']  } does not exist.")
            result = "Incorrect tool name, please try and select tool from list of available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        #appends the tool message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'] ,content=str(result)))
    print("Tools execution complete, back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)

graph.add_edge("retriever_agent","llm")
graph.set_entry_point("llm")

checkpointer = InMemorySaver()
agent = graph.compile(checkpointer=checkpointer)



def running_rag():
    print("RAG AGENT =====")

    config = {
        "configurable":{
            "thread_id": "conversation_1"
        }
    }
    
    while True:
        user_input = input("\nWhat is your question?: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        messages = [HumanMessage(content=user_input)]
        result = agent.invoke({"messages": messages}, config=config)

        print("====ANSWER====\n")
        print(result["messages"][-1].content)


running_rag()