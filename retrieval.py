#retrieval.py
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
from pathlib import Path
import settings
import ingestion
import os

llm = ChatOllama(model=settings.chat_model, base_url=settings.ollama_base_url)

embeddings = OllamaEmbeddings(model=settings.embedding_model)


db = Chroma(
    persist_directory=settings.chroma_path,
    embedding_function=ingestion.get_embedding_function(),
    relevance_score_fn=lambda distance: 1 - distance / 2,
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


RELEVANCE_THRESHOLD = 0.35


def _search_knowledge_base(query: str, start_at: int = 1):
    """Search Chroma. Returns (text for the LLM, structured sources for the UI)."""

    results = db.similarity_search_with_relevance_scores(query, k=5)

    print(f"\nSearch query: {query}")
    print(f"Retrieved {len(results)} documents")

    relevant = []
    for i, (doc, score) in enumerate(results, start=1):
        accepted = score >= RELEVANCE_THRESHOLD
        print(f"  Document {i} | score {score:.3f} | "
              f"{'accepted' if accepted else 'REJECTED'} | {doc.metadata}")
        if accepted:
            relevant.append((doc, score))

    print(f"Accepted {len(relevant)} of {len(results)} documents")

    if not relevant:
        return (
            "NO_RELEVANT_EVIDENCE: The knowledge base did not return "
            "passages above the relevance threshold. Do not answer "
            "from general knowledge. Tell the user the documents "
            "do not contain enough relevant information.",
            [],
        )

    passages = []
    sources = []

    for number, (doc, score) in enumerate(relevant, start=start_at):
        meta = doc.metadata
        source = meta.get("source", "Unknown source")
        filename = Path(source).name
        page = meta.get("page")
        content_type = meta.get("content_type", "document")
        page_text = f", page {page}" if page is not None else ""

        passages.append(
            f"[Source {number}: {filename}{page_text}; "
            f"type={content_type}; relevance={score:.3f}]\n"
            f"{doc.page_content}"
        )

        sources.append({
            "number": number,
            "source": source,
            "filename": filename,
            "page": page,
            "content_type": content_type,
            "image_path": meta.get("image_path"),
            "score": float(score),
            "text": doc.page_content,
        })

    return "\n\n".join(passages), sources


@tool(response_format="content_and_artifact")
def retriever_tool(query: str):
    """This tool Searches the knowledge base and return relevant evidence."""
    return _search_knowledge_base(query)

@tool
def knowledge_base_overview_tool(query: str) -> str:
    """
    Retrieve document-level summaries from the entire knowledge base.

    Use this tool when the user asks:
    - What are my files about?
    - What documents do I have?
    - Give me an overview of my knowledge base.
    - What topics does a particular document cover?
    - What do my documents discuss?
    - Compare the subjects of my documents.

    Use retriever_tool for precise questions about facts,
    definitions, explanations, and details.
    """

    results = db.get(
        where={
            "content_type": "document_summary"
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    summaries = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not summaries:
        return (
            "NO_DOCUMENT_SUMMARIES_FOUND: "
            "No document-level summaries exist yet. "
            "Generate missing summaries from the Knowledge Base page."
        )

    # Organize summaries by filename
    documents = []

    for summary, metadata in zip(summaries, metadatas):

        if not summary or not metadata:
            continue

        filename = metadata.get(
            "document_name",
            metadata.get("source", "Unknown document")
        )

        documents.append(
            f"[Document: {filename}]\n{summary}"
        )

    # Sort by filename
    documents.sort(key=lambda item: item.lower())

    # Prevent excessively large tool responses
    max_documents = 20

    output = documents[:max_documents]

    response = (
        f"KNOWLEDGE BASE OVERVIEW\n"
        f"Number of indexed document summaries: {len(documents)}\n\n"
        + "\n\n---\n\n".join(output)
    )

    if len(documents) > max_documents:

        response += (
            f"\n\nShowing {max_documents} of "
            f"{len(documents)} summaries."
        )

    return response


tools=[retriever_tool, knowledge_base_overview_tool]

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
8. Some retrieved passages originate from images and contain
   AI-generated visual descriptions. Use them as evidence, but
   do not invent unreadable text, labels, or numerical values.
9. When the user asks about the overall contents of the knowledge
   base, the subjects of documents, or what files are about,
   use knowledge_base_overview_tool.
10. When the user asks for specific facts, explanations,
    definitions, or details, use retriever_tool.
11. When the user asks about a particular document, use its
    document-level summary to understand its overall subject.
12. Document summaries are generated descriptions of the original
    documents. Use them for orientation, but retrieve original
    passages when answering questions requiring precise evidence.
13. You may use both tools when a question requires understanding
    the overall document and retrieving specific supporting evidence.
14. Do not assume that a document discusses something just because
    its filename suggests it. Base your answer on its indexed content.
15. Always answer in the language of the user's latest question (for example,
    French for a French question), even if the documents are in another language.
    Keep source labels such as [Source 1, page 3] unchanged, keep quotations and
    technical terms in their original language, and translate the refusal
    sentence from rule 3 into the user's language.
16. For any question about a person, name or entity ("who is X", "what is Y"),
    call retriever_tool with a keyword query (for example "Mehdi resume profile")
    and also call knowledge_base_overview_tool. Never say the documents lack
    information before you have searched.
"""

def call_llm(state:AgentState) -> AgentState:
    """Function to call llm with the current state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    messages = llm.invoke(messages)
    return {"messages":[messages]}

tools_dict = {a_tools.name: a_tools for a_tools in tools}

def _sources_already_shown(messages) -> int:
    """How many sources earlier searches in the current turn already numbered."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.artifact:
            count += len(message.artifact)
    return count

#Retriever Agent: executing tool
def take_action(state: AgentState):
    """Execute tool calls from the llm response."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    shown = _sources_already_shown(state["messages"])

    for t in tool_calls:
        query = t["args"].get("query", "")
        print(f"Calling tool: {t['name']} with query {query or 'no query provided'}")
        artifact = None

        if t["name"] not in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect tool name, please try and select tool from list of available tools."
        elif t["name"] == "retriever_tool":
            result, artifact = _search_knowledge_base(query, start_at=shown + 1)
            shown += len(artifact)
        else:
            result = tools_dict[t["name"]].invoke(query)

        print(f"Result length: {len(str(result))}")
        results.append(ToolMessage(
            tool_call_id=t["id"],
            name=t["name"],
            content=str(result),
            artifact=artifact,
        ))

    print("Tools execution complete, back to the model!")
    return {"messages": results}


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


if __name__ == "__main__":
    running_rag()
