#app.py
import streamlit as st
import uuid
import hashlib
import shutil
import re

from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import settings
import ingestion


st.set_page_config(
    page_title="Rag",
    layout="wide"
)

@st.cache_resource
def load_backend():
    from retrieval import agent, db
    return agent, db


try:
    rag_agent, vector_db = load_backend()
except Exception as e:
    st.error(f"Error loading agent: {e}")
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



#display indexed documents
def get_indexed_documents(db):
    """
    Retrieve indexed documents and group their
    chunks, images and summaries by source.
    """

    documents = {}

    offset = 0
    batch_size = 1000

    while True:

        results = db.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset
        )

        ids = results.get("ids", [])
        metadatas = results.get("metadatas") or []

        if not ids:
            break

        for metadata in metadatas:

            if not metadata:
                continue

            source = metadata.get("source")

            if not source:
                continue

            # Initialize document entry
            if source not in documents:

                documents[source] = {
                    "chunks": 0,
                    "images": 0,
                    "types": set(),
                    "has_summary": False
                }

            content_type = metadata.get(
                "content_type",
                "document"
            )

            # Handle summaries separately
            if content_type == "document_summary":

                documents[source]["has_summary"] = True

                continue

            # Count actual content chunks
            documents[source]["chunks"] += 1

            documents[source]["types"].add(content_type)

            if content_type == "image":
                documents[source]["images"] += 1

        offset += len(ids)

        if len(ids) < batch_size:
            break

    return documents



def collect_sources(config) -> list[dict]:
    """Sources retrieved while answering the latest question."""
    messages = rag_agent.get_state(config).values["messages"]
    sources = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.artifact:
            sources.extend(message.artifact)
    return sorted(sources, key=lambda s: s["number"])


def cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    """Keep the sources the answer cites. If no citation is detected, keep all."""
    cited = set()
    pattern = r"Sources?\s+(\d+(?:\s*(?:,|and|et|&)\s*\d+)*)"
    for group in re.findall(pattern, answer, flags=re.IGNORECASE):
        cited.update(int(n) for n in re.findall(r"\d+", group))
    return [s for s in sources if s["number"] in cited] or sources


@st.cache_data(show_spinner=False)
def read_file_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


@st.cache_data(show_spinner=False)
def render_pdf_page(path: str, page: int):
    import pypdfium2 as pdfium   # installed together with docling
    pdf = pdfium.PdfDocument(path)
    try:
        return pdf[page - 1].render(scale=1.5).to_pil()
    finally:
        pdf.close()


def render_sources(sources, key_prefix: str) -> None:
    if not sources:
        return

    st.markdown("**Sources**")

    for s in sources:
        label = f"Source {s['number']} · {s['filename']}"
        if s["page"] is not None:
            label += f" · page {s['page']}"

        with st.expander(label):
            kind = "Image description" if s["content_type"] == "image" else "Passage"
            st.caption(f"{kind} · relevance {s['score']:.2f}")

            # The picture itself, for image chunks
            image_path = s.get("image_path")
            if s["content_type"] == "image" and image_path and Path(image_path).is_file():
                st.image(image_path)

            st.code(s["text"], language=None, wrap_lines=True)

            original = Path(s["source"])
            if not original.is_file():
                st.caption("The original file is no longer on disk.")
                continue

            # Page preview (PDF only)
            if original.suffix.lower() == ".pdf" and s["page"] is not None:
                if st.checkbox("Show the original page",
                               key=f"{key_prefix}_page_{s['number']}"):
                    try:
                        st.image(render_pdf_page(str(original), s["page"]))
                    except Exception as e:
                        st.caption(f"Could not render the page: {e}")

            st.download_button(
                "Download original file",
                data=read_file_bytes(str(original)),
                file_name=s["filename"],
                key=f"{key_prefix}_download_{s['number']}",
            )



#delete a document from the knowledge base
def delete_document(
    db,
    source: str,
    delete_original: bool = False
):
    """
    Delete all Chroma records belonging to a source.

    Optionally delete the original file, but only
    if it is inside the configured docs directory.
    """

    source_path = Path(source).resolve()

    docs_root = Path(settings.file_path).resolve()

    if delete_original:

        try:
            source_path.relative_to(docs_root)

        except ValueError:
            raise ValueError(
                "Cannot delete a file outside the "
                "configured documents directory."
            )

    # Retrieve matching chunk IDs and image metadata
    records = db.get(
        where={"source": source},
        include=["metadatas"]
    )

    ids = records.get("ids", [])
    metadatas = records.get("metadatas") or []

    # Collect extracted image paths
    image_paths = [
        metadata.get("image_path")
        for metadata in metadatas
        if metadata and metadata.get("image_path")
    ]

    # Delete all indexed chunks
    if ids:
        db.delete(ids=ids)

    # Clean up extracted images belonging to this document
    image_root = Path("extracted_images").resolve()

    for image_path in image_paths:

        path = Path(image_path).resolve()

        try:
            path.relative_to(image_root)

            if path.is_file():
                path.unlink()

            try:
                path.parent.rmdir()
            except OSError:
                pass

        except ValueError:
            # Never delete images outside our managed directory
            continue

    # Optionally delete the original document
    if delete_original and source_path.is_file():
        source_path.unlink()

    return len(ids)

#knowledge base
def knowledge_base_page():

    st.title("Knowledge Base")

    st.caption(
        "Manage your documents and indexed knowledge."
    )

    # ====================================
    # UPLOAD DOCUMENTS
    # ====================================

    st.subheader("Add documents")

    st.write(
        "Upload documents to process and add them "
        "to your knowledge base."
    )

    with st.form("upload_documents"):

        uploaded_files = st.file_uploader(
            "Choose documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        submitted = st.form_submit_button(
            "Add to knowledge base",
            type="primary"
        )

    if submitted:

        if not uploaded_files:

            st.warning("Choose at least one document.")

        else:

            docs_directory = Path(
                settings.file_path
            ).resolve()

            docs_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            for uploaded_file in uploaded_files:

                # Sanitize the uploaded filename
                filename = Path(
                    uploaded_file.name.replace("\\", "/")
                ).name

                extension = Path(filename).suffix.lower()

                if extension not in {".pdf", ".docx", ".txt"}:

                    st.error(
                        f"Unsupported file: {filename}"
                    )

                    continue

                target_path = docs_directory / filename

                # Avoid accidental overwrites
                if target_path.exists():

                    st.warning(
                        f"{filename} already exists. "
                        "Remove the existing file or rename "
                        "the new upload before importing it."
                    )

                    continue

                try:

                    # Save uploaded file locally
                    target_path.write_bytes(
                        uploaded_file.getvalue()
                    )

                    with st.spinner(
                        f"Processing {filename}..."
                    ):

                        # Run your existing ingestion pipeline
                        chunks = ingestion.loading_and_chunking_documents(
                            str(target_path)
                        )

                        if not chunks:

                            raise ValueError(
                                "No chunks were generated."
                            )
                        
                        summary_chunks = ingestion.generate_document_summaries(
                            chunks
                        )

                        all_chunks = chunks + summary_chunks

                        # Add chunks to the existing Chroma instance
                        ingestion.add_to_chroma(
                            all_chunks,
                            batch_size=50,
                            db=vector_db
                        )

                    st.success(
                        f"{filename} successfully indexed "
                        f"({len(chunks)} chunks)."
                    )

                except Exception as e:

                    # Avoid leaving a failed upload in docs
                    if target_path.exists():
                        target_path.unlink()

                    st.error(
                        f"Failed to process {filename}: {e}"
                    )

    st.divider()

    # ====================================
    # KNOWLEDGE BASE STATISTICS
    # ====================================

    st.subheader("Your documents")

    if st.button("Refresh knowledge base"):
        st.rerun()

    documents = get_indexed_documents(vector_db)

    total_documents = len(documents)

    total_chunks = sum(
        item["chunks"]
        for item in documents.values()
    )

    total_images = sum(
        item["images"]
        for item in documents.values()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Documents", total_documents)
    col2.metric("Total chunks", total_chunks)
    col3.metric("Images", total_images)

    st.divider()

    # ====================================
    # DOCUMENT LIST
    # ====================================

    if not documents:

        st.info(
            "Your knowledge base is empty. "
            "Upload a document to get started."
        )

        return

    st.write(
        f"**Indexed documents ({total_documents})**"
    )

    for source, info in sorted(
        documents.items(),
        key=lambda item: item[0].lower()
    ):

        filename = source.replace("\\", "/").split("/")[-1]

        document_key = hashlib.sha1(
            source.encode("utf-8")
        ).hexdigest()[:12]

        with st.expander(
            f"{filename}",
            expanded=False
        ):

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Chunks",
                info["chunks"]
            )

            col2.metric(
                "Images",
                info["images"]
            )

            col3.metric(
                "Content types",
                len(info["types"])
            )

            #show summary availability
            if info["has_summary"]:

                st.success("Document overview available")

            else:

                st.warning("Document overview not generated")

            st.caption("Original source")
            st.code(source)

            source_exists = Path(source).is_file()

            if not source_exists:
                st.warning(
                    "The original file is missing from disk, "
                    "but its indexed chunks still exist."
                )

            st.divider()

            st.write("**Document actions**")

            delete_original = st.checkbox(
                "Also delete the original file from my docs folder",
                key=f"delete_file_{document_key}"
            )

            confirm = st.checkbox(
                "I confirm that I want to remove this document",
                key=f"confirm_{document_key}"
            )

            if not source_exists:
                st.caption(
                    "Only the indexed records can be removed."
                )

            if st.button(
                "Remove from knowledge base",
                key=f"remove_{document_key}",
                disabled=not confirm,
                type="secondary"
            ):

                try:

                    deleted_count = delete_document(
                        vector_db,
                        source,
                        delete_original=delete_original
                    )

                    st.success(
                        f"Removed {filename}: "
                        f"{deleted_count} chunks deleted."
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Could not remove document: {e}"
                    )
    st.divider()
    st.subheader("Document summaries")

    if st.button("Generate missing summaries"):

        with st.spinner(
            "Analyzing your documents and generating summaries..."
        ):

            summary_docs = (
                ingestion.generate_missing_document_summaries(
                    vector_db
                )
            )

            if summary_docs:

                ingestion.add_to_chroma(
                    summary_docs,
                    batch_size=50,
                    db=vector_db
                )

                st.success(
                    f"Generated {len(summary_docs)} document summaries."
                )

            else:

                st.info(
                    "All indexed documents already have summaries."
                )

        st.rerun()
    with st.expander("Maintenance"):
        if st.button("Delete ALL summaries (then regenerate)"):
            removed = ingestion.delete_all_summaries(vector_db)
            st.warning(f"Deleted {removed} summary records. Now click 'Generate missing summaries'.")




#sidebar
with st.sidebar:
    st.title("Rag Agent")
    st.caption("This is a simple RAG agent that can answer questions based on the provided context.")
    st.divider()

    # if st.button("+ New conversation", use_container_width=True):
    #     st.session_state.thread_id = str(uuid.uuid4())
    #     st.session_state.chat_history = []
    #     st.rerun()

    page = st.radio(
        "Workspace",
        ["Chat", "Knowledge Base"],
        format_func=lambda x: (
            "Chat" if x == "Chat"
            else "Knowledge Base"
        ),
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Workspace")

    st.markdown("Document knowledge base")

    st.caption("Your indexed documents are stored locally.")

    st.divider()

    st.markdown("### Models")

    st.markdown("**LLM**")
    st.caption(settings.chat_model)

    st.markdown("**Embedding model**")
    st.caption(settings.embedding_model)

    st.divider()

    st.caption("RAG - Local")


# ==========================================
# MAIN CHAT INTERFACE
# ==========================================

if page == "Knowledge Base":

    knowledge_base_page()

    st.stop()

st.title("RAG Agent")

st.caption(
    "Ask questions, explore your documents, "
    "and discover information across your knowledge base."
)


# Display welcome message

if not st.session_state.chat_history:

    st.markdown("### What would you like to research?")

    st.info(
        "Ask a question about your indexed documents. "
        "RAG Agent will retrieve relevant evidence "
        "before generating an answer."
    )


# Display previous messages

for index, message in enumerate(st.session_state.chat_history):

    with st.chat_message(message["role"]):

        st.markdown(message["content"])
        render_sources(message.get("sources"), key_prefix=f"msg{index}")


# ==========================================
# USER INPUT
# ==========================================

if prompt := st.chat_input(
    "Ask something about your documents..."
):

    # Display user message

    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)


    # Generate assistant response

    # with st.chat_message("assistant"):

    #     with st.spinner(
    #         "RAG Agent is searching your knowledge base..."
    #     ):

    #         try:

    #             config = {
    #                 "configurable": {
    #                     "thread_id": st.session_state.thread_id
    #                 }
    #             }

    #             result = rag_agent.invoke(
    #                 {
    #                     "messages": [
    #                         HumanMessage(content=prompt)
    #                     ]
    #                 },
    #                 config=config
    #             )

    #             answer = result["messages"][-1].content

    #             st.markdown(answer)

    #             st.session_state.chat_history.append({
    #                 "role": "assistant",
    #                 "content": answer
    #             })

    #         except Exception as e:

    #             st.error(f"Something went wrong: {e}")

    with st.chat_message("assistant"):

        placeholder = st.empty()
        placeholder.markdown("_Searching your knowledge base..._")

        answer = ""
        current_step = None

        try:
            config = {
                "configurable": {"thread_id": st.session_state.thread_id}
            }

            for chunk, metadata in rag_agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="messages",
            ):
                # Only the "llm" node writes text; skip tool output
                if metadata.get("langgraph_node") != "llm" or not isinstance(chunk, AIMessage):
                    continue

                # A new LLM step means the previous one ended in a tool call,
                # so discard any text it wrote and start the answer fresh.
                step = metadata.get("langgraph_step")
                if step != current_step:
                    current_step = step
                    if answer:
                        answer = ""
                        placeholder.markdown("_Searching your knowledge base..._")

                if isinstance(chunk.content, str) and chunk.content:
                    answer += chunk.content
                    placeholder.markdown(answer + "▌")

            if not answer:
                # Fallback: read the finished answer from the checkpointed state
                answer = rag_agent.get_state(config).values["messages"][-1].content

            placeholder.markdown(answer)

            sources = cited_sources(answer, collect_sources(config))
            render_sources(sources, key_prefix=f"msg{len(st.session_state.chat_history)}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as e:
            st.error(f"Something went wrong: {e}")