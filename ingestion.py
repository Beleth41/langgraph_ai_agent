#ingestion.py
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from pathlib import Path
from hashlib import sha256
from collections import Counter, defaultdict
from typing import Optional
import time
import settings

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling.datamodel.backend_options import PdfBackendOptions
from docling_core.types.doc import PictureItem

from ollama import chat as ollama_chat


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Pictures whose longest side is below this (in pixels) are almost always
# icons, bullets or logos. They are not sent to the vision model.
MIN_IMAGE_SIDE = 100

# Used for document summaries. num_ctx matters: a 14 000 character batch plus the
# prompt is ~4-5k tokens, which does not fit in Ollama's default context window
# (often 4096), so the model would silently truncate its input.
summary_llm = ChatOllama(
    model=settings.chat_model,
    base_url=settings.ollama_base_url,
    temperature=0.1,
    num_predict=450,
    num_ctx=8192,
    reasoning=False,
    # If qwen3.5 "thinks" before answering, those tokens eat num_predict and slow
    # everything down. With a recent langchain-ollama you can turn that off:
    # reasoning=False,
)


#loading and splitting with docling:
pipeline_options = PdfPipelineOptions(do_table_structure= True, generate_picture_images= True,generate_page_images= True, images_scale=2.0)
pipeline_options.table_structure_options.do_cell_matching = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend_options=PdfBackendOptions(include_bitmap_images=True))
    }
)
chunker = HybridChunker()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""],
)


def get_embedding_function():
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_indexed_sources(db, batch_size: int = 1000) -> set:
    """Return the sources that already have real content (not just a summary) in the DB."""

    sources = set()
    offset = 0

    while True:
        records = db.get(include=["metadatas"], limit=batch_size, offset=offset)

        ids = records.get("ids", [])
        if not ids:
            break

        for metadata in records.get("metadatas") or []:
            if not metadata or not metadata.get("source"):
                continue
            if metadata.get("content_type") == "document_summary":
                continue
            sources.add(metadata["source"])

        offset += len(ids)
        if len(ids) < batch_size:
            break

    return sources

def _chunk_page(chunk) -> Optional[int]:
    """First page a chunk appears on (None if Docling has no page info, e.g. DOCX)."""
    pages = [
        prov.page_no
        for item in getattr(chunk.meta, "doc_items", [])
        for prov in getattr(item, "prov", [])
    ]
    return min(pages) if pages else None

def loading_and_chunking_documents(
    doc_path: str,
    skip_sources: Optional[set] = None
) -> list[Document]:
    """Parse PDF/DOCX with Docling; load TXT directly.

    Files whose source path is in skip_sources are not processed again.
    """

    skip_sources = skip_sources or set()

    folder = Path(doc_path).resolve()

    if folder.is_file():
        files = [folder]

    elif folder.is_dir():
        files = [
            path for path in folder.rglob("*")
            if path.is_file()
        ]

    else:
        raise FileNotFoundError(
            f"Path not found: {folder}"
        )

    chunks = []

    for file_path in files:
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        source = str(file_path.resolve())

        if source in skip_sources:
            print(f"Skipping already indexed file: {file_path.name}")
            continue

        print(f"Processing {file_path.name}")

        if suffix in {".pdf", ".docx"}:

            result = converter.convert(source)
            docling_doc = result.document

            # 1. Extract and understand images
            image_docs = extract_image_documents(
                docling_doc,
                file_path,
                source
            )

            # 2. Extract and chunk document content
            docling_chunks = list(chunker.chunk(docling_doc))

            for index, chunk in enumerate(docling_chunks):

                text = chunker.contextualize(chunk).strip()

                if not text:
                    continue

                text = f"Document: {file_path.stem.replace('_', ' ')}\n{text}"

                page_no = _chunk_page(chunk)

                chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "page": page_no,
                            "chunk_index": index,
                            "content_type": "document",
                        },
                    )
                )

            # 3. Add image descriptions to the same collection
            chunks.extend(image_docs)

        elif suffix == ".txt":
            loaded_docs = TextLoader(
                source, encoding="utf-8"
            ).load()

            for doc in loaded_docs:
                doc.metadata["source"] = source

            chunks.extend(text_splitter.split_documents(loaded_docs))

    return chunks


def extract_image_documents(
    docling_doc,
    file_path: Path,
    source: str
) -> list[Document]:

    image_docs = []

    # Create a unique image directory for this document
    file_hash = sha256(source.encode("utf-8")).hexdigest()[:10]

    output_dir = (
        Path("extracted_images") /
        f"{file_path.stem}_{file_hash}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    image_index = 0

    for item, _level in docling_doc.iterate_items():

        if not isinstance(item, PictureItem):
            continue

        image = item.get_image(docling_doc)

        if image is None:
            continue

        # Skip tiny pictures (icons, bullets, logos)
        if max(image.size) < MIN_IMAGE_SIDE:
            continue

        image_index += 1

        # Get page number
        provenance = getattr(item, "prov", [])

        page_no = (
            provenance[0].page_no
            if provenance
            else None
        )

        # Save image
        image_path = output_dir / (
            f"picture_{image_index:03d}_page_{page_no or 'unknown'}.png"
        )

        image.save(image_path, format="PNG")

        print(f"\nAnalyzing image: {image_path}")

        # Send image to the vision model
        try:
            description = describe_image(
                image_path,
                file_path.name
            )

        except Exception as e:
            print(f"Image analysis failed: {e}")
            continue

        if not description:
            continue

        print(f"Image description:\n{description}")

        # Create a searchable document
        image_document = Document(
            page_content=(
                f"Visual content from document: {file_path.name}\n"
                f"Page: {page_no or 'Unknown'}\n\n"
                f"Image description:\n{description}"
            ),

            metadata={
                "source": source,
                "page": page_no,
                "content_type": "image",
                "image_path": str(image_path.resolve()),
                "image_index": image_index,
            }
        )

        image_docs.append(image_document)

    return image_docs


def calculate_chunk_ids(chunks: list[Document]):

    for position, chunk in enumerate(chunks):

        metadata = chunk.metadata

        index = metadata.get(
            "chunk_index",
            metadata.get("image_index", position)
        )

        if metadata.get("content_type") == "document_summary":
            # One summary per document. The ID must NOT depend on the text,
            # because the LLM writes a different summary every run and that
            # would pile up duplicate summaries.
            unique_key = f"{metadata['source']}:document_summary"
        else:
            unique_key = (
                f"{metadata['source']}:"
                f"{metadata.get('content_type', 'document')}:"
                f"{metadata.get('page')}:"
                f"{index}:"
                f"{chunk.page_content}"
            )

        chunk.metadata["id"] = sha256(
            unique_key.encode("utf-8")
        ).hexdigest()

    return chunks


VISION_MODEL = settings.vision_model
def describe_image(image_path: Path, document_name: str) -> str:
    """
    Analyze an image using a local vision-language model
    and return a textual description.
    """

    prompt = f"""
    You are analyzing an image extracted from a document:
    {document_name}

    Produce an informative description for a RAG knowledge base.

    Instructions:
    1. Identify the type of visual content:
       photograph, diagram, flowchart, chart, screenshot, etc.

    2. Describe the important information visible.

    3. If it is a chart:
       - Identify its title.
       - Describe the X and Y axes.
       - Identify units and legends.
       - Extract visible numbers and labels.
       - Explain the trends and comparisons.

    4. If it is a diagram:
       - Identify its components.
       - Explain their relationships.
       - Describe the direction of arrows and connections.

    5. Extract all readable text when relevant.

    6. Never invent missing values or labels.
       Explicitly mention when something is unreadable.

    7. For technical and scientific diagrams:
       - Identify every major component.
       - Explain the relationships between components.
       - Follow directional arrows and information flow.
       - Preserve technical terminology.
       - Distinguish between separate blocks and subcomponents.

    8. For neural network architectures:
       - Identify inputs and outputs.
       - Describe the encoder and decoder if present.
       - Identify attention mechanisms.
       - Explain the connections between layers.
       - Identify dimensions and numerical parameters
         when visibly specified.

    9. Describe only what is actually visible.
       Do not fill missing information using external knowledge.

    10. Write the description in English, but copy any visible text exactly as
        written, in its original language. Do not translate it.

    Be detailed, accurate and structured.
    """
    response = ollama_chat(model=VISION_MODEL,
                           messages=[
                               {
                                   "role":"user",
                                   "content":prompt,
                                   "images":[str(image_path)]
                               }
                           ],
                           options={"temperature":0.1, "num_predict":300},
                           think=False,
                           # think=False,  # with a recent ollama python package, if the vision model "thinks"
                           )
    return response.message.content.strip()



def add_to_chroma(chunks: list[Document], batch_size:int = 50, db=None):
    if db is None:
        db = Chroma(
            persist_directory=settings.chroma_path,
            embedding_function=get_embedding_function()
        )
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    id_counts = Counter(
        chunk.metadata["id"]
        for chunk in chunks_with_ids
    )

    duplicates = {
        chunk_id: count
        for chunk_id, count in id_counts.items()
        if count > 1
    }

    if duplicates:

        print("\n===== DUPLICATE ID DIAGNOSTICS =====")

        for duplicate_id, count in duplicates.items():

            print(f"\nDuplicate ID: {duplicate_id}")
            print(f"Occurrences: {count}")

            for chunk in chunks_with_ids:

                if chunk.metadata["id"] == duplicate_id:

                    print({
                        "source": chunk.metadata.get("source"),
                        "content_type": chunk.metadata.get("content_type"),
                        "chunk_index": chunk.metadata.get("chunk_index"),
                        "content_preview": chunk.page_content[:150]
                    })
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Remove duplicates within the incoming batch
    unique_chunks = {}

    for chunk in chunks_with_ids:

        chunk_id = chunk.metadata["id"]

        if chunk_id not in unique_chunks:
            unique_chunks[chunk_id] = chunk

    # Only add IDs that don't already exist in ChromaDB
    new_chunks = [
        chunk
        for chunk_id, chunk in unique_chunks.items()
        if chunk_id not in existing_ids
    ]

    print(f"Incoming chunks: {len(chunks_with_ids)}")
    print(f"Unique chunks: {len(unique_chunks)}")
    print(f"New chunks: {len(new_chunks)}")

    if len(new_chunks) == 0:
        print("No new chunks to add")
        return

    print(f"Adding {len(new_chunks)} chunks in  {batch_size}")
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]

        batch_ids = [chunk.metadata["id"] for chunk in batch]

        db.add_documents(
            documents=batch,
            ids=batch_ids
        )

        processed = min(i + len(batch), len(new_chunks))
        print(f"Processed {processed}/{len(new_chunks)} chunks")
    print("All batches added successfully")


# ======================================================================
# Document summaries
# ======================================================================

def _chunk_sort_key(chunk: Document):
    """Keep a document's chunks in reading order (page, then position)."""

    page = chunk.metadata.get("page")
    index = chunk.metadata.get("chunk_index", chunk.metadata.get("image_index", 0))

    return (
        page if isinstance(page, int) else 10**9,
        index if isinstance(index, int) else 0,
    )


def _build_batches(chunks: list[Document], max_chars: int) -> list[str]:
    """Group a document's chunks into text batches of at most max_chars."""

    parts = []

    for chunk in sorted(chunks, key=_chunk_sort_key):
        page = chunk.metadata.get("page")
        content_type = chunk.metadata.get("content_type", "document")

        parts.append(
            f"[{content_type.upper()}, PAGE {page if page is not None else 'unknown'}]\n"
            f"{chunk.page_content}"
        )

    batches = []
    current_batch = []
    current_size = 0

    for part in parts:

        # Handle an exceptionally large individual chunk
        for start in range(0, len(part), max_chars):
            piece = part[start:start + max_chars]

            if current_batch and current_size + len(piece) > max_chars:
                batches.append("\n\n".join(current_batch))
                current_batch = []
                current_size = 0

            current_batch.append(piece)
            current_size += len(piece)

    if current_batch:
        batches.append("\n\n".join(current_batch))

    return batches


def _pick_evenly(items: list, limit: int) -> list:
    """Pick at most `limit` items spread evenly, always keeping the first and last."""

    if len(items) <= limit:
        return items

    if limit == 1:
        return [items[0]]

    return [
        items[round(i * (len(items) - 1) / (limit - 1))]
        for i in range(limit)
    ]


def _run_summary_llm(prompt: str) -> str:
    response = summary_llm.invoke(prompt)
    text = (response.content or "").strip()

    if not text:
        meta = response.response_metadata or {}
        print(
            f"    Empty output | done_reason={meta.get('done_reason')} "
            f"| eval_count={meta.get('eval_count')} "
            f"| reasoning_returned={bool(response.additional_kwargs.get('reasoning_content'))}"
        )
    return text


def _summarize_document(
    filename: str,
    chunks: list[Document],
    max_chars: int,
    max_batches: int
) -> str:
    """Summarize ONE document. Returns an empty string if nothing was produced."""

    batches = _build_batches(chunks, max_chars)

    if not batches:
        return ""

    total_batches = len(batches)
    batches = _pick_evenly(batches, max_batches)
    sampled = total_batches > len(batches)

    if sampled:
        print(f"  Long document: summarizing {len(batches)} evenly spaced sections out of {total_batches}")

    partial_summaries = []

    for index, batch in enumerate(batches, start=1):

        if len(batches) > 1:

            print(f"  Summarizing section {index}/{len(batches)}")

            prompt = f"""
You are analyzing part of a document named:
{filename}

Summarize this portion using ONLY the supplied content.

Identify:
- Main topics and subjects.
- Important concepts and terminology.
- Section headings and their topics.
- Important arguments, findings or explanations.
- Tables, diagrams and visual content.
- Important numerical information when available.

Preserve the document's terminology.
Do not introduce external knowledge.
Do not invent missing information.

This is only part of the document.
Do not pretend it represents the entire document.

DOCUMENT CONTENT:

{batch}
"""

        else:

            prompt = f"""
Create a comprehensive overview of this document:

{filename}

Based ONLY on the supplied content, explain:

1. What the document is about.
2. Its main subject and purpose.
3. Its major sections and topics.
4. Its important concepts and terminology.
5. Its notable arguments, findings or explanations.
6. Its important diagrams, tables and visual content.

Write approximately 150-250 words.

Preserve the document's terminology.
Do not use external knowledge.
Do not invent information.

DOCUMENT CONTENT:

{batch}
"""

        text = _run_summary_llm(prompt)

        if text:
            partial_summaries.append(text)

    if not partial_summaries:
        return ""

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    print("  Combining partial summaries...")

    sampling_note = (
        "The sections were sampled evenly across the whole document.\n"
        if sampled else ""
    )

    final_prompt = f"""
You are creating a document-level overview.

Document: {filename}

Below are summaries of different portions
of the same document.
{sampling_note}
Combine them into one coherent overview.

Include:
- What the document is about.
- Its purpose and scope.
- Its main sections and topics.
- Its key concepts and terminology.
- Important findings or arguments.
- Significant visual content.

Avoid repetition.
Preserve important distinctions.
Do not introduce external knowledge.

Write approximately 200-300 words.

PARTIAL SUMMARIES:

{chr(10).join(partial_summaries)}
"""

    return _run_summary_llm(final_prompt)


def generate_document_summaries(
    chunks: list[Document],
    max_chars: int = 14000,
    max_batches: int = 4
) -> list[Document]:
    """
    Generate one summary per source document.

    Uses both text chunks and image descriptions.
    max_batches caps the LLM calls per document: a long document is sampled
    evenly instead of being summarized section by section from start to end.
    Raise it for more coverage, lower it for more speed.
    """

    chunks_by_source = defaultdict(list)

    # grouping chunks of the same file
    for chunk in chunks:
        if chunk.metadata.get("content_type") == "document_summary":
            continue
        source = chunk.metadata.get("source")
        if source:
            chunks_by_source[source].append(chunk)

    summary_documents = []

    for position, (source, file_chunks) in enumerate(chunks_by_source.items(), start=1):

        filename = Path(source).name
        print(f"\n[{position}/{len(chunks_by_source)}] Generating summary for {filename}")
        started = time.perf_counter()

        # One failing document must not stop the others
        try:
            summary = _summarize_document(
                filename, file_chunks, max_chars, max_batches
            )
        except Exception as e:
            print(f"  Summary failed for {filename}: {e}")
            continue

        if not summary:
            print(f"  The model returned an empty summary for {filename}, skipping it")
            continue

        summary_documents.append(
            Document(
                page_content=(
                    f"Document: {filename}\n\n"
                    f"Document overview:\n{summary}"
                ),

                metadata={
                    "source": source,
                    "document_name": filename,
                    "content_type": "document_summary",
                    "chunk_index": -1,
                    "summary_version": 1
                }
            )
        )

        print(f"  Summary generated in {time.perf_counter() - started:.0f}s")

    return summary_documents


#summaries for already existing files in the database
def generate_missing_document_summaries(
    db,
    batch_size: int = 500
) -> list[Document]:

    documents_by_source = defaultdict(list)
    existing_summaries = set()

    offset = 0

    while True:

        records = db.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset
        )

        ids = records.get("ids", [])
        texts = records.get("documents") or []
        metadatas = records.get("metadatas") or []

        if not ids:
            break

        for text, metadata in zip(texts, metadatas):

            if not text or not metadata:
                continue

            # Keep the source EXACTLY as stored. Normalizing it (lower-casing,
            # slashes) made the summary's source differ from its document's
            # source, so the UI listed the summary as a separate document.
            source = metadata.get("source")

            if not source:
                continue

            if metadata.get("content_type") == "document_summary":
                existing_summaries.add(source)
                continue

            documents_by_source[source].append(
                Document(
                    page_content=text,
                    metadata=dict(metadata)
                )
            )

        offset += len(ids)

        if len(ids) < batch_size:
            break

    # Identify documents without summaries
    missing_sources = [
        source
        for source in documents_by_source
        if source not in existing_summaries
    ]

    print("\n===== SUMMARY GENERATION REPORT =====")
    print(f"Unique indexed documents: {len(documents_by_source)}")
    print(f"Existing summaries: {len(existing_summaries)}")
    print(f"Missing summaries: {len(missing_sources)}")

    missing_chunks = []

    for source in missing_sources:
        missing_chunks.extend(documents_by_source[source])

    if not missing_chunks:
        print("No missing summaries.")
        return []

    return generate_document_summaries(missing_chunks)


def delete_all_summaries(db, batch_size: int = 500) -> int:
    """Remove every document_summary record (use once to clean up, then regenerate)."""

    records = db.get(
        where={"content_type": "document_summary"},
        include=[]
    )

    ids = records.get("ids", [])

    for start in range(0, len(ids), batch_size):
        db.delete(ids=ids[start:start + batch_size])

    return len(ids)


def main():
    started = time.perf_counter()

    db = Chroma(
        persist_directory=settings.chroma_path,
        embedding_function=get_embedding_function()
    )

    # Files that are already indexed are not parsed, described or embedded again
    indexed_sources = get_indexed_sources(db)
    print(f"{len(indexed_sources)} documents already indexed")

    print("Loading documents...")
    chunks = loading_and_chunking_documents(
        settings.file_path,
        skip_sources=indexed_sources
    )
    print(f"Loading took {time.perf_counter() - started:.0f}s")

    if chunks:
        print(f"created {len(chunks)} chunks")
        print("adding chunks to ChromaDB")
        add_to_chroma(chunks, batch_size=50, db=db)
    else:
        print("No new documents to index")

    # Chunks are saved before summarizing, so if summarizing is interrupted
    # the next run simply picks up the documents that still lack a summary.
    summary_docs = generate_missing_document_summaries(db)

    if summary_docs:
        add_to_chroma(summary_docs, batch_size=50, db=db)

    print(f"Ingestion complete in {time.perf_counter() - started:.0f}s")

if __name__ == "__main__":
    main()