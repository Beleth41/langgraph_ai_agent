from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
from pathlib import Path
import settings
import os

#llm = ChatOllama(model="qwen3.5:0.8b")

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

#file_path = "C:/Users/mehdi/OneDrive/Desktop/multi_model_rag/docs"
#chroma_path = "db/chroma_db"


#loading
def loading_documents(doc_path:str):
    """This function load .txt, .pdf and .docx files from a directory"""

    folder = Path(doc_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Directory not found {folder}")

    documents = []

    #loading pdfs
    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            pdf_loader = PyPDFLoader(str(file_path))
            documents.extend(pdf_loader.load())

        elif suffix == ".txt":
            txt_loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(txt_loader.load())
        elif suffix == ".docx":
            docx_loader = Docx2txtLoader(str(file_path))
            documents.extend(docx_loader.load())
    return documents

    # pdf_loader = PyPDFDirectoryLoader(doc_path)
    # documents.extend(pdf_loader.load())


    #return loader.load()



#splitting


def split_documents(docs:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        length_function=len,
        is_separator_regex=True,
        separators=[r"\n\n", r"\n", r"(?<=[.?!])\s+"]
        #separators = ["\n\n", "\n", "."]
    )
    return text_splitter.split_documents(docs)


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    return embeddings

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index +=1
        else:
            current_chunk_index=0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id=current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks: list[Document], batch_size:int):
    db = Chroma(
        persist_directory=settings.chroma_path,
        embedding_function=get_embedding_function()
    )
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

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
        #new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        #db.add_documents(new_chunks, ids=new_chunk_ids)
        #db.persist()
    print("All batches added successfully")


# #creating vector store
# def creating_vectore_store(chunks, persist_directory=chroma_path, batch_size=25):
#     print("Creating database")
#     if not os.path.exists(persist_directory):
#         os.mkdir(persist_directory)

#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i + batch_size]
#         vectorestore = Chroma.from_documents(
#             documents=batch,
#             embedding=get_embedding_function()
#             persist_directory=persist_directory,
#             collection_configuration={"hnsw:space": "cosine"}
#         )
#         print(f"processed batches {i // batch_size +1} of {len(chunks) // batch_size +1}")
#     print("vector store created successfully")
#     return vectorestore




print("Loading documents...")
documents = loading_documents(settings.file_path)
if len(documents) == 0:
    raise FileNotFoundError(f"No PDF files found in {settings.file_path}")
print("documents loaded successfully")

print(f"Loaded {len(documents)} pages")

print("chunking...")
chunks = split_documents(documents)
print(f"created {len(chunks)} chunks")

print("adding chunks to ChromaDB")
add_to_chroma(chunks, batch_size=50)
print("Ingstion complete")