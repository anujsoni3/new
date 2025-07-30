import os
import tempfile
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader
)
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

def inject_markdown_headers(text):
    """
    Tag waiting period and medical clauses using markdown headers.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        line = line.strip()

        # Match headings like "Two years waiting period"
        if re.search(r"(one|two|three|\d+)[-\s]?(year)?s?\s+waiting period", line, re.IGNORECASE):
            result.append(f"# {line}")
        # Match clause lines like "a. Cataract"
        elif re.match(r"^[a-zA-Z]\.\s+.+", line):
            result.append(f"### {line}")
        else:
            result.append(line)

    return "\n".join(result)

def load_and_split_documents(file_objects):
    """
    Loads PDF/DOCX/EML files, adds metadata, performs intelligent splitting using markdown headers.
    Falls back to character splitting if headers aren't found.
    """
    all_docs = []

    for file in file_objects:
        suffix = file.filename.split(".")[-1].lower()
        file_name = file.filename

        # Save file temporarily
        temp = tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix)
        file.save(temp.name)

        try:
            # Choose the right loader
            if suffix == "pdf":
                loader = PyMuPDFLoader(temp.name)
            elif suffix == "docx":
                loader = UnstructuredWordDocumentLoader(temp.name)
            elif suffix == "eml":
                loader = UnstructuredEmailLoader(temp.name)
            else:
                print(f"Unsupported file type: {file_name}")
                continue

            docs = loader.load()

            # Add metadata to each page/document
            for doc in docs:
                doc.metadata["source_file"] = file_name
                if "page_number" not in doc.metadata:
                    doc.metadata["page_number"] = None

            all_docs.extend(docs)

        except Exception as e:
            print(f"Failed to load {file_name}: {e}")
        finally:
            temp.close()
            os.unlink(temp.name)

    if not all_docs:
        return []

    # Combine all content for markdown processing
    combined_text = "\n\n".join([doc.page_content for doc in all_docs])
    markdown_text = inject_markdown_headers(combined_text)

    headers_to_split_on = [("#", "section"), ("###", "clause")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    try:
        md_chunks = md_splitter.split_text(markdown_text)

        # Attach metadata to each markdown chunk
        if md_chunks:
            final_chunks = []
            for chunk in md_chunks:
                final_chunks.append(Document(
                    page_content=chunk.page_content.strip(),
                    metadata={
                        "source_file": all_docs[0].metadata.get("source_file"),
                        "page_number": None  # Page can't be tracked due to merged text
                    }
                ))

            if len(final_chunks) >= 5:
                return final_chunks
            else:
                print("Too few markdown chunks. Falling back to character splitter...")

    except Exception as e:
        print(f"Markdown splitting failed: {e}")

    # === Fallback to recursive char splitter ===
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fallback_chunks = char_splitter.split_documents(all_docs)
    return fallback_chunks


# def create_and_save_faiss_vectorstore(chunks, output_dir="store"):
#     """
#     Embed and save the FAISS vector store with metadata.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     embedder = HuggingFaceInferenceAPIEmbeddings(
#     api_key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

#     vectorstore = Chroma.from_documents(
#                 documents=chunks,
#                 embedding=embedder,
#                 collection_name="hackrx_memory_store"
#             )
    

#     with open(os.path.join(output_dir, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
