import os
import tempfile
import requests
import json
from flask import Flask, request, jsonify
from doc_loader import load_and_split_documents
from query_expansion import expand_query_and_thought
from auth import require_auth
import urllib.parse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

# Initialize LLM + Embedder
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    do_sample=False,
    repetition_penalty=1.0,
)

model=ChatHuggingFace(llm=llm)
embedder = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

@app.route("/hackrx/run", methods=["POST"])
@require_auth
def hackrx_run():
    data = request.get_json()

    document_url = data.get("documents")
    questions = data.get("questions")

    if not document_url or not questions:
        return jsonify({"error": "Both 'documents' and 'questions' fields are required."}), 400

    try:
        # Download file from the document URL
        response = requests.get(document_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download document from URL."}), 400

        parsed_url = urllib.parse.urlparse(document_url)
        path = parsed_url.path  # e.g. "/assets/policy.pdf"
        suffix = path.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + suffix) as tmp:
            tmp.write(response.content)
            tmp.flush()
            tmp_path = tmp.name

        # Wrap downloaded file in file-like object list
        with open(tmp_path, 'rb') as f:
            class UploadedFile:
                def __init__(self, file_obj, filename):
                    self.file = file_obj
                    self.filename = filename
                def save(self, path):
                    with open(path, 'wb') as out:
                        out.write(self.file.read())
                    self.file.seek(0)
            files = [UploadedFile(f, f"document.{suffix}")]

            # Load and split documents
            chunks = load_and_split_documents(files)

        # Vectorstore
        vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedder,
                collection_name="hackrx_memory_store"
            )

        answers = []

        # Process each question
        for q in questions:
            result = expand_query_and_thought(q)
            expanded_query = result["expanded_query"]
            thought_steps = result["thought_steps"]

            retrieved_docs = vectorstore.similarity_search(expanded_query, k=5)

            reasoning_prompt = f"""
You are a health policy expert. Use only the retrieved clauses below to answer the user's question accurately and concisely.

## Question:
{q}

## Reformulated Query:
{expanded_query}

## Reasoning:
{thought_steps}

## Policy Clauses:
{chr(10).join([doc.page_content for doc in retrieved_docs])}

## Instructions:
- Respond in 1–2 sentences only.
- Base your answer strictly on the provided clauses.
- Cite clause references only if relevant (e.g., "as per clause ii").
- Do NOT speculate or assume anything not stated.
- Do NOT include placeholder terms like [Insert Location] or unrelated geography.
- If information is missing, clearly say: "The policy document does not specify this."


## Answer:
"""
            response = model.invoke(reasoning_prompt)
            answers.append(response.content.strip())

        return jsonify({"answers": answers})

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

