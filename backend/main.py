from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
import uuid
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import io
import traceback
import json
load_dotenv()

app = FastAPI(
    title="EdTech RAG Textbook Reader API",
    description="Multimodal RAG for PDF & Image Textbook Queries with Groq + Pinecone",
    version="1.0.0"
)

# CORS – allow Vercel frontend (update later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq init (only for generation)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Pinecone init
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "edtech-rag-index")

# Local embeddings (no Groq embeddings needed)
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Create Pinecone index if not exists (run once) – dimension 384 for bge-small
# Create Pinecone index if not exists (run once)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # ← BGE-small-en-v1.5 dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# OCR reader
reader = easyocr.Reader(['en'])

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "EdTech RAG API is live"}

@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    # Save temp
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        # Open and process PDF
        doc = fitz.open(temp_path)
        chunks = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            chunk = f"Page {page_num}: {text[:1000]}"  # truncate for demo
            chunks.append(chunk)

        doc.close()  # Close document BEFORE deleting file

        # Embed and upsert each chunk (local embedder)
        for chunk in chunks:
            vector = embedder.encode(chunk).tolist()

            index.upsert([{
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {"text": chunk, "source": file.filename}
            }])

        # Cleanup
        os.remove(temp_path)

        return {"status": "success", "message": f"Ingested {file.filename} with {len(chunks)} chunks"}
    except Exception as e:
        traceback.print_exc()  # Show real error in terminal
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, f"PDF ingestion failed: {str(e)}")

@app.post("/ingest-image")
async def ingest_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only images allowed")

    try:
        contents = await file.read()
        # OCR with EasyOCR
        ocr_result = reader.readtext(contents, detail=0)
        text = " ".join(ocr_result)

        # Embed locally
        vector = embedder.encode(text).tolist()

        index.upsert([{
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {"text": text, "source": file.filename, "type": "image"}
        }])

        return {"status": "success", "extracted_text": text[:500]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Image ingestion failed: {str(e)}")

@app.post("/query")
async def query_rag(req: QueryRequest):
    try:
        # Embed query locally with sentence-transformers
        query_vector = embedder.encode(req.query).tolist()

        # Retrieve from Pinecone
        results = index.query(vector=query_vector, top_k=req.top_k, include_metadata=True)
        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])

        # Define web_search tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for real-time or additional information if the answer is not fully in the provided context (e.g., current events, definitions, recent data).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query to send to the web"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful textbook assistant for students. "
                    "Answer using the provided context from PDFs/images. "
                    "Be accurate, concise, and cite sources. "
                    "If the context is insufficient or the question asks for current information, use the web_search tool exactly once. "
                    "Do NOT say you will search — just call the tool if needed."
                )
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"}
        ]

        # First LLM call – may trigger tool
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=500
        )

        message = response.choices[0].message

        # Handle tool call
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.function.name == "web_search":
                try:
                    args = json.loads(tool_call.function.arguments)
                    search_query = args.get("query", req.query)  # fallback

                    # Execute real web search
                    with DDGS() as ddgs:
                        web_results = [r["body"] for r in ddgs.text(search_query, max_results=3) if r.get("body")]

                    web_context = "\n".join(web_results) if web_results else "No relevant web results found."

                    # Add tool response to messages
                    messages.append({
                        "role": "tool",
                        "content": web_context,
                        "tool_call_id": tool_call.id
                    })

                    # Second LLM call with tool result
                    final_response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500
                    )

                    answer = final_response.choices[0].message.content
                    used_tool = True
                except Exception as tool_err:
                    traceback.print_exc()
                    answer = f"Tool call failed: {str(tool_err)}. Falling back to context only."
                    used_tool = False
            else:
                answer = message.content
                used_tool = False
        else:
            answer = message.content
            used_tool = False

        return {
            "answer": answer.strip(),
            "sources": [m["metadata"] for m in results["matches"]],
            "used_tool": used_tool
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)