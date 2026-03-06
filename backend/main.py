from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
import uuid
import fitz  # PyMuPDF
import easyocr
import traceback
import json
import base64
import requests

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

def get_embedding(text: str):
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        json={"inputs": [text]},
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"HuggingFace embedding error: {response.text}")

    embedding = response.json()

    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]

    return embedding

def should_force_web_search(query: str, has_context: bool) -> bool:
    q = query.lower().strip()

    current_info_triggers = [
        "latest", "current", "today", "recent", "news", "now",
        "live", "weather", "stock", "price", "update", "who is",
        "what is happening", "search", "web search", "look up"
    ]

    asks_external = any(trigger in q for trigger in current_info_triggers)

    return (not has_context) or asks_external


def describe_image_with_groq(image_bytes: bytes, mime_type: str) -> str:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image clearly and concisely. "
                            "If it contains text, include the readable text. "
                            "If it does not contain text, describe the visible objects, scene, and subject."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.3,
        max_completion_tokens=300
    )

    content = response.choices[0].message.content
    return content.strip() if content else "Image uploaded, but no description could be generated."

app = FastAPI(
    title="EdTech RAG Textbook Reader API",
    description="Multimodal RAG for PDF & Image Textbook Queries with Groq + Pinecone",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "edtech-rag-index")

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# Lazy-load OCR for lower memory usage
reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'])
    return reader

SIMILARITY_THRESHOLD = 0.75

CASUAL_RESPONSES = {
    "hi": "Hi! How are you? You can upload a PDF or image and ask me questions about it.",
    "hello": "Hello! How are you? Upload a PDF or image and I’ll help you with it.",
    "hey": "Hey! How are you? Feel free to upload a PDF or image and ask me anything from it.",
    "how are you": "I’m doing well, thank you! Upload a PDF or image whenever you're ready, and I’ll help you understand it.",
    "good morning": "Good morning! Upload a PDF or image and ask me anything from the content.",
    "good afternoon": "Good afternoon! Upload a PDF or image and ask me anything from it.",
    "good evening": "Good evening! Upload a PDF or image and I’ll help you with your questions.",
    "thanks": "You’re welcome! Upload a PDF or image anytime if you want help with the content.",
    "thank you": "You’re welcome! I’m here whenever you need help with a PDF, image, or study question.",
    "bye": "Goodbye! Come back anytime if you want help with your study material.",
}

# In-memory latest upload context
LATEST_CONTEXT = {
    "type": None,      # "image" or "pdf"
    "source": None,
    "text": None
}

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


def looks_like_image_followup(query: str) -> bool:
    q = query.lower().strip()
    triggers = [
        "this image",
        "that image",
        "uploaded image",
        "the image",
        "image says",
        "image mean",
        "explain this image",
        "summarize this image",
        "what is in this image",
        "what does this image say",
        "what happens next",
        "what happens after",
        "continue this",
        "explain the passage",
        "summarize the passage",
        "what is written here",
        "what does it mean",
        "explain about image",
        "about the image"
    ]
    return any(t in q for t in triggers)


def build_latest_context_text():
    if not LATEST_CONTEXT.get("text"):
        return ""

    source = LATEST_CONTEXT.get("source", "uploaded file")
    ctype = LATEST_CONTEXT.get("type", "file")
    return f"Latest uploaded {ctype} ({source}):\n{LATEST_CONTEXT['text']}"


@app.get("/health")
async def health():
    return {"status": "healthy", "message": "EdTech RAG API is live"}


@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    temp_path = f"temp_{uuid.uuid4()}.pdf"
    contents = await file.read()

    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        doc = fitz.open(temp_path)
        chunks = []
        combined_preview = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if not text:
                continue

            chunk = f"Page {page_num}: {text[:1000]}"
            vector = get_embedding(chunk)

            chunks.append({
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {
                    "text": chunk,
                    "source": file.filename,
                    "page": page_num,
                    "type": "pdf"
                }
            })

            if len(combined_preview) < 3:
                combined_preview.append(chunk)

        doc.close()

        if chunks:
            index.upsert(vectors=chunks)

        LATEST_CONTEXT["type"] = "pdf"
        LATEST_CONTEXT["source"] = file.filename
        LATEST_CONTEXT["text"] = "\n".join(combined_preview)

        os.remove(temp_path)

        return {
            "status": "success",
            "message": f"Ingested {file.filename} with {len(chunks)} chunks"
        }

    except Exception as e:
        traceback.print_exc()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {str(e)}")


@app.post("/ingest-image")
async def ingest_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only images allowed")

    try:
        contents = await file.read()

        extracted_text = ""
        try:
            ocr_result = get_reader().readtext(contents, detail=0)
            extracted_text = " ".join(ocr_result).strip()
        except Exception:
            extracted_text = ""

        # If OCR text exists, use it. Otherwise use vision description.
        if extracted_text:
            image_text = extracted_text
        else:
            image_text = describe_image_with_groq(contents, file.content_type)

        if not image_text:
            raise HTTPException(status_code=400, detail="Could not extract or describe the image")

        vector = get_embedding(image_text)

        index.upsert(vectors=[{
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                "text": image_text,
                "source": file.filename,
                "type": "image"
            }
        }])

        LATEST_CONTEXT["type"] = "image"
        LATEST_CONTEXT["source"] = file.filename
        LATEST_CONTEXT["text"] = image_text

        return {
            "status": "success",
            "extracted_text": image_text[:500]
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image ingestion failed: {str(e)}")

@app.post("/query")
async def query_rag(req: QueryRequest):
    try:
        user_query = req.query.strip()
        normalized_query = user_query.lower()

        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if normalized_query in CASUAL_RESPONSES:
            return {
                "answer": CASUAL_RESPONSES[normalized_query],
                "sources": [],
                "used_tool": False
            }

        query_vector = get_embedding(user_query)

        results = index.query(
            vector=query_vector,
            top_k=req.top_k,
            include_metadata=True
        )

        matches = results.get("matches", [])

        filtered_matches = [
            match for match in matches
            if match.get("score", 0) >= SIMILARITY_THRESHOLD and match.get("metadata")
        ]

        rag_context = "\n".join([
            match["metadata"]["text"]
            for match in filtered_matches
            if match.get("metadata", {}).get("text")
        ])

        latest_context = build_latest_context_text()
        prioritize_latest = looks_like_image_followup(user_query)

        if prioritize_latest and latest_context:
            context = f"{latest_context}\n\nRelevant retrieved context:\n{rag_context}".strip()
        else:
            context = rag_context if rag_context.strip() else latest_context

        if not context.strip():
            context = "No relevant document context found."

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web for real-time or additional information if the answer "
                        "is not fully available in the provided context."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to send to the web"
                            }
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
                    "Answer using the provided context from uploaded PDFs and images. "
                    "If the provided context is insufficient, you MUST call the web_search tool "
                    "to retrieve information from the internet before answering. "
                    "Do NOT guess answers when context is missing."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_query}"
            }
        ]

        used_tool = False

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.5,
                max_tokens=500
            )

            message = response.choices[0].message

            if message.tool_calls:
                tool_call = message.tool_calls[0]

                if tool_call.function.name == "web_search":
                    args = json.loads(tool_call.function.arguments)
                    search_query = args.get("query", user_query)

                    web_context = "No relevant web results found."

                    try:
                        effective_query = f"current {search_query}"

                        with DDGS() as ddgs:
                            raw_results = list(ddgs.text(effective_query, max_results=5))

                        web_results = []
                        for r in raw_results:
                            title = r.get("title", "")
                            body = r.get("body", "")
                            href = r.get("href", "")
                            combined = f"Title: {title}\nSnippet: {body}\nURL: {href}".strip()
                            if title or body:
                                web_results.append(combined)

                        if web_results:
                            web_context = "\n\n".join(web_results)

                    except Exception as search_err:
                        print(f"Web search failed: {search_err}")

                    messages.append(message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": web_context
                    })

                    final_response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        temperature=0.5,
                        max_tokens=500
                    )

                    answer = final_response.choices[0].message.content.strip()
                    used_tool = True
                else:
                    answer = message.content.strip() if message.content else "No response generated."
            else:
                answer = message.content.strip() if message.content else "No response generated."

        except Exception as tool_err:
            err_text = str(tool_err)

            if "tool_use_failed" in err_text or "Failed to call a function" in err_text:
                web_context = "No relevant web results found."

                try:
                    effective_query = f"current {user_query}"

                    with DDGS() as ddgs:
                        raw_results = list(ddgs.text(effective_query, max_results=5))

                    web_results = []
                    for r in raw_results:
                        title = r.get("title", "")
                        body = r.get("body", "")
                        href = r.get("href", "")
                        combined = f"Title: {title}\nSnippet: {body}\nURL: {href}".strip()
                        if title or body:
                            web_results.append(combined)

                    if web_results:
                        web_context = "\n\n".join(web_results)

                except Exception as search_err:
                    print(f"Fallback web search failed: {search_err}")

                fallback_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Answer the user's question using the web search results provided. "
                            "If the answer is present in the search results, state it directly. "
                            "If multiple results agree, prefer the most likely current answer. "
                            "Be concise and accurate."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {user_query}\n\nWeb search results:\n{web_context}"
                    }
                ]

                final_response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=fallback_messages,
                    temperature=0.5,
                    max_tokens=500
                )

                answer = final_response.choices[0].message.content.strip()
                used_tool = True
            else:
                raise tool_err

        cleaned_sources = []
        for match in filtered_matches:
            metadata = match.get("metadata", {})
            cleaned_sources.append({
                "source": metadata.get("source"),
                "text": metadata.get("text"),
                "type": metadata.get("type"),
                "page": metadata.get("page"),
                "score": match.get("score")
            })

        if prioritize_latest and LATEST_CONTEXT.get("source"):
            already_present = any(
                s.get("source") == LATEST_CONTEXT["source"] for s in cleaned_sources
            )
            if not already_present:
                cleaned_sources.insert(0, {
                    "source": LATEST_CONTEXT.get("source"),
                    "text": LATEST_CONTEXT.get("text"),
                    "type": LATEST_CONTEXT.get("type"),
                    "page": None,
                    "score": None
                })

        return {
            "answer": answer,
            "sources": cleaned_sources,
            "used_tool": used_tool
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )