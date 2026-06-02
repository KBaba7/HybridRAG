from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from liteparse import LiteParse
import uuid
import os
import logging
import traceback
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="10-K Multi-Agent RAG API")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Retrieve credentials securely
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX", "/opt/homebrew/share/tessdata")

os.environ.setdefault("TESSDATA_PREFIX", TESSDATA_PREFIX)

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in the .env file")

# Initialize remote Qdrant Client
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "10k_filings"

# Initialize HuggingFace Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create Qdrant collection securely (checks if it exists first)
if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="doc_id",
    field_schema=PayloadSchemaType.KEYWORD,
)

# Initialize the local LiteParse engine
parser = LiteParse(ocr_enabled=True)

# --- Pydantic Models ---

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    user_message: str
    active_doc_ids: Optional[List[str]] = None

# --- Helper Functions ---

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Basic character-level chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def is_table_query(text: str) -> bool:
    table_terms = ("table", "balance sheet", "income statement", "cash flow", "extract")
    lowered = text.lower()
    return any(term in lowered for term in table_terms)


def table_context_chunks(query: str, query_filter: Optional[Filter], limit: int = 35) -> List[str]:
    """Fetch extra same-document chunks for table questions, where vector search often returns only the title."""
    if not query_filter:
        return []

    records, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=query_filter,
        limit=300,
        with_payload=True,
        with_vectors=False,
    )

    query_terms = {
        term for term in query.lower().replace("-", " ").split()
        if len(term) > 4
    }
    finance_terms = {
        "assets", "liabilities", "equity", "cash", "securities", "receivables",
        "inventory", "property", "debt", "payables", "shareholders", "stockholders",
        "balance", "sheet", "consolidated", "current", "non-current", "retained",
    }
    match_terms = query_terms | finance_terms

    matches = []
    for record in records:
        payload = record.payload or {}
        text = payload.get("text", "")
        lowered = text.lower()
        if any(term in lowered for term in match_terms):
            matches.append(payload)

    matches.sort(key=lambda item: (item.get("page_num", 0), item.get("chunk_idx", 0)))
    return [
        f"--- Source: {item.get('company', 'Unknown')} (Page {item.get('page_num', 'N/A')}) ---\n{item.get('text', '')}"
        for item in matches[:limit]
    ]

# --- Endpoints ---

@app.get("/api/documents")
async def list_documents():
    """
    Scans Qdrant for all distinct documents and returns their metadata
    (doc_id, company name, page count, table count) so the frontend
    can restore its state without re-uploading.
    """
    try:
        docs: dict = {}
        offset = None

        while True:
            records, next_offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=None,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                payload = record.payload or {}
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue

                if doc_id not in docs:
                    docs[doc_id] = {
                        "id": doc_id,
                        "name": payload.get("company", "Unknown"),
                        "pages": set(),
                        "tables": 0,
                    }

                docs[doc_id]["pages"].add(payload.get("page_num", 0))
                if payload.get("has_table"):
                    docs[doc_id]["tables"] += 1

            if next_offset is None:
                break
            offset = next_offset

        result = [
            {
                "id": d["id"],
                "name": d["name"],
                "pages": len(d["pages"]),
                "tableCount": d["tables"],
            }
            for d in docs.values()
        ]

        return {"documents": result}

    except Exception as e:
        logger.error("List documents error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...), doc_id: str = Form(...), company_name: str = Form(...)):
    """
    Parses the PDF locally using LiteParse, chunks the spatially-aware text,
    embeds it, and stores it in Qdrant.
    """
    try:
        file_bytes = await file.read()
        parsed_result = parser.parse(file_bytes)
        
        points = []
        total_tables = 0
        
        for page in parsed_result.pages:
            page_text = page.text
            
            # Simple heuristic for table detection in spatially parsed text
            has_table = "  " in page_text and "\n" in page_text 
            if has_table:
                total_tables += 1

            chunks = chunk_text(page_text)
            embeddings = embedder.encode(chunks)
            
            for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={
                            "doc_id": doc_id,
                            "company": company_name,
                            "page_num": page.page_num,
                            "chunk_idx": chunk_idx,
                            "text": chunk,
                            "has_table": has_table
                        }
                    )
                )
                
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        return {
            "status": "success", 
            "pages_processed": len(parsed_result.pages),
            "tables_detected": total_tables
        }
    
    except Exception as e:
        logger.error("Ingest error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Retrieves relevant chunks and generates an analytical response."""
    try:
        # Embed the user's question
        query_vector = embedder.encode(request.user_message).tolist()

        query_filter = None
        if request.active_doc_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=request.active_doc_ids),
                    )
                ]
            )
        
        # Retrieve top chunks from Qdrant
        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=15,
        )
        search_results = search_response.points
        
        # Format the retrieved context (preserving the spatial layout)
        context_chunks = [
            f"--- Source: {hit.payload['company']} (Page {hit.payload.get('page_num', 'N/A')}) ---\n{hit.payload['text']}" 
            for hit in search_results
        ]
        if is_table_query(request.user_message):
            context_chunks.extend(table_context_chunks(request.user_message, query_filter))
        retrieved_context = "\n\n".join(dict.fromkeys(context_chunks))
        
        client = Groq(api_key=GROQ_API_KEY)
        
        system_prompt = f"""You are an expert multi-document financial analyst.
        Use ONLY the retrieved context below to answer the user's query. The text preserves spatial layout, meaning tables are formatted using spacing. Read them carefully by aligning the columns.

        Your capabilities:
        1. FINANCIAL ANALYSIS: Extract and analyze metrics. If comparing, always cite the source company.
        2. TABLE EXTRACTION: When the user asks for a table, output the complete table from the retrieved context as HTML <table class="msg-table"> with <thead> and <tbody>. Use class="num" for numeric cells, "pos" for positive, "neg" for negative. Do not return only the table title; if rows are missing from the retrieved context, say which rows are missing.
        3. CHART GENERATION: When asked to plot or chart, respond EXACTLY with this JSON block format:
        ```chart
        {{
            "type": "bar|line|pie|doughnut",
            "title": "Chart Title",
            "labels": ["2021", "2022", "2023"],
            "datasets": [
            {{ "label": "Metric", "data": [1, 2, 3], "color": "#4f8ef7" }}
            ]
        }}
        ```
        RETRIEVED CONTEXT:
        {retrieved_context}
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": msg.role, "content": msg.content} for msg in request.messages])
        messages.append({"role": "user", "content": request.user_message})

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=3000
        )
        
        return {"response": response.choices[0].message.content}

    except Exception as e:
        logger.error("Chat error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
