from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchAny,
    PayloadSchemaType, PointStruct, VectorParams,
)
from sentence_transformers import SentenceTransformer
from liteparse import LiteParse
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import uuid
import os
import json
import logging
import traceback
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="10-K Multi-Agent RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX", "/opt/homebrew/share/tessdata")

os.environ.setdefault("TESSDATA_PREFIX", TESSDATA_PREFIX)

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in .env")

# --- Clients ---
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
parser = LiteParse(ocr_enabled=True)

COLLECTION_NAME = "10k_filings"

if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

try:
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
except Exception:
    pass  # index already exists

# --- Pydantic Models ---

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    user_message: str
    active_doc_ids: Optional[List[str]] = None

# --- LangGraph Tools ---
# doc_ids is injected per-request via closure; tools are stateless functions.

def make_tools(doc_ids: Optional[List[str]]):

    @tool
    def retrieve_chunks(query: str) -> str:
        """
        Search the vector database for document chunks relevant to the query.
        Always call this first to get context before answering any financial question.
        Returns the retrieved text passages with their source and page number.
        """
        query_vector = embedder.encode(query).tolist()

        query_filter = None
        if doc_ids:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
            )

        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=15,
        )

        if not response.points:
            return "No relevant chunks found in the document."

        chunks = [
            f"--- Source: {hit.payload['company']} (Page {hit.payload.get('page_num', 'N/A')}) ---\n{hit.payload['text']}"
            for hit in response.points
        ]
        return "\n\n".join(chunks)

    @tool
    def generate_chart(
        chart_type: str,
        title: str,
        labels: List[str],
        datasets: List[dict],
    ) -> str:
        """
        Generate a chart from structured financial data.
        Call this when the user asks to plot, visualize, or chart data.
        chart_type: one of bar, line, pie, doughnut
        labels: x-axis labels e.g. ["2021", "2022", "2023"]
        datasets: list of dicts with keys: label (str), data (list of numbers), color (hex string)
        Returns a JSON string the frontend renders as a chart.
        """
        chart_def = {
            "type": chart_type,
            "title": title,
            "labels": labels,
            "datasets": datasets,
        }
        return f"```chart\n{json.dumps(chart_def, indent=2)}\n```"

    return [retrieve_chunks, generate_chart]


# --- Helper ---

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


# --- Endpoints ---

@app.get("/api/documents")
async def list_documents():
    """Return distinct documents stored in Qdrant so the frontend can restore state."""
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

        return {
            "documents": [
                {"id": d["id"], "name": d["name"], "pages": len(d["pages"]), "tableCount": d["tables"]}
                for d in docs.values()
            ]
        }

    except Exception as e:
        logger.error("List documents error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    company_name: str = Form(...),
):
    """Parse PDF with LiteParse, embed chunks, store in Qdrant."""
    try:
        file_bytes = await file.read()
        parsed_result = parser.parse(file_bytes)

        points = []
        total_tables = 0

        for page in parsed_result.pages:
            page_text = page.text
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
                            "has_table": has_table,
                        },
                    )
                )

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return {
            "status": "success",
            "pages_processed": len(parsed_result.pages),
            "tables_detected": total_tables,
        }

    except Exception as e:
        logger.error("Ingest error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    LangGraph ReAct agent with two tools:
      - retrieve_chunks: semantic search over Qdrant
      - generate_chart:  produce chart JSON for the frontend
    The LLM decides which tools to call and in what order.
    """
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=3000,
        )

        tools = make_tools(request.active_doc_ids)
        agent = create_react_agent(llm, tools)

        system_prompt = (
            "You are an expert financial analyst specialising in 10-K filings.\n"
            "Always call retrieve_chunks first to get the relevant document context "
            "before answering any question.\n"
            "When the user asks to plot, visualise, or chart data, call generate_chart "
            "after retrieving the data — never before.\n"
            "For tables, output them as HTML <table class=\"msg-table\"> with <thead> "
            "and <tbody>. Use class=\"num\" for numbers, \"pos\" for positive values, "
            "\"neg\" for negative values.\n"
            "Base all answers strictly on the retrieved context."
        )

        # Build message history
        messages: List[Any] = [{"role": "system", "content": system_prompt}]
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.user_message})

        result = agent.invoke({"messages": messages})

        # Last message in the graph output is the final answer
        final = result["messages"][-1].content
        return {"response": final}

    except Exception as e:
        logger.error("Chat error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
