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
    logger.info("Payload index on doc_id ensured.")
except Exception as e:
    # Index already exists — not an error
    logger.info("Payload index already exists or skipped: %s", e)

# --- Pydantic Models ---

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    user_message: str
    active_doc_ids: Optional[List[str]] = None

# --- LangGraph Tools ---
# doc_ids and doc_catalogue are injected per-request via closure.

def get_doc_catalogue(allowed_ids: Optional[List[str]]) -> List[dict]:
    """Build a lightweight {id, name} catalogue from Qdrant for the given doc IDs."""
    catalogue: dict = {}
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
            if allowed_ids and doc_id not in allowed_ids:
                continue
            if doc_id not in catalogue:
                catalogue[str(doc_id)] = payload.get("company", "Unknown")
        if next_offset is None:
            break
        offset = next_offset
    return [{"id": k, "name": v} for k, v in catalogue.items()]


def make_tools(doc_ids: Optional[List[str]]):

    # Build catalogue once per request
    catalogue = get_doc_catalogue(doc_ids)

    @tool
    def select_documents(query: str) -> str:
        """
        Given the user query, decide which documents are relevant to answer it.
        Always call this FIRST before retrieve_chunks.
        Returns a JSON list of selected doc_ids to search.
        """
        if not catalogue:
            return json.dumps([])

        # If only one doc, skip the LLM call
        if len(catalogue) == 1:
            return json.dumps([catalogue[0]["id"]])

        selector_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=256,
        )

        doc_list = "\n".join(f'- id: {d["id"]}  name: {d["name"]}' for d in catalogue)
        prompt = (
            "You are a document routing assistant.\n"
            "Given the user query and the list of available documents below, "
            "return ONLY a JSON array of the document IDs that are relevant to answer the query.\n"
            "If the query spans multiple documents (e.g. a comparison), include all relevant ones.\n"
            "If unsure, include all.\n"
            "Output ONLY the JSON array, no explanation.\n\n"
            f"Available documents:\n{doc_list}\n\n"
            f"User query: {query}"
        )

        response = selector_llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        selected = json.loads(raw)
        # Ensure strings, fallback to all if empty
        if not isinstance(selected, list) or len(selected) == 0:
            selected = [d["id"] for d in catalogue]
        else:
            selected = [str(i) for i in selected]

        logger.info("Document routing: selected %s from %s", selected, [d["id"] for d in catalogue])
        return json.dumps(selected)

    @tool
    def retrieve_chunks(query: str, doc_ids_json: str) -> str:
        """
        Search the vector database for chunks relevant to the query,
        scoped only to the documents selected by select_documents.
        query: the user's question
        doc_ids_json: JSON array of doc_ids returned by select_documents
        Returns retrieved text passages with source and page number.
        """
        try:
            selected_ids = json.loads(doc_ids_json)
            # Ensure all IDs are strings — LLM may return bare integers
            selected_ids = [str(i) for i in selected_ids]
        except Exception:
            selected_ids = [str(i) for i in (doc_ids or [])]

        query_vector = embedder.encode(query).tolist()

        query_filter = None
        if selected_ids:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=selected_ids))]
            )

        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=15,
        )

        if not response.points:
            return "No relevant chunks found in the selected documents."

        chunks = [
            f"--- Source: {hit.payload['company']} (Page {hit.payload.get('page_num', 'N/A')}) ---\n{hit.payload['text']}"
            for hit in response.points
        ]
        return "\n\n".join(chunks)

    @tool
    def generate_chart(context: str, user_request: str) -> str:
        """
        Given raw retrieved text context and the user's chart request, uses an LLM
        to extract the relevant numbers and produce a chart definition.
        Call this when the user asks to plot, visualise, or chart any data.
        context: the text passages returned by retrieve_chunks
        user_request: what the user wants to chart, e.g. 'revenue and net income by year'
        Returns a ```chart JSON block the frontend renders directly.
        """
        chart_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=1000,
        )

        prompt = (
            "You are a data extraction assistant. Given the financial document context below "
            "and the user's chart request, extract the relevant numbers and output ONLY a single "
            "JSON object — no explanation, no markdown, no extra text.\n\n"
            "The JSON must follow this exact schema:\n"
            "{\n"
            '  "type": "bar" | "line" | "pie" | "doughnut",\n'
            '  "title": "string",\n'
            '  "labels": ["string", ...],\n'
            '  "datasets": [\n'
            '    { "label": "string", "data": [number, ...], "color": "#hexcolor" }\n'
            "  ]\n"
            "}\n\n"
            f"User request: {user_request}\n\n"
            f"Context:\n{context}"
        )

        response = chart_llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip accidental markdown fences if the LLM adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Validate it's real JSON before returning
        chart_def = json.loads(raw)
        return f"```chart\n{json.dumps(chart_def, indent=2)}\n```"

    return [select_documents, retrieve_chunks, generate_chart]


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
            "For every question, follow this exact tool order:\n"
            "1. Call select_documents to identify which documents are relevant.\n"
            "2. Call retrieve_chunks with the query and the doc_ids returned by select_documents.\n"
            "3. Answer from the retrieved context. If the user asked for a chart, also call generate_chart.\n"
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

        # Extract tool calls from the message trace
        tools_used = []
        for msg in result["messages"]:
            # ToolMessage has a name attribute
            if hasattr(msg, "name") and msg.name:
                if msg.name not in tools_used:
                    tools_used.append(msg.name)

        # Last message in the graph output is the final answer
        final = result["messages"][-1].content
        return {"response": final, "tools_used": tools_used}

    except Exception as e:
        logger.error("Chat error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
