import os
import asyncio
from fastapi import FastAPI, HTTPException, Request
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq

# Imports actualizados para MCP 2026
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

# --- CONFIGURACIÓN DE VARIABLES DE ENTORNO ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Inicialización de Clientes
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Modelo optimizado para RAM < 512MB
model = SentenceTransformer(
    'paraphrase-albert-small-v2', 
    device='cpu', 
    trust_remote_code=True
)

# --- INICIALIZACIÓN MCP ---
mcp_server = Server("johadooruri-brain")
sse = SseServerTransport("/mcp/messages")
app = FastAPI(title="JohaDoorUri RAG + MCP Server")

# --- LÓGICA CORE REUTILIZABLE ---

async def run_ingest(text: str):
    """Proceso de Ingesta: Texto -> Vector -> Supabase"""
    vector = model.encode(text).tolist()
    supabase.table("documents").insert({
        "content": text,
        "embedding": vector,
        "metadata": {"source": "mcp_engine"}
    }).execute()
    return f"Éxito: Información indexada correctamente."

async def run_ask(question: str):
    """Proceso RAG: Pregunta -> Recuperación -> Groq"""
    q_vector = model.encode(question).tolist()
    
    rpc_res = supabase.rpc("match_documents", {
        "query_embedding": q_vector,
        "match_threshold": 0.35,
        "match_count": 3
    }).execute()
    
    context = "\n".join([d['content'] for d in rpc_res.data]) if rpc_res.data else "Sin datos previos."
    
    prompt = (
        f"Eres un asistente experto. Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n\nRespuesta técnica:"
    )
    
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return completion.choices[0].message.content

# --- CONFIGURACIÓN DE HERRAMIENTAS MCP ---

@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """Define las herramientas disponibles para clientes como Claude Desktop"""
    return [
        Tool(
            name="aprender_dato",
            description="Guarda información técnica nueva en la base de datos de JohaDoorUri.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        Tool(
            name="consultar_cerebro",
            description="Realiza preguntas al conocimiento almacenado en Supabase.",
            inputSchema={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        ),
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Ejecuta la lógica según la herramienta llamada por el cliente MCP"""
    if name == "aprender_dato":
        msg = await run_ingest(arguments["text"])
        return [TextContent(type="text", text=msg)]
    elif name == "consultar_cerebro":
        answer = await run_ask(arguments["question"])
        return [TextContent(type="text", text=answer)]
    raise ValueError(f"Herramienta desconocida: {name}")

# --- ENDPOINTS DE TRANSPORTE MCP (SSE) ---

@app.get("/mcp/sse")
async def handle_sse(request: Request):
    """Establece la conexión de eventos del servidor (SSE)"""
    async with sse.connect_sse(request.scope, request.receive, request.send) as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )

@app.post("/mcp/messages")
async def handle_messages(request: Request):
    """Maneja los mensajes entrantes del cliente MCP"""
    await sse.handle_post_request(request.scope, request.receive, request.send)

# --- ENDPOINTS HTTP TRADICIONALES (Para pruebas rápidas) ---

@app.get("/health")
def health():
    return {"status": "ok", "protocol": "MCP Active"}

@app.get("/ingest")
async def ingest_endpoint(text: str):
    detail = await run_ingest(text)
    return {"status": "success", "detail": detail}

@app.get("/ask")
async def ask_endpoint(question: str):
    answer = await run_ask(question)
    return {"answer": answer}