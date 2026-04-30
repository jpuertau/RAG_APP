import os
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq
from mcp.server.fastapi import FastApiServer
from mcp.server import Server
from mcp.types import Tool, TextContent

# --- CONFIGURACIÓN ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
model = SentenceTransformer('paraphrase-albert-small-v2', device='cpu', trust_remote_code=True)

# --- INICIALIZACIÓN MCP ---
mcp_server = Server("johadooruri-brain")
app = FastAPI(title="JohaDoorUri RAG + MCP")

# --- LÓGICA CORE (Reutilizable) ---

async def run_ingest(text: str):
    vector = model.encode(text).tolist()
    supabase.table("documents").insert({
        "content": text,
        "embedding": vector,
        "metadata": {"source": "mcp_ingest"}
    }).execute()
    return f"Información indexada: {text[:50]}..."

async def run_ask(question: str):
    q_vector = model.encode(question).tolist()
    rpc_res = supabase.rpc("match_documents", {
        "query_embedding": q_vector,
        "match_threshold": 0.35,
        "match_count": 3
    }).execute()
    
    context = "\n".join([d['content'] for d in rpc_res.data]) if rpc_res.data else "Sin contexto."
    prompt = f"Contexto: {context}\n\nPregunta: {question}\n\nRespuesta técnica:"
    
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# --- REGISTRO DE HERRAMIENTAS MCP ---

@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="aprender_dato",
            description="Guarda nueva información técnica en la memoria a largo plazo.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        Tool(
            name="consultar_cerebro",
            description="Busca en la base de datos de conocimientos para responder preguntas.",
            inputSchema={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        ),
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "aprender_dato":
        msg = await run_ingest(arguments["text"])
        return [TextContent(type="text", text=msg)]
    elif name == "consultar_cerebro":
        answer = await run_ask(arguments["question"])
        return [TextContent(type="text", text=answer)]
    raise ValueError(f"Herramienta no encontrada: {name}")

# --- ENDPOINTS HTTP Y MCP ---

@app.get("/health")
def health(): return {"status": "ok"}

# Rutas URL tradicionales (para que sigas usando tus links)
@app.get("/ingest")
async def ingest_endpoint(text: str):
    res = await run_ingest(text)
    return {"status": "success", "detail": res}

@app.get("/ask")
async def ask_endpoint(question: str):
    res = await run_ask(question)
    return {"answer": res}

# Montar el transporte MCP (Server-Sent Events)
mcp_app = FastApiServer(mcp_server)
app.mount("/mcp", mcp_app)