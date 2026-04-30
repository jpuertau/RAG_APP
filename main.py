import os
from fastapi import FastAPI, Request, HTTPException, Query
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from typing import Optional

# --- INICIALIZACIÓN ---
app = FastAPI(title="JohaDoorUri Hybrid RAG-MCP Server")
server = Server("johadooruri-brain")
sse = SseServerTransport("/mcp/sse")

# --- LÓGICA DE HERRAMIENTAS MCP ---

@server.list_tools()
async def handle_list_tools():
    """Define las herramientas que Claude verá en el martillo."""
    return [
        {
            "name": "consultar_cerebro",
            "description": "Busca información técnica o personal en la base de datos de memoria.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Lo que deseas buscar"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "ingestar_dato",
            "description": "Guarda un nuevo conocimiento en la base de datos vectorial.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "El texto a memorizar"}
                },
                "required": ["text"]
            }
        }
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    """Manejador interno para las llamadas de Claude."""
    if name == "consultar_cerebro":
        result = await ejecutar_rag_logic(arguments.get("query"))
        return [{"type": "text", "text": f"Resultado: {result}"}]
    
    if name == "ingestar_dato":
        await ejecutar_ingesta_logic(arguments.get("text"))
        return [{"type": "text", "text": "Dato guardado exitosamente en el cerebro digital."}]
    
    raise ValueError(f"Herramienta no encontrada: {name}")

# --- LÓGICA DE NEGOCIO (RAG & INGESTA) ---

async def ejecutar_rag_logic(query: str):
    # Aquí es donde usas sentence-transformers y supabase.rpc('match_documents')
    return f"Simulación de búsqueda para: '{query}'"

async def ejecutar_ingesta_logic(text: str):
    # Aquí generas el embedding y haces supabase.table('documents').insert(...)
    print(f"Ingestando en Supabase: {text}")
    return True

# --- ENDPOINTS API (GET & POST) ---

@app.get("/ask")
@app.post("/ask")
async def api_ask(request: Request, query: Optional[str] = None):
    """Permite consultas RAG vía URL (GET) o Body (POST)."""
    if request.method == "POST":
        body = await request.json()
        query = body.get("query")
    
    if not query:
        raise HTTPException(status_code=400, detail="Falta el parámetro 'query'")
    
    result = await ejecutar_rag_logic(query)
    return {"status": "success", "response": result}

@app.get("/ingest")
@app.post("/ingest")
async def api_ingest(request: Request, text: Optional[str] = None):
    """Permite ingesta de datos vía URL (GET) o Body (POST)."""
    if request.method == "POST":
        body = await request.json()
        text = body.get("text")
    
    if not text:
        raise HTTPException(status_code=400, detail="Falta el parámetro 'text'")
    
    await ejecutar_ingesta_logic(text)
    return {"status": "success", "message": "Dato procesado e ingestando"}

# --- ENDPOINTS PROTOCOLO MCP (ASGI) ---

@app.get("/")
async def health_check():
    return {"status": "online", "mode": "hybrid", "engine": "FastAPI + MCP"}

@app.get("/mcp/sse")
async def handle_sse(scope, receive, send):
    async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
        await server.handle_sse(read_stream, write_stream, sse.extra_context())

@app.post("/mcp/messages")
async def handle_messages(scope, receive, send):
    await sse.handle_post_messages(scope, receive, send)