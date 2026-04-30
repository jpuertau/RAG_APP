import os
from fastapi import FastAPI
from mcp.server import Server
# FIX: Importación correcta para el servidor SSE en las versiones actuales
from mcp.server.sse import SseServerTransport 

# --- INICIALIZACIÓN ---
app = FastAPI(title="JohaDoorUri RAG Server")
server = Server("johadooruri-brain")

# Reemplazamos FastApiSseServer por el transportador estándar
sse = SseServerTransport("/mcp/sse")

# --- DEFINICIÓN DE HERRAMIENTAS (TOOLS) ---
@server.list_tools()
async def handle_list_tools():
    return [
        {
            "name": "consultar_cerebro",
            "description": "Busca información técnica o personal en la base de datos de memoria local.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Lo que deseas buscar"}
                },
                "required": ["query"]
            }
        }
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "consultar_cerebro":
        query = arguments.get("query")
        return [{"type": "text", "text": f"Buscando en Supabase: '{query}'..."}]
    raise ValueError(f"Tool no encontrada: {name}")

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "online", "message": "JohaDoorUri RAG active"}

# Endpoint para establecer la conexión SSE
@app.get("/mcp/sse")
async def handle_sse(scope, receive, send):
    async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
        await server.handle_sse(read_stream, write_stream, sse.extra_context())

# Endpoint para recibir los mensajes del protocolo
@app.post("/mcp/messages")
async def handle_messages(scope, receive, send):
    await sse.handle_post_messages(scope, receive, send)