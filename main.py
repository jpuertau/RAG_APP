import os
from fastapi import FastAPI
from mcp.server import Server
from mcp.server.sse import SseServerTransport

# --- INICIALIZACIÓN ---
# Abrimos el puerto lo más rápido posible para evitar el status 1 de Render
app = FastAPI(title="JohaDoorUri RAG Server")
server = Server("johadooruri-brain")

# Usamos el transportador estándar de SSE
# El endpoint /mcp/sse es donde el bridge local se conectará
sse = SseServerTransport("/mcp/sse")

# --- DEFINICIÓN DE HERRAMIENTAS (TOOLS) ---
@server.list_tools()
async def handle_list_tools():
    """Lista las herramientas para que Claude las reconozca"""
    return [
        {
            "name": "consultar_cerebro",
            "description": "Busca información técnica o personal en la base de datos de memoria (Supabase)",
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
    """Ejecuta la búsqueda RAG o inserción de datos"""
    if name == "consultar_cerebro":
        query = arguments.get("query")
        # Aquí es donde integrarás supabase y sentence-transformers
        return [{"type": "text", "text": f"Buscando en Supabase: '{query}'..."}]
    raise ValueError(f"Tool no encontrada: {name}")

# --- ENDPOINTS ASGI ---
# IMPORTANTE: Usamos la firma (scope, receive, send) para bypass de errores de FastAPI

@app.get("/")
async def root():
    return {"status": "online", "message": "JohaDoorUri RAG active"}

@app.get("/mcp/sse")
async def handle_sse(scope, receive, send):
    """Maneja la conexión inicial del protocolo MCP"""
    async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
        await server.handle_sse(read_stream, write_stream, sse.extra_context())

@app.post("/mcp/messages")
async def handle_messages(scope, receive, send):
    """Recibe los mensajes de ejecución de herramientas de Claude"""
    await sse.handle_post_messages(scope, receive, send)

# Render maneja el arranque con: uvicorn main:app --host 0.0.0.0 --port $PORT