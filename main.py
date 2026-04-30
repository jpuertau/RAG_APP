import os
from fastapi import FastAPI, Request
from mcp.server.fastapi import FastApiSseServer
from mcp.server import Server
from mcp.types import Tool, TextContent

# --- CONFIGURACIÓN DEL SERVIDOR MCP ---
server = Server("johadooruri-brain")
sse = FastApiSseServer()

app = FastAPI(title="JohaDoorUri RAG Server")

# --- DEFINICIÓN DE HERRAMIENTAS (TOOLS) ---
@server.list_tools()
async def handle_list_tools():
    return [
        Tool(
            name="consultar_cerebro",
            description="Busca información técnica o personal en la base de datos Supabase",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Lo que deseas buscar"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "consultar_cerebro":
        query = arguments.get("query")
        # Aquí se integra tu lógica de Supabase/Embeddings
        return [TextContent(type="text", text=f"Resultado de búsqueda para: '{query}'")]
    raise ValueError(f"Tool no encontrada: {name}")

# --- ENDPOINTS CRÍTICOS ---

@app.get("/")
async def root():
    return {"status": "online", "message": "JohaDoorUri RAG active"}

@app.get("/mcp/sse")
async def handle_sse(request: Request):
    """
    Establece el canal SSE. 
    FIX: Usamos request._send para evitar AttributeError en FastAPI/Starlette.
    """
    async with sse.connect_sse(
        request.scope, 
        request.receive, 
        request._send 
    ) as (read_stream, write_stream):
        await server.handle_sse(read_stream, write_stream, sse.extra_context())

@app.post("/mcp/messages")
async def handle_messages(request: Request):
    """Maneja el flujo de mensajes una vez establecida la conexión SSE."""
    return await sse.handle_post_messages(request.scope, request.receive, request._send)

# NOTA: No incluyas uvicorn.run() aquí para evitar conflictos con Render.