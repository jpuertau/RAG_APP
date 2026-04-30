import os
from fastapi import FastAPI
from mcp.server.fastapi import FastApiSseServer
from mcp.server import Server
from mcp.types import Tool, TextContent

# --- INICIALIZACIÓN ---
# Creamos la app y el servidor MCP de inmediato para abrir el puerto rápido
app = FastAPI(title="JohaDoorUri RAG Server")
server = Server("johadooruri-brain")
sse = FastApiSseServer()

# --- DEFINICIÓN DE HERRAMIENTAS (TOOLS) ---
@server.list_tools()
async def handle_list_tools():
    """Lista las herramientas disponibles."""
    return [
        Tool(
            name="consultar_cerebro",
            description="Busca información técnica o personal en la base de datos de memoria local.",
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
    """Ejecuta la lógica de las herramientas."""
    if name == "consultar_cerebro":
        query = arguments.get("query")
        
        # TIP TÉCNICO: Podrías importar sentence-transformers aquí adentro 
        # para que el servidor no pese tanto al arrancar.
        return [TextContent(type="text", text=f"Buscando en Supabase: '{query}'...")]
        
    raise ValueError(f"Tool no encontrada: {name}")

# --- ENDPOINTS ASGI DIRECTOS ---
# Usamos la firma (scope, receive, send) para evitar errores del objeto Request

@app.get("/")
async def root():
    return {"status": "online", "message": "JohaDoorUri RAG active"}

@app.get("/mcp/sse")
async def handle_sse(scope, receive, send):
    """
    Establece el canal SSE usando la interfaz ASGI cruda.
    Esto elimina el error 'Request object has no attribute send'.
    """
    async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
        await server.handle_sse(read_stream, write_stream, sse.extra_context())

@app.post("/mcp/messages")
async def handle_messages(scope, receive, send):
    """Maneja los mensajes del protocolo MCP via ASGI."""
    return await sse.handle_post_messages(scope, receive, send)

# --- NOTA PARA RENDER ---
# El puerto se maneja mediante el Start Command de Render:
# uvicorn main:app --host 0.0.0.0 --port $PORT