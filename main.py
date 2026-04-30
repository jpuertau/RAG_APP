import os
import sys
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from mcp.server.fastapi import FastApiSseServer
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import uvicorn

# --- CONFIGURACIÓN DE HERRAMIENTAS MCP ---
# Definimos el servidor MCP con el nombre que configuraste
server = Server("johadooruri-brain")
sse = FastApiSseServer()

app = FastAPI(title="JohaDoorUri RAG Server")

# --- DEFINICIÓN DE TOOLS ---
@server.list_tools()
async def handle_list_tools():
    """Lista las herramientas disponibles para Claude"""
    return [
        Tool(
            name="aprender_dato",
            description="Guarda información nueva en el cerebro de largo plazo (Supabase)",
            inputSchema={
                "type": "object",
                "properties": {
                    "dato": {"type": "string", "description": "La información a recordar"},
                    "contexto": {"type": "string", "description": "Categoría o contexto del dato"}
                },
                "required": ["dato"]
            }
        ),
        Tool(
            name="consultar_cerebro",
            description="Busca información guardada previamente en la memoria",
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
    """Maneja la ejecución de las herramientas"""
    if name == "aprender_dato":
        dato = arguments.get("dato")
        # Aquí iría tu lógica de inserción en Supabase
        print(f"Aprendiendo: {dato}") 
        return [TextContent(type="text", text=f"Entendido, he guardado: '{dato}' en tu memoria.")]

    if name == "consultar_cerebro":
        query = arguments.get("query")
        # Aquí iría tu lógica de búsqueda RAG
        return [TextContent(type="text", text=f"Buscando en la base de datos vectores para: '{query}'...")]
    
    raise ValueError(f"Herramienta no encontrada: {name}")

# --- ENDPOINTS DE CONEXIÓN ---

@app.get("/")
async def root():
    return {"status": "online", "message": "JohaDoorUri RAG Server is active"}

@app.get("/mcp/sse")
async def handle_sse(request: Request):
    """
    Endpoint crítico para la conexión con Claude.
    FIX: Usamos request._send para acceder a la interfaz ASGI directamente.
    """
    try:
        async with sse.connect_sse(
            request.scope, 
            request.receive, 
            request._send # <--- Aquí está el fix para el error 500
        ) as (read_stream, write_stream):
            await server.handle_sse(read_stream, write_stream, sse.extra_context())
    except Exception as e:
        print(f"Error en el transporte SSE: {e}")
        # No levantamos HTTPException aquí porque el stream ya podría haber iniciado

@app.post("/mcp/messages")
async def handle_messages(request: Request):
    """Maneja los mensajes del protocolo MCP"""
    return await sse.handle_post_messages(request.scope, request.receive, request._send)

# --- INICIO DEL SERVIDOR ---
if __name__ == "__main__":
    # Render asigna dinámicamente el puerto mediante la variable de entorno PORT
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)