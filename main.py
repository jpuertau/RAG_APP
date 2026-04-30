import os
import asyncio
from typing import Optional
from fastapi import FastAPI, Request, HTTPException

# Componentes MCP
from mcp.server import Server
from mcp.server.sse import SseServerTransport

# Componentes de Lógica RAG
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import groq

# --- INICIALIZACIÓN ---
app = FastAPI(title="JohaDoorUri Hybrid RAG-MCP")
server = Server("johadooruri-brain")
sse = SseServerTransport("/mcp/sse")

# Clientes con validación de existencia de variables
try:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    client_groq = groq.Groq(api_key=GROQ_API_KEY)
    
    # Modelo ligero para no exceder los 512MB de Render
    # all-MiniLM-L6-v2 genera vectores de 384 dimensiones
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"CRITICAL CONFIG ERROR: {str(e)}")

# --- LÓGICA DE NEGOCIO ---

async def ejecutar_ingesta_logic(text: str):
    try:
        embedding = embed_model.encode(text).tolist()
        data = {
            "content": text,
            "embedding": embedding,
            "metadata": {"source": "hybrid_api"}
        }
        supabase.table("documents").insert(data).execute()
        return True
    except Exception as e:
        print(f"INGEST ERROR: {str(e)}")
        raise e

async def ejecutar_rag_logic(query: str):
    try:
        # 1. Vectorizar
        query_embedding = embed_model.encode(query).tolist()
        
        # 2. Búsqueda en Supabase (RPC)
        rpc_params = {
            "query_embedding": query_embedding,
            "match_threshold": 0.5,
            "match_count": 3
        }
        
        response = supabase.rpc("match_documents", rpc_params).execute()
        
        contexto = "No se encontró información relevante."
        if response.data and len(response.data) > 0:
            contexto = "\n".join([doc['content'] for doc in response.data])
        
        # 3. Respuesta con Groq
        prompt = (
            f"Contexto: {contexto}\n\n"
            f"Pregunta: {query}\n"
            f"Respuesta concisa:"
        )
        
        completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"RAG ERROR: {str(e)}")
        return f"Error técnico en el procesamiento: {str(e)}"

# --- HERRAMIENTAS MCP ---

@server.list_tools()
async def handle_list_tools():
    return [
        {
            "name": "consultar_cerebro",
            "description": "Busca en la memoria técnica/personal del usuario.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta de búsqueda"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "ingestar_dato",
            "description": "Guarda nueva información en la base de datos.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Dato a memorizar"}
                },
                "required": ["text"]
            }
        }
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "consultar_cerebro":
        ans = await ejecutar_rag_logic(arguments.get("query"))
        return [{"type": "text", "text": ans}]
    if name == "ingestar_dato":
        await ejecutar_ingesta_logic(arguments.get("text"))
        return [{"type": "text", "text": "Información guardada exitosamente."}]
    raise ValueError(f"Tool {name} no reconocida.")

# --- ENDPOINTS API ---

@app.get("/ask")
@app.post("/ask")
async def api_ask(request: Request, query: Optional[str] = None):
    if request.method == "POST":
        body = await request.json()
        query = body.get("query")
    
    if not query:
        return {"status": "error", "message": "Query parameter is missing"}
        
    respuesta = await ejecutar_rag_logic(query)
    return {"status": "success", "response": respuesta}

@app.get("/ingest")
@app.post("/ingest")
async def api_ingest(request: Request, text: Optional[str] = None):
    if request.method == "POST":
        body = await request.json()
        text = body.get("text")
        
    if not text:
        return {"status": "error", "message": "Text parameter is missing"}
        
    await ejecutar_ingesta_logic(text)
    return {"status": "success", "message": "Data ingested and vectorized"}

# --- ENDPOINTS MCP (ASGI) ---

@app.get("/")
async def status():
    return {"status": "online", "engine": "Hybrid RAG-MCP"}

@app.get("/mcp/sse")
async def handle_sse(scope, receive, send):
    async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
        await server.handle_sse(read_stream, write_stream, sse.extra_context())

@app.post("/mcp/messages")
async def handle_messages(scope, receive, send):
    await sse.handle_post_messages(scope, receive, send)