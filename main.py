import os
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from mcp.server import Server
from mcp.server.sse import SseServerTransport

# Librerías para la lógica de negocio
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import groq

# --- INICIALIZACIÓN DE COMPONENTES ---
app = FastAPI(title="JohaDoorUri Hybrid RAG-MCP Server")
server = Server("johadooruri-brain")
sse = SseServerTransport("/mcp/sse")

# Inicialización de Clientes (Carga perezosa recomendada para entornos serverless)
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

supabase: Client = create_client(supabase_url, supabase_key)
# El modelo all-MiniLM-L6-v2 es ligero (80MB), ideal para el tier gratuito de Render
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client_groq = groq.Groq(api_key=groq_api_key)

# --- LÓGICA DE NEGOCIO (RAG & INGESTA) ---

async def ejecutar_ingesta_logic(text: str):
    """Genera embeddings y guarda en Supabase."""
    embedding = embed_model.encode(text).tolist()
    data = {
        "content": text,
        "embedding": embedding,
        "metadata": {"source": "hybrid_api"}
    }
    supabase.table("documents").insert(data).execute()
    return True

async def ejecutar_rag_logic(query: str):
    """Busca en la DB vectorial y genera respuesta con Groq."""
    # 1. Vectorizar la pregunta
    query_embedding = embed_model.encode(query).tolist()
    
    # 2. Búsqueda por similitud (RPC match_documents)
    # Nota: Debes haber creado la función match_documents en el SQL de Supabase
    rpc_params = {
        "query_embedding": query_embedding,
        "match_threshold": 0.5,
        "match_count": 3
    }
    
    response = supabase.rpc("match_documents", rpc_params).execute()
    
    if not response.data:
        contexto = "No se encontró información relevante en la base de datos."
    else:
        contexto = "\n".join([doc['content'] for doc in response.data])
    
    # 3. Generar respuesta aumentada con Groq
    prompt = (
        f"Eres un asistente experto. Utiliza el siguiente contexto para responder de forma concisa.\n"
        f"Contexto: {contexto}\n\n"
        f"Pregunta: {query}\n"
        f"Respuesta:"
    )
    
    chat_completion = client_groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    
    return chat_completion.choices[0].message.content

# --- CONFIGURACIÓN DE HERRAMIENTAS MCP ---

@server.list_tools()
async def handle_list_tools():
    return [
        {
            "name": "consultar_cerebro",
            "description": "Busca información en tu memoria técnica y personal.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Pregunta o término de búsqueda"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "ingestar_dato",
            "description": "Guarda un nuevo conocimiento en tu cerebro digital.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Información a memorizar"}
                },
                "required": ["text"]
            }
        }
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "consultar_cerebro":
        respuesta = await ejecutar_rag_logic(arguments.get("query"))
        return [{"type": "text", "text": respuesta}]
    
    if name == "ingestar_dato":
        await ejecutar_ingesta_logic(arguments.get("text"))
        return [{"type": "text", "text": "Dato memorizado correctamente."}]
    
    raise ValueError(f"Tool no encontrada: {name}")

# --- ENDPOINTS API (GET & POST) ---

@app.get("/ask")
@app.post("/ask")
async def api_ask(request: Request, query: Optional[str] = None):
    if request.method == "POST":
        body = await request.json()
        query = body.get("query")
    
    if not query:
        raise HTTPException(status_code=400, detail="Falta el parámetro 'query'")
    
    respuesta = await ejecutar_rag_logic(query)
    return {"status": "success", "response": respuesta}

@app.get("/ingest")
@app.post("/ingest")
async def api_ingest(request: Request, text: Optional[str] = None):
    if request.method == "POST":
        body = await request.json()
        text = body.get("text")
    
    if not text:
        raise HTTPException(status_code=400, detail="Falta el parámetro 'text'")
    
    await ejecutar_ingesta_logic(text)
    return {"status": "success", "message": "Dato ingestando y vectorizado"}

# --- ENDPOINTS MCP (INTERFAZ ASGI CRUDA) ---

@app.get("/")
async def health():
    return {"status": "online", "engine": "Hybrid RAG-MCP"}

@app.get("/mcp/sse")
async def handle_sse(scope, receive, send):
    async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
        await server.handle_sse(read_stream, write_stream, sse.extra_context())

@app.post("/mcp/messages")
async def handle_messages(scope, receive, send):
    await sse.handle_post_messages(scope, receive, send)