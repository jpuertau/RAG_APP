import os
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from mcp.server import Server
from mcp.server.sse import SseServerTransport

from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import groq

app = FastAPI(title="JohaDoorUri Hybrid RAG-MCP")
server = Server("johadooruri-brain")
sse = SseServerTransport("/mcp/sse")

# Carga de componentes
try:
    supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    client_groq = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
    # Asegúrate de que este modelo genere 384 dimensiones para que coincida con tu SQL
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error de inicialización: {e}")

async def ejecutar_ingesta_logic(text: str):
    embedding = embed_model.encode(text).tolist()
    data = {"content": text, "embedding": embedding, "metadata": {"source": "api"}}
    supabase.table("documents").insert(data).execute()
    return True

async def ejecutar_rag_logic(query: str):
    try:
        query_embedding = embed_model.encode(query).tolist()
        
        # AJUSTE CRÍTICO: Bajamos el threshold a 0.3 para ser más permisivos en la búsqueda
        rpc_params = {
            "query_embedding": query_embedding,
            "match_threshold": 0.3, 
            "match_count": 5
        }
        
        response = supabase.rpc("match_documents", rpc_params).execute()
        
        if not response.data:
            return "Lo siento, no encontré información específica en mi base de datos sobre eso."

        # Construcción del contexto con lo que SI encontró
        contexto = "\n".join([f"- {doc['content']}" for doc in response.data])
        
        # Prompt mejorado para obligar a la IA a usar el contexto
        prompt = (
            f"Eres el cerebro digital de Joha. Responde basándote estrictamente en este contexto:\n"
            f"{contexto}\n\n"
            f"Pregunta del usuario: {query}\n"
            f"Respuesta:"
        )
        
        # Modelo actualizado para evitar el error de depreciación
        completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2 # Menor temperatura = respuesta más fiel al contexto
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error en la búsqueda: {str(e)}"

# --- ENDPOINTS ---

@app.get("/ask")
@app.post("/ask")
async def api_ask(request: Request, query: Optional[str] = None):
    if request.method == "POST":
        body = await request.json()
        query = body.get("query")
    if not query:
        return {"status": "error", "message": "Falta la consulta"}
    res = await ejecutar_rag_logic(query)
    return {"status": "success", "response": res}

@app.get("/ingest")
async def api_ingest(text: str):
    await ejecutar_ingesta_logic(text)
    return {"status": "success", "message": "Dato guardado"}

# --- MCP ENDPOINTS ---
@app.get("/mcp/sse")
async def handle_sse(scope, receive, send):
    async with sse.connect_sse(scope, receive, send) as (read, write):
        await server.handle_sse(read, write, sse.extra_context())

@app.post("/mcp/messages")
async def handle_messages(scope, receive, send):
    await sse.handle_post_messages(scope, receive, send)