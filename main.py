import os
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- CONFIGURACIÓN DE VARIABLES DE ENTORNO ---
# Estas se deben configurar en el panel de Render -> Environment
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- INICIALIZACIÓN DE CLIENTES ---
# Verificación de seguridad para asegurar que las llaves estén presentes
if not all([SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY]):
    print("CRITICAL ERROR: Faltan variables de entorno en el servidor.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# --- CARGA DEL MODELO OPTIMIZADA PARA RAM < 512MB ---
# Usamos un modelo ALBERT por ser extremadamente eficiente en memoria
try:
    model = SentenceTransformer(
        'paraphrase-albert-small-v2', 
        device='cpu', 
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error fatal al cargar el modelo de embeddings: {e}")

app = FastAPI(title="RAG Service - JohaDoorUri")

@app.get("/")
def home():
    return {
        "message": "RAG Service is Online", 
        "status": "ready",
        "model": "paraphrase-albert-small-v2"
    }

@app.get("/health")
def health():
    """Ruta de verificación para que Render mantenga el servicio activo"""
    return {"status": "ok"}

@app.post("/ingest")
async def ingest_text(text: str):
    """Convierte texto en vectores y lo guarda en la base de datos Supabase"""
    if not text:
        raise HTTPException(status_code=400, detail="El texto está vacío")
    
    try:
        # Generar el vector numérico (Embedding)
        vector = model.encode(text).tolist()
        
        # Guardar en la tabla 'documents' de Supabase
        data = {
            "content": text,
            "embedding": vector,
            "metadata": {"source": "manual_upload"}
        }
        supabase.table("documents").insert(data).execute()
        
        return {"status": "success", "message": "Información indexada correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en ingesta: {str(e)}")

@app.get("/ask")
async def ask_question(question: str):
    """Flujo RAG: Recupera contexto de Supabase y genera respuesta con Groq"""
    if not question:
        raise HTTPException(status_code=400, detail="La pregunta está vacía")
    
    try:
        # 1. RETRIEVAL: Convertir pregunta a vector
        q_vector = model.encode(question).tolist()
        
        # 2. BÚSQUEDA VECTORIAL: Consultar Supabase mediante la función RPC
        rpc_res = supabase.rpc("match_documents", {
            "query_embedding": q_vector,
            "match_threshold": 0.35,
            "match_count": 3
        }).execute()
        
        # Extraer el contenido recuperado
        if not rpc_res.data:
            context = "No hay información disponible en la base de datos."
        else:
            context = "\n".join([d['content'] for d in rpc_res.data])

        # 3. GENERATION: Crear el prompt enriquecido para el LLM (Llama 3)
        prompt = (
            f"Contexto relevante:\n{context}\n\n"
            f"Pregunta del usuario: {question}\n\n"
            f"Instrucción: Responde de forma técnica y profesional basada únicamente en el contexto proporcionado."
        )
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        return {
            "answer": completion.choices[0].message.content,
            "context_used": [d['content'] for d in rpc_res.data] if rpc_res.data else []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en flujo RAG: {str(e)}")