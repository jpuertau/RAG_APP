import os
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- CONFIGURACIÓN DE VARIABLES DE ENTORNO ---
# Estas deben estar configuradas en el panel de Render -> Environment
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- INICIALIZACIÓN DE CLIENTES ---
# Verificamos que las variables existan para evitar errores silenciosos
if not all([SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY]):
    print("ERROR: Faltan variables de entorno. Revisa el panel de Render.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# --- CARGA DEL MODELO (OPTIMIZADO PARA < 512MB RAM) ---
# Usamos 'paraphrase-albert-small-v2' por su bajísimo consumo de memoria
try:
    model = SentenceTransformer('paraphrase-albert-small-v2', device='cpu')
except Exception as e:
    print(f"Error cargando el modelo: {e}")

app = FastAPI(title="RAG Service - JohaDoorUri")

@app.get("/")
def home():
    return {"message": "RAG Service is Online", "model": "paraphrase-albert-small-v2"}

@app.get("/health")
def health():
    """Ruta necesaria para que Render sepa que la app está viva"""
    return {"status": "ok"}

@app.post("/ingest")
async def ingest_text(text: str):
    """Convierte texto en vectores y lo guarda en Supabase"""
    if not text:
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")
    
    try:
        # Generar el embedding
        vector = model.encode(text).tolist()
        
        # Insertar en Supabase
        data = {
            "content": text,
            "embedding": vector,
            "metadata": {"source": "manual_ingest"}
        }
        supabase.table("documents").insert(data).execute()
        
        return {"status": "success", "message": "Información indexada correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask")
async def ask_question(question: str):
    """Flujo completo de Retrieval-Augmented Generation (RAG)"""
    if not question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    
    try:
        # 1. RETRIEVAL: Convertir pregunta a vector y buscar en Supabase
        q_vector = model.encode(question).tolist()
        
        # Llamamos a la función RPC que creamos en el editor SQL de Supabase
        rpc_res = supabase.rpc("match_documents", {
            "query_embedding": q_vector,
            "match_threshold": 0.35, # Ajustable según precisión
            "match_count": 3
        }).execute()
        
        if not rpc_res.data:
            context = "No se encontró información relevante en los documentos."
        else:
            context = "\n".join([d['content'] for d in rpc_res.data])

        # 2. GENERATION: Enviar contexto y pregunta a Groq (Llama 3)
        prompt = (
            f"Eres un asistente técnico experto. Usa el siguiente contexto para responder.\n\n"
            f"Contexto: {context}\n\n"
            f"Pregunta: {question}\n\n"
            f"Respuesta concisa y profesional:"
        )
        
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        return {
            "answer": completion.choices[0].message.content,
            "sources": [d['content'] for d in rpc_res.data] if rpc_res.data else []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el proceso RAG: {str(e)}")