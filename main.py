# =========================================================
# 🌐 GLYNNE ECOSISTEM API
# Servidor central que orquesta los agentes LLM, auditorías y TTS.
# =========================================================

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import base64
import tempfile
import os
import edge_tts
import time
import json
import traceback
import uvicorn

# =========================================================
# 1. IMPORTACIÓN DE AGENTES PRINCIPALES
# =========================================================
from agent.chat import agente_node, get_memory, State, TEMP_JSON_PATH
from agent.chat1 import agente_node as agente_node_alt, get_memory as get_memory_alt
from agent.auditor import generar_auditoria as auditor_llm

# =========================================================
# 2. IMPORTACIÓN DE AGENTES SECUNDARIOS
# =========================================================
from agent2.chat import (
    agente_node as agente2_node,
    get_memory as get_memory2,
    State as State2,
    TEMP_JSON_PATH as TEMP_JSON_PATH2,
)
from agent2.auditor import generar_auditoria as auditor_llm2

# =========================================================
# 3. IMPORTACIÓN DE AGENTE DE VOZ (TTS)
# =========================================================
from agentTTS.chat import responder_asistente


# =========================================================
# 4. CONFIGURACIÓN FASTAPI
# =========================================================
app = FastAPI(
    title="GLYNNE LLM API",
    description="API para agentes GLY-IA, Auditoría y TTS.",
    version="2.0",
)

# =========================================================
# 5. CONFIGURACIÓN CORS
# =========================================================
origins = [
    "https://glynne-sst-ai-hsiy.vercel.app",
    "http://localhost:3000",
    "https://glynne-ecosistem.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 6. MODELOS DE DATOS
# =========================================================
class ChatRequest(BaseModel):
    mensaje: str
    rol: Optional[str] = "auditor"
    user_id: str

class ChatResponse(BaseModel):
    respuesta: str
    historial: dict


# =========================================================
# 7. ENDPOINTS DE CHAT (AGENTE 1 Y 2)
# =========================================================
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat principal basado en agent/chat.py"""
    try:
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id es obligatorio")

        state: State = {
            "mensaje": request.mensaje,
            "rol": request.rol,
            "historial": "",
            "respuesta": "",
            "user_id": request.user_id,
        }

        result = agente_node(state)
        memoria = get_memory(request.user_id).load_memory_variables({})
        return ChatResponse(respuesta=result.get("respuesta", ""), historial=memoria)
    except Exception as e:
        print("❌ Error en /chat endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/chat1", response_model=ChatResponse)
def chat1(request: ChatRequest):
    """Chat alternativo basado en agent/chat1.py"""
    try:
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id es obligatorio")

        state: State = {
            "mensaje": request.mensaje,
            "rol": request.rol,
            "historial": "",
            "respuesta": "",
            "user_id": request.user_id,
        }

        memory = get_memory_alt(request.user_id)
        result = agente_node_alt(state)
        memoria = memory.load_memory_variables({}) or {}
        return ChatResponse(respuesta=result.get("respuesta", ""), historial=memoria)
    except Exception as e:
        print("❌ Error en /chat1 endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/chat2", response_model=ChatResponse)
def chat2(request: ChatRequest):
    """Chat basado en agent2/chat.py"""
    try:
        if not request.user_id:
            raise HTTPException(status_code=400, detail="user_id es obligatorio")

        state: State2 = {
            "mensaje": request.mensaje,
            "rol": request.rol,
            "historial": "",
            "respuesta": "",
            "user_id": request.user_id,
        }

        result = agente2_node(state)
        memoria = get_memory2(request.user_id).load_memory_variables({})
        return ChatResponse(respuesta=result.get("respuesta", ""), historial=memoria)
    except Exception as e:
        print("❌ Error en /chat2 endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# =========================================================
# 8. ENDPOINTS DE AUDITORÍA Y PLAN
# =========================================================
@app.post("/generar_auditoria")
def generar_auditoria(user_id: str):
    try:
        if not os.path.exists(TEMP_JSON_PATH):
            raise HTTPException(status_code=404, detail="No hay conversación para generar auditoría")
        resultado = auditor_llm()
        return {"mensaje": "✅ Auditoría generada correctamente", "auditoria": resultado}
    except Exception as e:
        print("❌ Error en /generar_auditoria:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/generar_plan")
def generar_plan(user_id: str):
    try:
        if not os.path.exists(TEMP_JSON_PATH2):
            raise HTTPException(status_code=404, detail="No hay conversación para generar plan")
        resultado = auditor_llm2()
        return {"mensaje": "✅ Plan estratégico generado correctamente", "plan": resultado}
    except Exception as e:
        print("❌ Error en /generar_plan:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# =========================================================
# 9. ENDPOINTS DE UTILIDAD (STATUS Y RESET)
# =========================================================
@app.get("/")
async def get_root():
    """Verifica que el backend esté activo"""
    return {"status": "ok", "message": "Backend is live"}


@app.get("/reset")
def reset_all():
    """Resetea memorias temporales"""
    try:
        for path in [TEMP_JSON_PATH, TEMP_JSON_PATH2]:
            if os.path.exists(path):
                os.remove(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)
        return {"status": "ok", "message": "Memorias y JSONs reiniciados"}
    except Exception as e:
        print("❌ Error en /reset:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# =========================================================
# 🔊 10. ENDPOINT DE TEXTO A VOZ (TTS)
# =========================================================
activo = True

async def hablar_async_to_file(texto, filepath):
    """Convierte texto a audio usando Edge-TTS"""
    texto = texto.strip()
    if not texto:
        raise ValueError("Texto vacío recibido para TTS")

    texto = texto.replace("“", '"').replace("”", '"').replace("’", "'")

    communicate = edge_tts.Communicate(
        texto,
        voice="es-CO-SalomeNeural",
        rate="+18%",
        pitch="+13Hz",
    )
    await communicate.save(filepath)


@app.post("/conversar")
async def conversar(request: Request):
    """Convierte en tiempo real la respuesta del agente TTS"""
    global activo
    if not activo:
        return JSONResponse(content={"error": "Servicio desactivado temporalmente"}, status_code=503)

    try:
        data = await request.json()
        texto_usuario = data.get("texto", "").strip()

        if not texto_usuario:
            return JSONResponse(content={"error": "No se recibió texto válido"}, status_code=400)

        session_id = "default_session"
        resultado = await responder_asistente(texto_usuario, session_id)

        if isinstance(resultado, tuple):
            respuesta, tokens_info = resultado
        else:
            respuesta = resultado
            tokens_info = {
                "usuario": len(texto_usuario.split()),
                "llm": len(respuesta.split()),
                "total": len(texto_usuario.split()) + len(respuesta.split()),
            }

        temp_path = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time())}.mp3")
        await hablar_async_to_file(respuesta, temp_path)

        with open(temp_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        os.remove(temp_path)

        return {
            "transcripcion_usuario": texto_usuario,
            "respuesta_asistente": respuesta,
            "audio_base64": audio_base64,
            "tokens": tokens_info,
        }

    except Exception as e:
        print("❌ Error en /conversar:")
        traceback.print_exc()
        return JSONResponse(content={"error": f"Error interno: {str(e)}"}, status_code=500)


# =========================================================
# 11. ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    print("🚀 Servidor GLYNNE API corriendo en http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
