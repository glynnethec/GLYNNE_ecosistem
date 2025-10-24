# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
import traceback

# ========================
# 1. Importaciones de agentes
# ========================
from agent.chat import agente_node, get_memory, State, TEMP_JSON_PATH
from agent.chat1 import agente_node as agente_node_alt, get_memory as get_memory_alt
from agent.auditor import generar_auditoria as auditor_llm

# ========================
# 2. Inicialización FastAPI
# ========================
app = FastAPI(
    title="GLYNNE LLM API",
    description="API para interactuar con los agentes de GLY-AI (LangGraph, Auditoría, Chat1)",
    version="2.0"
)

# ========================
# 3. Middleware CORS
# ========================
origins = [
    "https://glynne-sst-ai-hsiy.vercel.app",
    "http://localhost:3000",  # para pruebas locales
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# 4. Modelos de datos
# ========================
class ChatRequest(BaseModel):
    mensaje: str
    rol: Optional[str] = "auditor"
    user_id: str  # obligatorio

class ChatResponse(BaseModel):
    respuesta: str
    historial: dict

# ========================
# 5. Endpoints Chat principal (agent/chat.py)
# ========================
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat principal basado en agent/chat.py"""
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id es obligatorio")

    state: State = {
        "mensaje": request.mensaje,
        "rol": request.rol,
        "historial": "",
        "respuesta": "",
        "user_id": request.user_id
    }

    try:
        result = agente_node(state)
        memoria = get_memory(request.user_id).load_memory_variables({})
        return ChatResponse(respuesta=result.get("respuesta", ""), historial=memoria)

    except Exception as e:
        print("❌ Error en /chat endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ========================
# 6. Endpoints Chat alternativo (agent/chat1.py) - CORREGIDO
# ========================
@app.post("/chat1", response_model=ChatResponse)
def chat1(request: ChatRequest):
    """Chat alternativo basado en agent/chat1.py (proceso separado)"""
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id es obligatorio")

    state: State = {
        "mensaje": request.mensaje,
        "rol": request.rol,
        "historial": "",
        "respuesta": "",
        "user_id": request.user_id
    }

    try:
        # Protección extra contra errores de memoria
        memory = get_memory_alt(request.user_id)
        if memory is None:
            raise Exception(f"Memoria no inicializada para user_id {request.user_id}")

        # Llamada al agente
        result = agente_node_alt(state)

        # Obtener historial seguro
        memoria = memory.load_memory_variables({})
        if memoria is None:
            memoria = {}

        return ChatResponse(respuesta=result.get("respuesta", ""), historial=memoria)

    except Exception as e:
        print("❌ Error en /chat1 endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ========================
# 7. Endpoint para obtener memoria por usuario
# ========================
@app.get("/user/{user_id}/memory")
def get_user_memory(user_id: str):
    try:
        memoria = get_memory(user_id).load_memory_variables({})
        return {"user_id": user_id, "historial": memoria}
    except Exception as e:
        print("❌ Error en /user/{user_id}/memory:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ========================
# 8. Endpoint reset de conversación temporal
# ========================
@app.get("/reset")
def reset_conversacion():
    """Elimina el JSON temporal y reinicia memoria"""
    try:
        if os.path.exists(TEMP_JSON_PATH):
            os.remove(TEMP_JSON_PATH)
        with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

        # Reiniciar memorias de ambos agentes
        for get_mem_func in [get_memory, get_memory_alt]:
            try:
                usuarios = get_mem_func.__defaults__[0] if get_mem_func.__defaults__ else {}
                for user_id in list(usuarios.keys()):
                    get_mem_func(user_id).clear()
            except Exception:
                pass

        return {"status": "ok", "message": "Conversaciones temporales reiniciadas"}
    except Exception as e:
        print("❌ Error en /reset endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ========================
# 9. Endpoint Auditoría
# ========================
@app.post("/generar_auditoria")
def generar_auditoria(user_id: str):
    """Genera auditoría llamando al agente de auditoría"""
    try:
        if not os.path.exists(TEMP_JSON_PATH):
            raise HTTPException(status_code=404, detail="No hay conversación para generar auditoría")

        resultado = auditor_llm()
        return {
            "mensaje": "✅ Auditoría generada correctamente",
            "auditoria": resultado
        }

    except Exception as e:
        print("❌ Error en /generar_auditoria endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# ========================
# 10. Endpoint Auditoría JSON
# ========================
@app.get("/generar_auditoria/json")
def generar_auditoria_json():
    """Devuelve la auditoría directamente en formato JSON"""
    try:
        if not os.path.exists(TEMP_JSON_PATH):
            raise HTTPException(status_code=404, detail="No hay conversación para generar auditoría")

        resultado = auditor_llm()
        with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=4)

        return resultado

    except Exception as e:
        print("❌ Error en /generar_auditoria/json endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
    
# ========================
# 12. Importaciones de agentes secundarios (agent2)
# ========================
from agent2.chat import agente_node as agente2_node, get_memory as get_memory2, State as State2, TEMP_JSON_PATH as TEMP_JSON_PATH2
from agent2.auditor import generar_auditoria as auditor_llm2

# ========================
# 13. Endpoints Chat principal (agent2/chat.py)
# ========================
@app.post("/chat2", response_model=ChatResponse)
def chat2(request: ChatRequest):
    """Chat principal basado en agent2/chat.py"""
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id es obligatorio")

    state: State2 = {
        "mensaje": request.mensaje,
        "rol": request.rol,
        "historial": "",
        "respuesta": "",
        "user_id": request.user_id
    }

    try:
        result = agente2_node(state)
        memoria = get_memory2(request.user_id).load_memory_variables({})
        return ChatResponse(respuesta=result.get("respuesta", ""), historial=memoria)

    except Exception as e:
        print("❌ Error en /chat2 endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 14. Endpoint para obtener memoria (agent2)
# ========================
@app.get("/user2/{user_id}/memory")
def get_user2_memory(user_id: str):
    """Obtiene la memoria del usuario en agent2"""
    try:
        memoria = get_memory2(user_id).load_memory_variables({})
        return {"user_id": user_id, "historial": memoria}
    except Exception as e:
        print("❌ Error en /user2/{user_id}/memory:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 15. Endpoint reset de conversación (agent2)
# ========================
@app.get("/reset2")
def reset_conversacion2():
    """Elimina el JSON temporal y reinicia memoria en agent2"""
    try:
        if os.path.exists(TEMP_JSON_PATH2):
            os.remove(TEMP_JSON_PATH2)
        with open(TEMP_JSON_PATH2, "w", encoding="utf-8") as f:
            json.dump([], f)

        # Reiniciar memorias del agente 2
        try:
            usuarios = get_memory2.__defaults__[0] if get_memory2.__defaults__ else {}
            for user_id in list(usuarios.keys()):
                get_memory2(user_id).clear()
        except Exception:
            pass

        return {"status": "ok", "message": "Conversaciones agent2 reiniciadas"}
    except Exception as e:
        print("❌ Error en /reset2 endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 16. Endpoint Documento Estratégico (agent2)
# ========================
@app.post("/generar_plan")
def generar_plan(user_id: str):
    """Genera el plan estratégico personalizado basado en agent2/auditor.py"""
    try:
        if not os.path.exists(TEMP_JSON_PATH2):
            raise HTTPException(status_code=404, detail="No hay conversación para generar el plan")

        resultado = auditor_llm2()
        return {
            "mensaje": "✅ Plan estratégico generado correctamente",
            "plan": resultado
        }

    except Exception as e:
        print("❌ Error en /generar_plan endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 17. Endpoint Documento Estratégico en JSON (agent2)
# ========================
@app.get("/generar_plan/json")
def generar_plan_json():
    """Devuelve el plan estratégico directamente en formato JSON"""
    try:
        if not os.path.exists(TEMP_JSON_PATH2):
            raise HTTPException(status_code=404, detail="No hay conversación para generar el plan")

        resultado = auditor_llm2()
        with open(TEMP_JSON_PATH2, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=4)

        return resultado

    except Exception as e:
        print("❌ Error en /generar_plan/json endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 11. Entrypoint Uvicorn
# ========================
if __name__ == "__main__":
    print("🚀 Servidor GLYNNE API corriendo con soporte para múltiples agentes")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
