# main.py
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form # 👈 Se agregaron UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import json
import traceback

# === Importaciones del agente CSV ===
import pandas as pd
import tempfile
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# ===================================

# === Nuevas importaciones para TTS ===
import asyncio
import base64
import time
import edge_tts
# ===================================

# ========================
# 1. Importaciones de agentes
# ========================
from agent.chat import agente_node, get_memory, State, TEMP_JSON_PATH
from agent.chat1 import agente_node as agente_node_alt, get_memory as get_memory_alt
from agent.auditor import generar_auditoria as auditor_llm
from agentTTS.chat import responder_asistente
# ========================

# ========================
# 2. Inicialización FastAPI y Configuración LLM para CSV
# ========================
# Configuración del LLM para el Agente CSV (extraído de p1.py)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Esto es una comprobación crítica que debería hacerse, pero la mantendré
    # comentada si se ejecuta el main.py sin el .env configurado.
    # raise ValueError("❌ No hay API key en el .env")
    print("⚠️ GROQ_API_KEY no encontrada. El agente CSV fallará.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.3,
)

app = FastAPI(
    title="GLYNNE LLM API",
    description="API para interactuar con los agentes de GLY-AI (LangGraph, Auditoría, Chat1, TTS, CSV Analyzer)",  # 👈 Descripción actualizada
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


# === Modelo de datos para el endpoint de Chat TTS ===
class ConversarRequest(BaseModel):
    texto: str
    session_id: Optional[str] = "default_session"


class ConversarResponse(BaseModel):
    transcripcion_usuario: str
    respuesta_asistente: str
    audio_base64: str
    tokens: dict
# =========================================================

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
# 6. Endpoints Chat alternativo (agent/chat1.py)
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
        memory = get_memory_alt(request.user_id)
        if memory is None:
            raise Exception(f"Memoria no inicializada para user_id {request.user_id}")

        result = agente_node_alt(state)
        memoria = memory.load_memory_variables({}) or {}
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
        return {"mensaje": "✅ Auditoría generada correctamente", "auditoria": resultado}

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
        return {"mensaje": "✅ Plan estratégico generado correctamente", "plan": resultado}
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
        with open(TEMP_JSON_PATH2, "w", encoding="utf-                utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=4)
        return resultado
    except Exception as e:
        print("❌ Error en /generar_plan/json endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")




# ========================
# 20. Funciones Auxiliares del Agente CSV (Extraídas de agentsCSV/p1.py)
# ========================

def analizar_csv(file_path: str) -> str:
    """Genera un análisis técnico simple del CSV"""
    df = pd.read_csv(file_path)
    df = df.dropna(axis=1, how='all')

    numericas = df.select_dtypes(include=['number'])
    categoricas = df.select_dtypes(include=['object'])
    fechas = df.select_dtypes(include=['datetime64', 'datetime'])

    reporte = []

    if not numericas.empty:
        desc_num = numericas.describe().T
        reporte.append("RESUMEN NUMÉRICO:")
        for col, row in desc_num.iterrows():
            reporte.append(f"- {col}: media={row['mean']:.2f}, min={row['min']}, max={row['max']}, nulos={df[col].isna().sum()}")
    else:
        reporte.append("No hay columnas numéricas")

    if not categoricas.empty:
        reporte.append("RESUMEN CATEGÓRICO:")
        for col in categoricas.columns:
            top_val = df[col].value_counts().head(3)
            reporte.append(f"- {col}: {df[col].nunique()} valores únicos. Top 3: {', '.join(top_val.index.astype(str))}")
    else:
        reporte.append("No hay columnas categóricas")

    if not fechas.empty:
        reporte.append("RESUMEN FECHAS:")
        for col in fechas.columns:
            reporte.append(f"- {col}: rango {df[col].min()} → {df[col].max()}, nulos={df[col].isna().sum()}")

    nulls = df.isna().sum()
    cols_with_nulls = nulls[nulls > 0]
    if not cols_with_nulls.empty:
        reporte.append("VALORES FALTANTES:")
        for col, val in cols_with_nulls.items():
            reporte.append(f"- {col}: {val} nulos ({val/len(df)*100:.1f}%)")

    reporte.append(f"ESTADÍSTICAS GENERALES:")
    reporte.append(f"- Filas: {len(df)}")
    reporte.append(f"- Columnas: {len(df.columns)}")
    reporte.append(f"- Memoria usada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return "\n".join(reporte)


def generar_informe_llm(analisis_tecnico: str, descripcion: str) -> str:
    """Llama al LLM para generar un informe ejecutivo narrativo considerando la descripción del usuario"""
    prompt_final = f"""
Eres un analista de datos senior con enfoque consultivo. Analiza el siguiente dataset y genera un **informe ejecutivo narrativo** de 6–8 párrafos.
Toma en cuenta la descripción que da el usuario para contextualizar la información.

**DESCRIPCIÓN DEL DATASET:**
{descripcion}

**DATOS TÉCNICOS DEL DATASET:**
{analisis_tecnico}

**INSTRUCCIONES:**
- No te enfoques en explicar cada tabla o estadística.
- Infiera sobre estado general, procesos, riesgos, oportunidades y recomendaciones.
- Usa un lenguaje claro, profesional y consultivo.
- Responde únicamente con el informe, sin encabezados.
"""
    respuesta = llm.invoke(prompt_final)
    return respuesta.content


def separar_columnas_csv(file_path: str):
    """Devuelve matrices separadas (numéricas / no numéricas)"""
    df = pd.read_csv(file_path)
    columnas_numericas = df.select_dtypes(include=["number"]).columns
    columnas_no_numericas = df.select_dtypes(exclude=["number"]).columns
    matriz_numerica = df[columnas_numericas].to_dict(orient="records")
    matriz_no_numerica = df[columnas_no_numericas].to_dict(orient="records")
    return {"numericas": matriz_numerica, "no_numericas": matriz_no_numerica}


# ========================
# 21. Endpoint Agente CSV (Procesar CSV)
# ========================
@app.post("/procesar-csv")
async def procesar_csv(
    file: UploadFile = File(...),
    descripcion: str = Form(...)
):
    """
    Recibe un CSV y una descripción corta del dataset.
    Genera un análisis técnico, un informe ejecutivo por LLM, y las tablas separadas.
    """
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos .csv")

        # 1. Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # 2. Analizar CSV (función extraída)
        analisis_tecnico = analizar_csv(tmp_path)

        # 3. Generar informe LLM con contexto (función extraída)
        informe = generar_informe_llm(analisis_tecnico, descripcion)

        # 4. Separar columnas para el frontend (función extraída)
        tablas = separar_columnas_csv(tmp_path)

        # 5. Limpiar el archivo temporal
        os.remove(tmp_path)

        return {
            "tablas": tablas,
            "analisis_tecnico": analisis_tecnico,
            "informe_ejecutivo": informe
        }

    except Exception as e:
        print("❌ Error en /procesar-csv endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")


# ========================
# 11. Entrypoint Uvicorn
# ========================
if __name__ == "__main__":
    print("🚀 Servidor GLYNNE API corriendo con soporte para múltiples agentes, TTS/SST y CSV")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)