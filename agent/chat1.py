import os
import random
import json
from dotenv import load_dotenv
from typing import TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory




# ========================
# 1. Configuración
# ========================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("en el .env no hay una api valida")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.8,
     
)

# ========================
# 2. Prompt optimizado para tokenización
# ========================
Prompt_estructura = """
[CONTEXTO]
Hoy es {fecha}.
Eres GLY-AI, guía experto de la plataforma GLYNNE.
Tu objetivo es interactuar de forma cercana y clara, mostrando cómo usar las herramientas del menú “+”, sus funciones y beneficios.

Herramientas principales:

Razonamiento:

 rol: enseña las funcionalidades que se encuentran en el menu de + ahi esta el modo de auditoria procesos empresariales, gestor de talento humano, analisis de data 
 
Auditor AI: mediante una combersacion con el usuario GLY-ai hace preguntas espoecificas para saber que es el dolor del cliente y con base a eso crear un orizonte para adaptarlos a la ia y como nosotros GLYNNE como podemos ayudarlo 

Asistente de talento humano: guía al usuario para adaptarse a la era de la IA e implementarla en su vida o trabajo este es el modo gestor humano, hace preguntas especificas sobre quien es y que hace el usuario y con base a eso señaliza en primera instancia como el usuario puede adaptar su vida profesional a la ia .

Analizador de datos:
Interpreta datasets CSV responde preguntas especificas referentes a la DB, organiza los datos y muestra gráficos y análisis automáticos usando álgebra lineal y estadística descriptiva e inferencial.

Off Framework (Desarrollador):
Framework descargable y listo para usar. Permite crear, configurar y ejecutar múltiples agentes de IA integrables en producción. ofrece motor completo de ia generativa para integrar a ccualquier proyeccto de desarrollo, esta en V1.0 por temas de optimizacio pronto saldra la version completa  

Documentación y comunidad:
Recursos para aprender, compartir y mejorar agentes de IA. 

recuerda que estas herramientas son open source pero que el nicho de glynne es el desarrollo de automatizacion de procesos con IA y arquitectura de software B2B 

Adapta el lenguaje al usuario, usa ejemplos simples y un tono profesional y educativo.

[MEMORIA]
Últimos 3 mensajes: {historial}

[ENTRADA DEL USUARIO]
Consulta: {mensaje}

[RESPUESTA COMO {rol}]
Máximo 100 palabras.

"""

prompt = PromptTemplate(
    input_variables=["rol", "mensaje", "historial", "fecha"],
    template=Prompt_estructura.strip(),
)

# ========================
# 3. Estado global
# ========================
class State(TypedDict):
    mensaje: str
    rol: str
    historial: str
    respuesta: str
    user_id: str

# memoria por usuario
usuarios = {}

def get_memory(user_id: str):
    if user_id not in usuarios:
        # limitar memoria para reducir tokens: solo últimos 3 mensajes
        usuarios[user_id] = ConversationBufferMemory(
            memory_key="historial",
            input_key="mensaje",
            output_key="respuesta",
            k=4
        )
    return usuarios[user_id]

# ========================
# 4. Función de almacenamiento temporal en JSON
# ========================
TEMP_JSON_PATH = "conversacion_temp.json"

if not os.path.exists(TEMP_JSON_PATH):
    with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

def guardar_conversacion(user_msg: str, ai_resp: str):
    with open(TEMP_JSON_PATH, "r+", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
        data.append({"user": user_msg, "ai": ai_resp})
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

# ========================
# 5. Nodo principal
# ========================
def agente_node(state: State) -> State:
    memory = get_memory(state.get("user_id", "default"))
    historial = memory.load_memory_variables({}).get("historial", "")

    # limitar historial a últimos 3 mensajes (si k falla)
    if historial:
        lineas = historial.strip().split("\n")
        if len(lineas) > 6:  # cada intercambio ≈ 2 líneas
            historial = "\n".join(lineas[-6:])

    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    texto_prompt = prompt.format(
        rol=state["rol"],
        mensaje=state["mensaje"],
        historial=historial,
        fecha=fecha_actual
    )

    respuesta = llm.invoke(texto_prompt).content

    # guardar en memoria
    memory.save_context({"mensaje": state["mensaje"]}, {"respuesta": respuesta})

    # guardar en JSON temporal
    guardar_conversacion(state["mensaje"], respuesta)

    state["respuesta"] = respuesta
    state["historial"] = historial
    return state

# ========================
# 6. Construcción del grafo
# ========================
workflow = StateGraph(State)
workflow.add_node("agente", agente_node)
workflow.set_entry_point("agente")
workflow.add_edge("agente", END)
app = workflow.compile()

# ========================
# 7. CLI interactiva
# ========================
print("LLM iniciado con LangGraph")

user_id = str(random.randint(10000, 90000))
print(f"tu user id es {user_id}")

rol = "auditor"
