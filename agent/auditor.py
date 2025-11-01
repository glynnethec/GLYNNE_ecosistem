# auditor.py
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# ========================
# 1. Cargar entorno y API
# ========================
load_dotenv()
api_key = os.getenv('GROQ_API_KEY2')
hf_api_key = os.getenv('HUGGINGFACE_API_KEY2')

if not api_key:
    raise ValueError('No hay una API key válida de Groq en el .env')

if not hf_api_key:
    raise ValueError('No hay una API key válida de Hugging Face en el .env')

# ========================
# 2. Inicializar LLM principal (Groq)
# ========================
llm = ChatGroq(
    model="Llama-3.1-8B-Instant",
    api_key=api_key,
    temperature=0.7,

)

# ========================
# 2b. LLM de fallback Hugging Face vía APIs
# ========================
def llm_huggingface_fallback(prompt_text: str) -> str:
    """
    Fallback a Hugging Face usando API Key
    """
    try:
        from transformers import pipeline

        generator = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            device=-1,
            use_auth_token=hf_api_key
        )

        output = generator(
            prompt_text,
            max_length=500,
            do_sample=True,
            top_p=0.95
        )
        return output[0]["generated_text"]

    except Exception as e:
        print("❌ Error fallback Hugging Face:", e)
        return "Lo siento, no pude generar la auditoría."

# ========================
# 3. Prompt de auditoría
# ========================
Prompt_estructura = """
[META]
Fecha del reporte: {fecha}

Genera un documento profesional y corporativo basado únicamente en la información del historial de conversación del usuario. El documento 


construye el documento dinamico la idea es que con historial puedas hacer que que el usuario sepa como la ia puede automatizar eso en especifico lo ideal es poder gestionar un documento claro y no tan tecnico 


Lenguaje claro, profesional y comprensible todo muy profesional.

No inventar datos; usar solo información del historial.

Entrada:
Historial de conversación: {historial}
responde como asesor ia en la estructura pero recuerda que el documento es de la empresa GLYNNE ai
Salida esperada:
Documento 
"""


prompt_template = PromptTemplate(
    input_variables=["historial", "fecha"],
    template=Prompt_estructura.strip()
)

# ========================
# 4. Función para generar auditoría (OPTIMIZADA)
# ========================

def generar_auditoria():
    json_path = "conversacion_temp.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo: {json_path}")

    # Leer conversación
    with open(json_path, "r", encoding="utf-8") as f:
        conversacion = json.load(f)

    # === 🔹 EXTRAER SOLO MENSAJES DEL USUARIO ===
    mensajes_usuario = []
    for intercambio in conversacion:
        user_msg = intercambio.get("user", "").strip()
        if user_msg:
            mensajes_usuario.append(user_msg)

    # Si no hay mensajes, error
    if not mensajes_usuario:
        raise ValueError("No hay mensajes del usuario en el historial para generar la auditoría.")

    # === 🔹 Crear texto del historial solo del usuario ===
    historial_texto = "\n".join([f"- {msg}" for msg in mensajes_usuario])

    # === 🔹 Contexto previo al modelo ===
    contexto = (
        "A continuación se te proporcionan únicamente los mensajes escritos por el usuario. "
        "El usuario ha hecho preguntas relacionadas con la optimización de procesos empresariales "
        "y desea construir una guía práctica sobre cómo puede usar la inteligencia artificial "
        "para mejorar dichos procesos. No estás recibiendo las respuestas anteriores del asistente, "
        "solo las preguntas del usuario.\n\n"
        "Mensajes del usuario:\n"
    )

    # === 🔹 Construir prompt completo ===
    fecha_actual = datetime.now().strftime("%d/%m/%Y")
    prompt_text = prompt_template.format(historial=contexto + historial_texto, fecha=fecha_actual)

    # === 🔹 Llamar al modelo LLM con fallback ===
    try:
        respuesta = llm.invoke(prompt_text)
        texto_final = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
    except Exception as e:
        print("❌ Error en Groq LLM:", e)
        texto_final = llm_huggingface_fallback(prompt_text)

    # === 🔹 Limpiar el archivo JSON después de usarlo ===
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("✅ Archivo de conversación limpiado después de generar la auditoría.")
    except Exception as e:
        print("❌ Error al limpiar el archivo JSON:", e)

    return texto_final


# ========================
# 5. CLI opcional para pruebas
# ========================
if __name__ == "__main__":
    print("LLM Auditoría iniciado")
    try:
        resultado = generar_auditoria()
        print("\n===== AUDITORÍA =====\n")
        print(resultado)
        print("\n=====================\n")
    except Exception as e:
        print("❌ Error general:", e)