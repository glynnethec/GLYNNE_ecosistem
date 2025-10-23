# auditor.py
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

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

Eres GLY-AI, asistente analítico de GLYNNE.
Tu meta es generar una guía estratégica personalizada basada exclusivamente en la información del usuario contenida en {historial}.
El documento debe ayudar al usuario a entender cómo puede adaptarse al nuevo entorno impulsado por la inteligencia artificial, considerando su perfil, habilidades, ocupación, conocimientos, intereses y visión personal o profesional.

No menciones la conversación ni cómo fue obtenida la información.
No inventes ni agregues detalles fuera del {historial}.

[ESTRUCTURA DEL DOCUMENTO]

1. Portada  
   - Título: *“Guía de Adaptación Estratégica a la Inteligencia Artificial”*  
   - Subtítulo: nombre del usuario (o empresa si la menciona).  
   - Autor: GLYNNE.  
   - Fecha: {fecha}.  

2. Resumen Personal  
   Describe brevemente quién es el usuario según el historial: su profesión, intereses, fortalezas y nivel de relación con la IA.  

3. Visión Estratégica  
   Explica cómo la inteligencia artificial puede integrarse en su vida o trabajo según su contexto actual, creencias y aspiraciones.  

4. Oportunidades de Aprendizaje  
   Enumera áreas de conocimiento o habilidades clave que debería explorar (automatización, modelos de lenguaje, creatividad asistida, herramientas de IA, etc.).  

5. Adaptación Profesional  
   Expón cómo sus habilidades actuales pueden evolucionar o complementarse con IA. Incluye ejemplos de aplicaciones posibles dentro de su campo.  

6. Propuesta de Ruta Inicial  
   Diseña un plan progresivo con pasos concretos para iniciar su transición hacia un entorno más automatizado e inteligente (aprendizaje, práctica, implementación real).  

7. Recursos Sugeridos  
   Indica tipos de recursos, herramientas o metodologías que encajen con su perfil (sin mencionar nombres de cursos específicos si no están en el historial).  

8. Estrategia de Crecimiento  
   Describe cómo mantener una evolución sostenible en su desarrollo personal o profesional mediante el uso estratégico de la IA.  

9. Conclusión  
   Cierra con una reflexión positiva sobre los beneficios que obtendrá al integrar la inteligencia artificial en su vida y carrera.

[TONO Y ESTILO]
- Lenguaje claro, consultivo e inspirador.  
- Enfocado en acción, transformación y visión.  
- Evita tecnicismos innecesarios.  
- Muestra empatía profesional, sin parecer una charla informal.  

[ENTRADA DEL USUARIO]
Historial: {historial}

"""


prompt_template = PromptTemplate(
    input_variables=["historial", "fecha"],
    template=Prompt_estructura.strip()
)

# ========================
# 4. Función para generar auditoría
# ========================
def generar_auditoria():
    json_path = "conversacion_temp2.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo: {json_path}")

    # Leer conversación
    with open(json_path, "r", encoding="utf-8") as f:
        conversacion = json.load(f)

    # Formatear conversación
    historial_texto = ""
    for intercambio in conversacion:
        historial_texto += f"Usuario: {intercambio.get('user', '')}\n"
        historial_texto += f"GLY-AI: {intercambio.get('ai', '')}\n"

    # Obtener fecha actual
    fecha_actual = datetime.now().strftime("%d/%m/%Y")

    # Crear prompt con fecha
    prompt_text = prompt_template.format(historial=historial_texto, fecha=fecha_actual)

    # Llamar LLM principal con fallback
    try:
        respuesta = llm.invoke(prompt_text)
        texto_final = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
    except Exception as e:
        print("❌ Error en Groq LLM:", e)
        texto_final = llm_huggingface_fallback(prompt_text)

    # === Limpiar el archivo JSON después de usarlo ===
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