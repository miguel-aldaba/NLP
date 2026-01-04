# Proyecto Final NLP - Chatbot RAG (UC3M)

Este repositorio contiene el código de nuestro proyecto final para la asignatura de **Procesamiento del Lenguaje Natural** (Máster en Inteligencia Artificial Aplicada, Curso 2025/2026).

Hemos desarrollado un **Chatbot RAG (Retrieval-Augmented Generation)** capaz de responder preguntas basándose exclusivamente en documentos oficiales del BOE. El sistema busca la información relevante, responde y cita la fuente, evitando inventarse datos (alucinaciones).

## 📂 ¿Qué hay en este repositorio?

* **`streamlit_app.py`**: El código de la interfaz gráfica. Es lo que ejecuta la web del chat para que el usuario pueda preguntar.
* **`rag.py`**: Aquí está la lógica del sistema. Este script conecta con la base de datos vectorial, busca los fragmentos de texto y se comunica con el LLM de la universidad.
* **`NLP_codigobase.ipynb`**: El notebook que usamos para preparar los datos. Aquí limpiamos los PDFs, creamos los embeddings y generamos el índice.
* **`faiss.index`** y **`chunks.json`**: Son los archivos de nuestra base de datos vectorial ya generada (donde busca el chatbot).
* **`requirements.txt`**: Las librerías que hacen falta para que funcione todo.
* **`data/`**: Carpeta con los documentos originales del BOE.

## 🚀 Cómo probarlo en local

Si quieres ejecutar el chatbot en tu ordenador, sigue estos pasos:

### 1. Prepara el entorno
Descarga el código y asegúrate de instalar las dependencias necesarias. Recomendamos usar un entorno virtual, pero puedes instalarlo directo con:

```bash
pip install -r requirements.txt
2. Configuración de la API
El proyecto usa los modelos Llama desplegados en los servidores de la UC3M.

Abre el archivo rag.py.

Busca la variable UC3M_API_KEY y asegúrate de que tiene la clave correcta para acceder a la URL yiyuan.tsc.uc3m.es.

3. Ejecutar el Chatbot
Para lanzar la aplicación, usa el siguiente comando en la terminal:

Bash

streamlit run streamlit_app.py
Automáticamente se debería abrir una pestaña en tu navegador (normalmente en http://localhost:8501) donde podrás empezar a chatear con los documentos.

🛠️ Tecnologías
Lenguaje: Python

Interfaz: Streamlit

RAG: Implementación propia usando LangChain/LlamaIndex.

Base de datos vectorial: FAISS

Modelo: Llama 3.1 (vía API UC3M)

✅ Funcionalidades clave
Búsqueda Semántica: Entiende el significado de la pregunta, no solo busca palabras clave.

Citas de fuentes: Cada respuesta te dice exactamente de qué documento del BOE ha sacado la información.

Control de alucinaciones: Si el chatbot no encuentra la respuesta en los documentos, te lo dice en lugar de inventársela.

Autores: ADRIANA GARCIA SANZ, GERARDO ESCUDERO LÓPEZ, SARA LORENA SUAREZ VILLAMIZAR, SARA MARIANOVA TODOROVA & MIGUEL ALDABA ZALBA. Máster en IA Aplicada - UC3M
