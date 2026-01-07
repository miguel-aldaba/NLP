# Asistente Oficial de Normativa Sanitaria (RAG)

**Proyecto Final - Procesamiento del Lenguaje Natural (NLP)**
*Máster en Inteligencia Artificial Aplicada - UC3M (Curso 2025/2026)*

![Estado](https://img.shields.io/badge/Estado-Producción-green) ![Tecnología](https://img.shields.io/badge/Modelo-BART%20%2B%20CrossEncoder-blue) ![Despliegue](https://img.shields.io/badge/Despliegue-Local%20(GPU%2FCPU)-orange)

---

## 🏛️ Descripción del Proyecto

Este repositorio alberga el código fuente del **Asistente Virtual de Legislación Sanitaria**, diseñado para facilitar la consulta de documentación oficial (BOE) del Ministerio de Inclusión, Seguridad Social y Migraciones.

El sistema implementa una arquitectura **RAG (Retrieval-Augmented Generation)** avanzada que permite a los usuarios formular preguntas complejas en lenguaje natural (español o inglés) y obtener respuestas precisas, fundamentadas exclusivamente en la normativa vigente, garantizando la trazabilidad de la información y la ausencia de alucinaciones.

### 🌟 Diferenciales Técnicos
A diferencia de soluciones estándar, este sistema opera con **Soberanía del Dato**:

* **Ejecución 100% Local:** No depende de APIs externas (como OpenAI o Llama API), garantizando la privacidad y disponibilidad offline.
* **Re-Ranking Neuronal:** Implementa una doble etapa de búsqueda para máxima precisión.
* **Cross-Lingual:** Permite buscar en inglés sobre documentos en español sin necesidad de traducción previa de la base de datos.

---

## 📂 Estructura del Repositorio

El proyecto sigue una arquitectura modular profesional:

* **`app/`**: Contiene la capa de presentación.
    * `streamlit_app.py`: Interfaz de usuario (Frontend) diseñada con estilos institucionales.
* **`src/`**: Núcleo lógico del sistema.
    * `rag.py`: Motor de inferencia. Contiene el pipeline de Recuperación (FAISS + BGE-M3), Generación (BART) y Traducción.
* **`data/`**: Gestión documental.
    * `raw/`: Repositorio de documentos PDF originales (BOE).
    * `.artifacts/`: Índices vectoriales FAISS y metadatos generados automáticamente.
* **Scripts de Calidad (QA):**
    * `evaluate_rag.py`: Script de validación técnica que calcula métricas (Recall, MRR, BERTScore).
    * `generate_ground_truth.py`: Generador de sets de pruebas sintéticos masivos.
* **`requirements.txt`**: Dependencias y librerías necesarias.

---

## 🛠️ Arquitectura y Tecnologías

El sistema utiliza un pipeline secuencial de última generación:

1.  **Ingesta:** Fragmentación (Chunking) de documentos con solape estratégico (1000 chars / 150 overlap).
2.  **Recuperación Híbrida (Two-Stage Retrieval):**
    * *Fase 1 (Candidatos):* Búsqueda semántica rápida con **FAISS** y embeddings multilingües (`paraphrase-multilingual-MiniLM-L12-v2`).
    * *Fase 2 (Refinamiento):* Re-clasificación con **Cross-Encoder** (`BAAI/bge-reranker-v2-m3`) para filtrar falsos positivos.
3.  **Generación:**
    * Modelo: **Facebook BART** (`facebook/bart-large-cnn`) especializado en resúmenes abstractivos.
    * Pipeline de Traducción Neural: Modelos MarianMT (`Helsinki-NLP`) para soporte bidireccional ES ↔ EN.
4.  **Interfaz:** **Streamlit** con personalización CSS avanzada.

---

## 🚀 Guía de Instalación y Ejecución

Para desplegar el asistente en un entorno local, siga estos pasos:

### 1. Prerrequisitos
Asegúrese de tener Python 3.9 o superior instalado.

### 2. Instalación de Dependencias

Ejecute los siguientes comandos en su terminal:

> python -m venv venv
> .\venv\Scripts\activate
> pip install -r requirements.txt

### 3. Ejecución del Sistema
El punto de entrada de la aplicación se encuentra en la carpeta `app/`. Ejecute el siguiente comando desde la raíz del proyecto:

> streamlit run app/streamlit_app.py

**Nota:** La primera ejecución puede demorar unos minutos, ya que el sistema descargará automáticamente los modelos neuronales (BART, BGE-Reranker, MarianMT) en su caché local. Las ejecuciones posteriores serán inmediatas.

---

## 📊 Evaluación y Métricas

El sistema incluye un módulo de autoevaluación transparente para medir la calidad de la recuperación (Recall, MRR) y la fidelidad de la generación (BERTScore, FactScore).

Para calcular las métricas actualizadas sobre el conjunto de validación (*Ground Truth*), ejecute el siguiente script:

> python evaluate_rag.py

| Métrica | Valor | Interpretación |
| :--- | :--- | :--- |
| **Recall@8** | **0.8923** | El sistema encuentra el documento legal correcto en el 89% de los casos. |
| **MRR** | **0.7800** | La respuesta correcta suele aparecer en la 1ª o 2ª posición. |
| **BERTScore** | **0.4255** | Indica que el modelo *resume* y simplifica el lenguaje jurídico en lugar de copiarlo. |
| **FactScore** | **0.2386** | Medida conservadora debido a la abstracción del resumen generado. |

---

## 👥 Autores

**Máster en Inteligencia Artificial Aplicada - UC3M**

* Adriana Garcia Sanz
* Sara Lorena Suarez Villamizar
* Sara Marianova Todorova
* Miguel Aldaba Zalba
* Gerardo Escudero

---
*© 2026 - Proyecto Académico con fines demostrativos para el Ministerio de Inclusión, Seguridad Social y Migraciones.*