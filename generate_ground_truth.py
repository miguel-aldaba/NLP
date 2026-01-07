import re
from pathlib import Path
import json
import random

# Configuración
DATA_DIR = Path("data/raw")  # Donde están tus PDFs/TXTs
OUTPUT_FILE = "big_test_set.json"

def extract_articles_from_text(text, filename):
    """
    Busca patrones tipo 'Artículo 1.' o 'Artículo uno.' y extrae su contenido.
    """
    qa_pairs = []
    
    # Patrón simple: "Artículo X." seguido de texto hasta el siguiente Artículo o final
    # Regex explica: Busca 'Artículo' seguido de número o letras, punto y captura lo que sigue
    pattern = re.compile(r'(Artículo\s+(?:\d+|[a-z]+)\.?)(.*?)(?=Artículo|$)', re.DOTALL | re.IGNORECASE)
    
    matches = pattern.findall(text)
    
    for title, content in matches:
        content = content.strip()
        if len(content) < 50: continue # Ignoramos artículos vacíos o muy cortos
        
        # Limpiamos el texto
        content_clean = re.sub(r'\s+', ' ', content).strip()
        
        # Creamos la pregunta sintética
        question = f"¿Qué establece el {title.strip()} del documento {filename}?"
        
        qa_pairs.append({
            "query": question,
            "expected_doc_ids": [filename.replace(".txt", ".pdf")], # Asumimos que el retriever busca PDFs
            "ideal_answer": content_clean[:500] # Cogemos los primeros 500 caracteres como "verdad"
        })
        
    return qa_pairs

def main():
    if not DATA_DIR.exists():
        print(f"Error: No encuentro la carpeta {DATA_DIR}")
        return

    all_questions = []
    
    # Procesamos los TXT que tengas en data/processed o data/raw
    # Si solo tienes PDFs en raw, intenta buscar los TXT que subiste o extráelos
    # Aquí busco en data/processed asumiendo que guardaste los txt ahí, si no, ajusta la ruta.
    files = list(Path("data/processed").glob("*.txt"))
    
    # Si no hay txt en processed, miramos si hay txt en la raíz (por si acaso)
    if not files:
        files = list(Path(".").glob("*.txt"))
        
    print(f"Procesando {len(files)} documentos...")
    
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            pairs = extract_articles_from_text(text, f.name)
            print(f"  -> {f.name}: {len(pairs)} preguntas generadas.")
            all_questions.extend(pairs)
        except Exception as e:
            print(f"Error leyendo {f.name}: {e}")

    # Mezclamos y guardamos
    random.shuffle(all_questions)
    
    # Guardamos en JSON para que el script de evaluación lo pueda leer
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
        
    print(f"\n¡Éxito! Se han generado {len(all_questions)} preguntas en '{OUTPUT_FILE}'.")
    print("Ahora ejecuta 'evaluate_rag.py' modificándolo para cargar este JSON.")

if __name__ == "__main__":
    main()