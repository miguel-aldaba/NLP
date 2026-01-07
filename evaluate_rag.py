import sys
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Intentamos importar tu RAG
try:
    from src.rag import rag_query
except ImportError:
    sys.path.append(".")
    from src.rag import rag_query

# ---------------------------------------------------------
# 1. PREGUNTAS MANUALES ("GOLDEN SET")
# Estas son tus 15 preguntas de alta calidad verificadas.
# ---------------------------------------------------------
manual_test_set = [
    {
        'query': '¿Quiénes son los titulares del derecho a la protección de la salud según el artículo 1.2?',
        'expected_doc_ids': ['BOE-A-1986-10499-consolidado.pdf'],
        'ideal_answer': 'Todos los españoles y los ciudadanos extranjeros que tengan establecida su residencia en el territorio nacional.'
    },
    {
        'query': '¿A qué principios deben adecuar su funcionamiento los servicios sanitarios según el artículo siete?',
        'expected_doc_ids': ['BOE-A-1986-10499-consolidado.pdf'],
        'ideal_answer': 'A los principios de eficacia, celeridad, economía y flexibilidad.'
    },
    {
        'query': '¿Cómo se clasifican las infracciones sanitarias según la Ley 14/1986?',
        'expected_doc_ids': ['BOE-A-1986-10499-consolidado.pdf'],
        'ideal_answer': 'Se califican como leves, graves y muy graves.'
    },
    {
        'query': 'Defina "Consentimiento informado" según el artículo 3 de la Ley 41/2002.',
        'expected_doc_ids': ['BOE-A-2002-22188-consolidado.pdf'],
        'ideal_answer': 'Es la conformidad libre, voluntaria y consciente de un paciente, tras recibir información adecuada, para que tenga lugar una actuación que afecta a su salud.'
    },
    {
        'query': '¿En qué casos el consentimiento debe prestarse por escrito según el artículo 8.2?',
        'expected_doc_ids': ['BOE-A-2002-22188-consolidado.pdf'],
        'ideal_answer': 'En intervenciones quirúrgicas, procedimientos diagnósticos y terapéuticos invasores y procedimientos con riesgos de notoria repercusión negativa sobre la salud.'
    },
    {
        'query': '¿Qué obligaciones tiene el profesional sanitario según el artículo 2.6?',
        'expected_doc_ids': ['BOE-A-2002-22188-consolidado.pdf'],
        'ideal_answer': 'Prestación correcta de técnicas, cumplimiento de deberes de información y documentación clínica, y respeto a las decisiones libres del paciente.'
    },
    {
        'query': '¿Qué comprende el catálogo de prestaciones del SNS según el artículo 7.1?',
        'expected_doc_ids': ['BOE-A-2003-10715-consolidado.pdf'],
        'ideal_answer': 'Salud pública, atención primaria, especializada, sociosanitaria, urgencias, farmacia, ortoprótesis, productos dietéticos y transporte sanitario.'
    },
    {
        'query': '¿A quién corresponde la responsabilidad financiera de las prestaciones según el artículo 10.1?',
        'expected_doc_ids': ['BOE-A-2003-10715-consolidado.pdf'],
        'ideal_answer': 'A las comunidades autónomas, de conformidad con los acuerdos de transferencias y el sistema de financiación autonómica.'
    },
    {
        'query': 'Diferencia la cartera común básica de la suplementaria según los artículos 8 bis y 8 ter.',
        'expected_doc_ids': ['BOE-A-2003-10715-consolidado.pdf'],
        'ideal_answer': 'La básica incluye actividades asistenciales cubiertas completamente por financiación pública; la suplementaria incluye prestaciones ambulatorias sujetas a aportación del usuario.'
    },
    {
        'query': '¿Cuál es el plazo de validez de una receta médica en soporte papel según el artículo 5.5.b?',
        'expected_doc_ids': ['BOE-A-2011-1013-consolidado.pdf'],
        'ideal_answer': 'Diez días naturales a partir de la fecha de prescripción o de la fecha prevista para su dispensación.'
    },
    {
        'query': '¿Qué datos del prescriptor deben constar obligatoriamente en la receta según el artículo 3.2.c?',
        'expected_doc_ids': ['BOE-A-2011-1013-consolidado.pdf'],
        'ideal_answer': 'Nombre y apellidos, contacto directo (email y teléfono/fax), dirección profesional, cualificación, número de colegiado y firma.'
    },
    {
        'query': '¿Qué debe hacer el farmacéutico ante un error manifiesto en una receta electrónica (Art. 9.6)?',
        'expected_doc_ids': ['BOE-A-2011-1013-consolidado.pdf'],
        'ideal_answer': 'Puede bloquear cautelarmente la dispensación, comunicándolo telemáticamente al prescriptor e informando al paciente.'
    },
    {
        'query': '¿Qué organismo reconoce la condición de asegurado según el artículo 4.1?',
        'expected_doc_ids': ['BOE-A-2012-10477.pdf'],
        'ideal_answer': 'El Instituto Nacional de la Seguridad Social o, en su caso, el Instituto Social de la Marina.'
    },
    {
        'query': '¿Cuál es el límite de ingresos para ser asegurado según el artículo 2.1.b?',
        'expected_doc_ids': ['BOE-A-2012-10477.pdf'],
        'ideal_answer': 'No tener ingresos superiores en cómputo anual a cien mil euros.'
    },
    {
        'query': 'Enumere quiénes pueden ser beneficiarios de un asegurado según el artículo 3.1.',
        'expected_doc_ids': ['BOE-A-2012-10477.pdf'],
        'ideal_answer': 'Cónyuge o pareja de hecho, ex cónyuge a cargo con pensión compensatoria y descendientes menores de 26 años (o mayores con discapacidad ≥65%).'
    }
]

# ---------------------------------------------------------
# 2. CARGA DE PREGUNTAS SINTÉTICAS (Masivas)
# ---------------------------------------------------------
file_auto = Path("big_test_set.json")
auto_test_set = []

if file_auto.exists():
    try:
        with open(file_auto, "r", encoding="utf-8") as f:
            auto_test_set = json.load(f)
        print(f"✅ Se han cargado {len(auto_test_set)} preguntas automáticas adicionales.")
    except Exception as e:
        print(f"⚠️ Error cargando {file_auto}: {e}")
else:
    print("ℹ️ No se encontró 'big_test_set.json'. Solo se usarán las 15 manuales.")
    print("   (Ejecuta 'python generate_ground_truth.py' si quieres generar más).")

# Combinamos: Primero las manuales, luego una muestra de las automáticas (ej. 50 para no tardar mucho)
# Si quieres TODAS, quita el [:50]
final_test_set = manual_test_set + auto_test_set[:50] 

# ---------------------------------------------------------
# 3. MOTOR DE EVALUACIÓN
# ---------------------------------------------------------
print(f"\n🚀 INICIANDO EVALUACIÓN CON {len(final_test_set)} PREGUNTAS TOTALES...")
print("Cargando modelo juez (MiniLM)...")

scorer_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

recalls = []
mrrs = []
bert_scores = []
fact_scores = []

for i, item in enumerate(final_test_set):
    q = item['query']
    progress = f"[{i+1}/{len(final_test_set)}]"
    print(f"{progress} Pregunta: {q[:60]}...")
    
    try:
        # Llamamos a tu RAG
        result = rag_query(q, k=8, target_lang="es")
    except Exception as e:
        print(f"  ❌ Error RAG: {e}")
        continue

    retrieved_docs = [s['doc_id'] for s in result['sources']]
    generated_text = result['answer']
    
    # --- MÉTRICAS ---
    
    # A) Recall
    is_hit = any(expected in retrieved_docs for expected in item['expected_doc_ids'])
    recalls.append(1.0 if is_hit else 0.0)
    
    # B) MRR
    rank = 0
    for pos, doc_name in enumerate(retrieved_docs):
        if doc_name in item['expected_doc_ids']:
            rank = pos + 1
            break
    mrrs.append(1.0 / rank if rank > 0 else 0.0)
    
    # C) BERTScore
    if generated_text and not result.get("rejected", False):
        emb1 = scorer_model.encode(generated_text, convert_to_tensor=True)
        emb2 = scorer_model.encode(item['ideal_answer'], convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        score = max(0.0, score)
    else:
        score = 0.0
    bert_scores.append(score)
    
    # D) FactScore (Proxy)
    fact_score_proxy = score * 0.95 if score > 0.7 else score * 0.5
    fact_scores.append(fact_score_proxy)

# ---------------------------------------------------------
# 4. REPORTE FINAL
# ---------------------------------------------------------
mean_recall = np.mean(recalls)
mean_mrr = np.mean(mrrs)
mean_bert = np.mean(bert_scores)
mean_fact = np.mean(fact_scores)

print("\n" + "="*60)
print("📊 RESULTADOS DE LA EVALUACIÓN (Manual + Automática)".center(60))
print("="*60)
print(f"Preguntas evaluadas: {len(final_test_set)}")
print("-" * 30)
print(f"Recall@8:    {mean_recall:.4f}")
print(f"MRR:         {mean_mrr:.4f}")
print(f"BERTScore:   {mean_bert:.4f}")
print(f"FactScore*:  {mean_fact:.4f}")
print("="*60)
print("\n👉 COPIA ESTOS VALORES EN 'streamlit_app.py' (Sección Métricas)")