import pandas as pd
import numpy as np
import os

# --- CONFIGURAÇÕES ---
EVALUATION_FILE = 'evaluations.json'
APPROACH_IDS = [
    "approach_1_text_only",
    "approach_2_hybrid_clip",
    "approach_3_pure_clip",
    "approach_4_unified_hybrid"
]

# --- SCRIPT DE ANÁLISE ---

# Carrega os dados da avaliação
try:
    df_eval = pd.read_json(EVALUATION_FILE, lines=True)
    df_eval['is_relevant'] = df_eval['evaluation'].apply(lambda x: 1 if x == 'relevante' else 0)
except ValueError:
    print(f"Arquivo '{EVALUATION_FILE}' não encontrado ou vazio.")
    exit()

# Passo 1: Construir o "Gabarito" (Ground Truth) para cada query
ground_truth = {}
# Agrupa por query e coleta todos os IDs de resultados únicos que foram marcados como relevantes
relevant_docs = df_eval[df_eval['is_relevant'] == 1].groupby('query')['result_id'].unique().apply(set)
ground_truth = relevant_docs.to_dict()

print("--- Gabarito (Total de Itens Relevantes por Query) ---")
for query, items in ground_truth.items():
    print(f"Query: '{query}' -> {len(items)} itens relevantes")

# Passo 2: Calcular as métricas para cada abordagem
final_results = []

for approach_id in APPROACH_IDS:
    df_approach = df_eval[df_eval['approach_id'] == approach_id]
    
    query_metrics = []
    for query, relevant_set in ground_truth.items():
        # Documentos que esta abordagem retornou para esta query
        retrieved_set = set(df_approach[df_approach['query'] == query]['result_id'])
        
        if not retrieved_set:
            continue # Pula se a busca não retornou nada

        # Calcula os componentes da matriz de confusão
        true_positives = len(retrieved_set.intersection(relevant_set))
        false_positives = len(retrieved_set) - true_positives
        false_negatives = len(relevant_set) - true_positives # Relevantes que não foram encontrados
        
        # Calcula as métricas para esta query
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        query_metrics.append({'precision': precision, 'recall': recall, 'f1': f1})

    # Calcula a média das métricas para a abordagem
    if query_metrics:
        avg_precision = np.mean([m['precision'] for m in query_metrics])
        avg_recall = np.mean([m['recall'] for m in query_metrics])
        avg_f1 = np.mean([m['f1'] for m in query_metrics])
    else:
        avg_precision, avg_recall, avg_f1 = 0, 0, 0
        
    final_results.append({
        "Abordagem": approach_id,
        "Precisão Média": f"{avg_precision:.2%}",
        "Recall Médio": f"{avg_recall:.2%}",
        "F1-Score Médio": f"{avg_f1:.3f}"
    })

# Exibe a tabela final
df_final = pd.DataFrame(final_results)
print("\n\n--- Tabela Final de Métricas (Precisão, Recall, F1-Score) ---")
print(df_final.to_string(index=False))