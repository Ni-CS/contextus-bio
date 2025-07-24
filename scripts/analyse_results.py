# scripts/analyze_results.py
import argilla as rg
import pandas as pd

# --- CONFIGURAÇÕES ---
DATASET_NAME = "avaliacao-final-contextus-bio"

# --- INICIALIZAÇÃO DO CLIENTE ARGILLA ---
try:
    # Use as credenciais de admin para ter acesso a todas as respostas
    client = rg.Argilla(
        api_url="http://localhost:6900",
        api_key="argilla.apikey" # ou a chave do seu usuário 'admin_contextus'
    )
    print("Conectado ao servidor Argilla com sucesso.")
except Exception as e:
    print(f"Não foi possível conectar ao Argilla. Erro: {e}")
    exit()

# --- CARREGAMENTO DO DATASET COM AS RESPOSTAS ---
try:
    print(f"Puxando o dataset '{DATASET_NAME}' do servidor...")
    # Carrega o dataset remoto
    remote_dataset = client.datasets(DATASET_NAME)
    print("Dataset carregado com sucesso.")
except Exception as e:
    print(f"Não foi possível encontrar o dataset '{DATASET_NAME}'. Certifique-se de que o nome está correto. Erro: {e}")
    exit()

# --- PROCESSAMENTO E ANÁLISE DAS RESPOSTAS ---
# Lista para armazenar nossos dados processados
analysis_data = []

# Itera sobre cada registro (cada par de busca/resultado) no dataset
for record in remote_dataset.records:
    # Para cada registro, itera sobre as respostas dadas pelos usuários
    if not record.responses:
        # Se um registro ainda não foi avaliado, podemos registrar isso
        base_info = {
            "query": record.fields["query"],
            "result_text": record.fields["result_text"],
            "page": record.fields["page"],
            "user": "N/A",
            "relevancia": "Pendente"
        }
        analysis_data.append(base_info)
    else:
        for response in record.responses:
            # Coleta as informações que queremos
            user_info = client.users(id=response.user_id) # Obtém o nome do usuário a partir do ID
            print(user_info)
            processed_info = {
                "query": record.fields["query"],
                "result_text": record.fields["result_text"],
                "page": record.fields["page"],
                "user": user_info.username, # Nome do usuário que avaliou
                "relevancia": response.value # O valor da resposta (1 ou 0)
            }
            analysis_data.append(processed_info)

# Converte a lista de dados para um DataFrame do Pandas para fácil visualização
df_results = pd.DataFrame(analysis_data)

# --- EXIBIÇÃO DOS RESULTADOS ---
print("\n--- Tabela de Avaliação Completa ---")
print(df_results)

print("\n--- Métricas Gerais de Relevância ---")
# Filtra apenas as respostas que não estão pendentes
completed_df = df_results[df_results['relevancia'] != 'Pendente'].copy()
# Converte a coluna 'relevancia' para numérico para poder calcular a média
completed_df['relevancia'] = pd.to_numeric(completed_df['relevancia'])

if not completed_df.empty:
    # A precisão geral é a média de todas as avaliações (soma de 1s / total)
    overall_precision = completed_df['relevancia'].mean()
    print(f"Total de Avaliações Submetidas: {len(completed_df)}")
    print(f"Precisão Geral (Média de Relevância): {overall_precision:.2%}")

    print("\n--- Análise por Usuário ---")
    # Agrupa por usuário e calcula a média de relevância e o total de avaliações
    user_summary = completed_df.groupby('user')['relevancia'].agg(['mean', 'count'])
    user_summary.rename(columns={'mean': 'Precisão Média', 'count': 'Avaliações Feitas'}, inplace=True)
    print(user_summary)
else:
    print("Nenhuma avaliação foi submetida ainda.")