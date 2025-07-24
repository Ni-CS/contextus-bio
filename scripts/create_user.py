# scripts/create_admin.py (Versão Corrigida com .add())
import argilla as rg
from argilla.users import User

# --- CONFIGURAÇÕES ---
# Conecta-se usando as credenciais do usuário 'owner', que é criado para bootstrap
# e tem permissão para criar outros usuários.
try:
    client = rg.Argilla(
        api_url="http://localhost:6900",
        api_key="argilla.apikey" 
    )
    print("Conectado ao Argilla com as credenciais de 'owner'.")
except Exception as e:
    print(f"Não foi possível conectar como 'owner'. Verifique se os contêineres do Docker estão rodando. Erro: {e}")
    exit()

# --- Detalhes do novo usuário administrador que queremos criar ---
NEW_ADMIN_USERNAME = "nicholas"
NEW_ADMIN_PASSWORD = "12345678" # IMPORTANTE: Use uma senha segura

# --- LÓGICA CORRIGIDA PARA CRIAR O USUÁRIO ---
try:
    print(f"Tentando criar o novo usuário administrador: '{NEW_ADMIN_USERNAME}'...")
    
    new_user = User(
        username=NEW_ADMIN_USERNAME,
        role="annotator", # Define o papel como administrador
        password=NEW_ADMIN_PASSWORD,
    )
    
    # O método correto para adicionar um usuário é .add()
    try:
        user = client.users(new_user.username)
        user.delete()
        print(f"Usuário '{NEW_ADMIN_USERNAME}' já existe e foi removido.")
    except Exception as delete_error:
        print(f"Não foi possível remover o usuário '{NEW_ADMIN_USERNAME}'. Pode ser que ele não exista. Erro: {delete_error}")
        
    client.users.add(new_user)
    workspace = client.workspaces("default")
    new_user.add_to_workspace(workspace)
    
    print("\nSUCESSO!")
    print(f"Usuário administrador '{NEW_ADMIN_USERNAME}' foi criado.")
    print("Use este usuário e senha para fazer login e acessar o painel de administração.")

except Exception as e:
    # Este erro provavelmente acontecerá se você rodar o script uma segunda vez
    if "already exists" in str(e):
        print(f"\nAVISO: O usuário '{NEW_ADMIN_USERNAME}' já existe. Nenhuma ação foi tomada.")
        print("Você já pode fazer login com as credenciais definidas neste script.")
    else:
        print(f"\nOcorreu um erro inesperado: {e}")