import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import warnings

# Importa a lógica de indexação do novo módulo.
from logic_indexing import executar_indexacao, criar_indice_se_necessario, NOME_DO_INDICE

# --- Configurações da Página ---
st.set_page_config(page_title="Buscador Semântico", page_icon="🔎", layout="centered")
warnings.filterwarnings("ignore", "Unverified HTTPS request")

# --- Constante do Modelo ---
MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Cache de Recursos ---
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer(MODELO_EMBEDDING)

@st.cache_resource
def conectar_elasticsearch():
    client = Elasticsearch(
        hosts=["http://localhost:9200"],
        verify_certs=False,
        ssl_show_warn=False
    )
    if not client.ping():
        st.error("Falha na conexão com Elasticsearch. Verifique os contêineres Docker.")
        return None
    return client

# --- Lógica da Busca ---
def buscar_semantica(client, model, consulta: str, top_k: int = 3):
    """Executa a busca semântica KNN no índice."""
    vetor_consulta = model.encode(consulta)
    query_knn = {
        "field": "embedding_texto",
        "query_vector": vetor_consulta,
        "k": top_k,
        "num_candidates": 10
    }
    try:
        response = client.search(
            index=NOME_DO_INDICE,
            knn=query_knn,
            source=["texto", "fonte_arquivo"]
        )
        return response['hits']['hits']
    except Exception as e:
        st.error(f"Erro na busca: {e}")
        return []

# --- Construção da Interface ---

# Carrega os recursos pesados no início
modelo = carregar_modelo()
cliente_es = conectar_elasticsearch()

# Barra lateral para ações administrativas
with st.sidebar:
    st.header("Administração")
    if st.button("Re-indexar Base de Dados"):
        if cliente_es and modelo:
            with st.spinner("Limpando índice e re-indexando... Isso pode demorar."):
                # Apaga o índice antigo para garantir consistência
                cliente_es.indices.delete(index=NOME_DO_INDICE, ignore_unavailable=True)
                # Cria o índice com o mapeamento correto
                criar_indice_se_necessario(cliente_es)
                # Executa a indexação em lote e captura os resultados
                sucessos, falhas = executar_indexacao(cliente_es, modelo)
            st.success(f"Indexação concluída: {sucessos} documentos processados.")
            if falhas > 0:
                st.warning(f"{falhas} documentos falharam.")
        else:
            st.error("Conexão ou modelo de IA indisponível.")

# Corpo principal da aplicação
st.title("🔎 Buscador Semântico de Arquivos")

if cliente_es and modelo:
    query_usuario = st.text_input(
        "Digite sua busca aqui:", 
        placeholder="Ex: quem foi Einstein?"
    )
    
    if query_usuario:
        resultados = buscar_semantica(cliente_es, modelo, query_usuario)
        st.subheader("Resultados da Busca:")
        if resultados:
            for resultado in resultados:
                with st.container(border=True):
                    st.markdown(f"**Texto Encontrado:** {resultado['_source']['texto']}")
                    st.caption(f"Fonte: **{resultado['_source']['fonte_arquivo']}**")
                    st.write(f"**Score de Similaridade:** {resultado['_score']:.4f}")
        else:
            st.info("Nenhum resultado encontrado.")
else:
    if cliente_es:
        st.error('modelo')
    else:
        st.error('client')
    st.error("A aplicação não pôde ser iniciada. Verifique a conexão com o Elasticsearch.")