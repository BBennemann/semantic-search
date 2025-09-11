import streamlit as st
import warnings
from logic_indexing import executar_indexacao, criar_indice_se_necessario, NOME_DO_INDICE
from logic_indexing import carregar_modelo, conectar_elasticsearch, buscar_semantica, listar_fontes_indexadas
from elasticsearch import ConnectionError

st.set_page_config(page_title="Buscador Semântico", page_icon="🔎", layout="centered")
warnings.filterwarnings("ignore", "Unverified HTTPS request")

# Carrega os recursos pesados no início
@st.cache_resource
def inicializar_recursos():
    try:
        modelo = carregar_modelo()
        cliente_es = conectar_elasticsearch()
        return modelo, cliente_es
    except ConnectionError as e: 
        st.error(f"Erro de conexão: {e}. Verifique se o Elasticsearch está no ar.")
        return None, None 
    except Exception as e:
        st.error(f"Ocorreu um erro durante a inicialização: {e}")
        return None, None
    
modelo, cliente_es = inicializar_recursos()

@st.cache_data
def carregar_documentos_indexados(_client):
    if not _client:
        return []
    return listar_fontes_indexadas(_client)

# Barra lateral para ações administrativas
with st.sidebar:
    st.header("Administração")
    if st.button("Re-indexar Base de Dados"):
        if cliente_es and modelo:
            with st.spinner("Limpando índice e re-indexando... Isso pode demorar."):
                cliente_es.indices.delete(index=NOME_DO_INDICE, ignore_unavailable=True)
                criar_indice_se_necessario(cliente_es)
                sucessos, falhas = executar_indexacao(cliente_es, modelo)
            st.success(f"Indexação concluída: {sucessos} documentos processados.")
            if falhas > 0:
                st.warning(f"{falhas} documentos falharam.")
            st.cache_data.clear()
        else:
            st.error("Conexão ou modelo de IA indisponível.")

    st.divider() 

    st.header("Documentos no Índice")
    with st.spinner("Carregando lista de documentos..."):
        lista_de_arquivos = carregar_documentos_indexados(cliente_es)
    
    if lista_de_arquivos:
        with st.expander(f"{len(lista_de_arquivos)} ficheiros encontrados", expanded=True):
            for nome_arquivo in sorted(lista_de_arquivos):
                st.markdown(f"📄 `{nome_arquivo}`")
    else:
        st.info("Nenhum documento foi indexado ainda.")

# Corpo principal da aplicação
st.title("🔎 Buscador Semântico de Arquivos")

if cliente_es and modelo:

    with st.form(key="search_form"):
        query_usuario = st.text_input(
            "Digite sua busca aqui:", 
            placeholder="Ex: quem foi Einstein?"
        )
        submit_button = st.form_submit_button(label="Buscar")
    
    if submit_button and query_usuario:

        with st.spinner("Buscando..."):
            try:
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
            except Exception as e:
                st.error(f"Ocorreu um erro durante a busca: {e}")
else:
    st.error("A aplicação não pôde ser iniciada. Verifique a conexão com o Elasticsearch.")