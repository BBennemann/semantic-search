import streamlit as st
import warnings
import os
from logic_indexing import (
    executar_indexacao, criar_indice_se_necessario, NOME_DO_INDICE,
    apagar_documentos_por_fonte, carregar_modelo, conectar_elasticsearch,
    buscar_semantica, listar_fontes_indexadas
)
from file_processor import salvar_arquivo_recebido
from elasticsearch import ConnectionError

st.set_page_config(page_title="Buscador Semântico", page_icon="🔎", layout="centered")
warnings.filterwarnings("ignore", "Unverified HTTPS request")

@st.cache_resource
def inicializar_recursos():
    """
    Loads and initializes heavy resources like the embedding model and Elasticsearch client.
    This function is cached by Streamlit to avoid reloading on every script run.
    """
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
    """
    Fetches the list of unique source files from the Elasticsearch index.
    This function is cached to prevent re-querying on every UI interaction.
    """
    if not _client:
        return []
    return listar_fontes_indexadas(_client)

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

    st.header("Upload e Indexar Novos Documentos")
    arquivos_upload = st.file_uploader(
        "Escolha os arquivos para upload (.txt, .pdf, .csv)",
        type=["txt", "pdf", "csv"],
        accept_multiple_files=True,
    )

    if st.button("Processar e Indexar Arquivos"):
        if arquivos_upload:
            with st.spinner("Processando e indexando os novos arquivos..."):
                for arquivo in arquivos_upload:
                    salvar_arquivo_recebido(arquivo.name, arquivo.getbuffer())

                sucessos, falhas = executar_indexacao(cliente_es, modelo)
                st.success(
                    f"Indexação concluída: {sucessos} documentos processados."
                )
                if falhas > 0:
                    st.warning(f"{falhas} documentos falharam durante a indexação.")
                st.cache_data.clear()
        else:
            st.warning("Por favor, faça o upload de pelo menos um arquivo.")

    st.divider()

    st.header("Documentos no Índice")
    with st.spinner("Carregando lista de documentos..."):
        lista_de_arquivos = carregar_documentos_indexados(cliente_es)

    if lista_de_arquivos:
         with st.expander("Ver documentos indexados", expanded=True):
            for nome_arquivo in sorted(lista_de_arquivos):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.markdown(f"📄 `{nome_arquivo}`")
                with col2:
                    if st.button("🗑️", key=f"delete_{nome_arquivo}", help=f"Apagar {nome_arquivo}"):
                        try:
                            apagados = apagar_documentos_por_fonte(cliente_es, nome_arquivo)
                            st.toast(f"Ficheiro '{nome_arquivo}' ({apagados} partes) apagado!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Falha ao apagar {nome_arquivo}.")
    else:
        st.info("Nenhum documento foi indexado ainda.")

st.title("🔎 Buscador Semântico de Arquivos")

if cliente_es and modelo:

    num_resultados = st.number_input(
        "Número de resultados a exibir:",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )

    with st.form(key="search_form"):
        query_usuario = st.text_input(
            "Digite sua busca aqui:",
            placeholder="Ex: quem foi Einstein?"
        )
        submit_button = st.form_submit_button(label="Buscar")

    if submit_button and query_usuario:

        with st.spinner("Buscando..."):
            try:
                resultados = buscar_semantica(cliente_es, modelo, query_usuario, top_k=num_resultados)

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
