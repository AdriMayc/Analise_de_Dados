#%% 
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from datetime import datetime

# Configuração da página - Layout amplo (sem barra lateral)
st.set_page_config(
    page_title="Análise de E-commerce Olist",
    layout="wide"
)

# Título principal
st.title("Análise de Dados do E-commerce Brasileiro - Olist")
st.markdown("""
### Sobre o Projeto

Este dashboard analisa dados reais do e-commerce brasileiro **Olist**, com foco em três pilares:

- **Comportamento do cliente**: padrões de compra, segmentação via RFM, churn e avaliação de pedidos.
- **Eficiência logística**: análise de atrasos na entrega e impacto na satisfação.
- **Insights de negócio**: identificação de oportunidades regionais e recomendações estratégicas.

O projeto começa com o **carregamento e tratamento completo dos dados**, passando por conversão de datas, tradução de categorias, preenchimento de ausências e cálculo de métricas logísticas.

A seguir, são exploradas **visualizações interativas** que revelam relações entre logística, avaliação e comportamento de compra. Por fim, são apresentadas **recomendações práticas** para melhorar a experiência do cliente e a performance da operação.

""")

st.markdown("---")

# Função para carregar os dados
def carregar_dados():
    """Carrega todos os arquivos de dados do conjunto Olist"""
    try:
        data_path = "data/"
        
        customers = pd.read_csv(data_path + "olist_customers_dataset.csv")
        geolocation = pd.read_csv(data_path + "olist_geolocation_dataset.csv")
        order_items = pd.read_csv(data_path + "olist_order_items_dataset.csv")
        order_payments = pd.read_csv(data_path + "olist_order_payments_dataset.csv")
        order_reviews = pd.read_csv(data_path + "olist_order_reviews_dataset.csv")
        orders = pd.read_csv(data_path + "olist_orders_dataset.csv")
        products = pd.read_csv(data_path + "olist_products_dataset.csv")
        sellers = pd.read_csv(data_path + "olist_sellers_dataset.csv")
        product_category = pd.read_csv(data_path + "product_category_name_translation.csv")
        
        
        return {
            "customers": customers,
            "geolocation": geolocation, 
            "order_items": order_items,
            "order_payments": order_payments,
            "order_reviews": order_reviews,
            "orders": orders,
            "products": products,
            "sellers": sellers,
            "product_category": product_category
        }
    
    except FileNotFoundError:
        st.error("Arquivos não encontrados. Verifique se os arquivos estão na pasta 'data'.")
        return None

# Função para fazer o tratamento inicial dos dados
def tratar_dados(dataframes):
    """Realiza o tratamento inicial dos dados"""
    
    # Criando colunas para organizar o conteúdo da esquerda para a direita
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Conversão de Datas")
        # Convertendo colunas de data para o formato datetime
        if 'orders' in dataframes:
            date_columns = ['order_purchase_timestamp', 'order_approved_at', 
                          'order_delivered_carrier_date', 'order_delivered_customer_date',
                          'order_estimated_delivery_date']
            
            for col in date_columns:
                dataframes['orders'][col] = pd.to_datetime(dataframes['orders'][col], errors='coerce')
            
            st.write("✅ Datas em tabela 'orders' convertidas para datetime")
        
        if 'order_reviews' in dataframes:
            dataframes['order_reviews']['review_creation_date'] = pd.to_datetime(dataframes['order_reviews']['review_creation_date'], errors='coerce')
            dataframes['order_reviews']['review_answer_timestamp'] = pd.to_datetime(dataframes['order_reviews']['review_answer_timestamp'], errors='coerce')
            st.write("✅ Datas em tabela 'order_reviews' convertidas para datetime")
    
    with col2:
        st.subheader("2. Tradução de Categorias")
        # Traduzindo nomes das categorias de produtos
        if 'products' in dataframes and 'product_category' in dataframes:
            # Mostrar amostra antes da tradução
            st.write("Antes da tradução (amostra):")
            st.dataframe(dataframes['products'][['product_id', 'product_category_name']].head(3))
            
            # Merge para traduzir os nomes das categorias
            products_with_category = dataframes['products'].merge(
                dataframes['product_category'], 
                on='product_category_name', 
                how='left'
            )
            
            # Substituir o DataFrame original
            dataframes['products'] = products_with_category
            
            # Mostrar amostra após a tradução
            st.write("Após a tradução (amostra):")
            st.dataframe(dataframes['products'][['product_id', 'product_category_name', 'product_category_name_english']].head(3))
    
    st.markdown("---")
    
    # Segunda linha de colunas
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("3. Cálculo de Métricas de Entrega")
        # Calcular métricas relacionadas a entregas
        if 'orders' in dataframes:
            # Tempo de entrega (em dias)
            dataframes['orders']['delivery_time'] = (
                dataframes['orders']['order_delivered_customer_date'] - 
                dataframes['orders']['order_purchase_timestamp']
            ).dt.days
            
            # Se a entrega atrasou
            dataframes['orders']['delayed_delivery'] = dataframes['orders']['order_delivered_customer_date'] > dataframes['orders']['order_estimated_delivery_date']
            
            # Status de entrega simplificado
            status_map = {
                'delivered': 'entregue',
                'shipped': 'enviado',
                'canceled': 'cancelado',
                'unavailable': 'indisponível',
                'invoiced': 'faturado',
                'processing': 'processando',
                'approved': 'aprovado',
                'created': 'criado'
            }
            dataframes['orders']['order_status_pt'] = dataframes['orders']['order_status'].map(status_map)
            
            # Mostrar amostra após os cálculos
            st.write("Novas colunas criadas:")
            st.dataframe(dataframes['orders'][['order_id', 'delivery_time', 'delayed_delivery', 'order_status', 'order_status_pt']].head(5))
    
    with col4:
        st.subheader("4. Tratamento de Valores Ausentes")
        # Tratando valores ausentes nos produtos
        if 'products' in dataframes:
            # Contagem de valores ausentes antes do tratamento
            missing_before = dataframes['products'][['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']].isna().sum()
            st.write("Valores ausentes antes do tratamento:")
            st.write(missing_before)
            
            # Preenchendo valores ausentes nas dimensões e peso com a mediana de sua categoria
            for col in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
                # Agrupar por categoria e calcular a mediana
                median_by_category = dataframes['products'].groupby('product_category_name')[col].transform('median')
                # Preencher NaN com a mediana da categoria
                dataframes['products'][col] = dataframes['products'][col].fillna(median_by_category)
                # Se ainda houver NaN (categorias sem valores), preencher com a mediana geral
                dataframes['products'][col] = dataframes['products'][col].fillna(dataframes['products'][col].median())
            
            # Contagem de valores ausentes após o tratamento
            missing_after = dataframes['products'][['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']].isna().sum()
            st.write("Valores ausentes após o tratamento:")
            st.write(missing_after)
    
    st.markdown("---")
    
    # Terceira linha de colunas
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("5. Tratamento de Avaliações")
        # Tratando valores ausentes nos reviews
        if 'order_reviews' in dataframes:
            # Estatísticas antes do tratamento
            review_stats_before = {
                "Comentários ausentes": dataframes['order_reviews']['review_comment_message'].isna().sum(),
                "Scores ausentes": dataframes['order_reviews']['review_score'].isna().sum()
            }
            st.write("Antes do tratamento:")
            st.write(review_stats_before)
            
            # Preencher comentários vazios
            dataframes['order_reviews']['review_comment_message'] = dataframes['order_reviews']['review_comment_message'].fillna('Sem comentário')
            # Converter score para int (após tratar possíveis NaN)
            dataframes['order_reviews']['review_score'] = dataframes['order_reviews']['review_score'].fillna(0).astype(int)
            
            # Estatísticas após o tratamento
            review_stats_after = {
                "Comentários ausentes": dataframes['order_reviews']['review_comment_message'].isna().sum(),
                "Scores ausentes": dataframes['order_reviews']['review_score'].isna().sum()
            }
            st.write("Após o tratamento:")
            st.write(review_stats_after)
    
    with col6:
        st.subheader("6. Remoção de Duplicatas")
        # Verificar e remover duplicatas
        duplicates_info = {}
        duplicates_removed = {}
        
        for name, df in dataframes.items():
            # Contar duplicatas antes da remoção
            duplicates = df.duplicated().sum()
            duplicates_info[name] = duplicates
            
            # Remover duplicatas se existirem
            if duplicates > 0:
                rows_before = df.shape[0]
                dataframes[name] = df.drop_duplicates(keep='first')
                rows_after = dataframes[name].shape[0]
                duplicates_removed[name] = rows_before - rows_after
        
        # Exibir informações sobre duplicatas
        st.write("Duplicatas encontradas por tabela:")
        for name, count in duplicates_info.items():
            if count > 0:
                st.write(f"• {name}: {count} duplicatas encontradas e removidas")
            else:
                st.write(f"• {name}: nenhuma duplicata")
        
        # Resumo da remoção
        if sum(duplicates_info.values()) > 0:
            st.success(f"Total de {sum(duplicates_info.values())} duplicatas foram encontradas e removidas com sucesso.")
        else:
            st.success("Não foram encontradas duplicatas nos dados.")
    
    st.markdown("---")
    st.subheader("Resumo do Tratamento de Dados")
    
    # Resumo do tratamento
    col7, col8 = st.columns(2)
    
    with col7:
        # Exibir dimensões de cada dataframe
        st.write("Dimensões dos Dados:")
        dimensions = {}
        for name, df in dataframes.items():
            dimensions[name] = df.shape
        
        for name, dim in dimensions.items():
            st.write(f"• {name}: {dim[0]:,} linhas × {dim[1]} colunas")
    
    with col8:
        # Exibir informações sobre tipos de dados
        st.write("Tipos de Dados (amostra para orders):")
        if 'orders' in dataframes:
            st.write(dataframes['orders'].dtypes)
    
    return dataframes

def main():
    st.header("Carregamento e Tratamento dos Dados Olist")
    
    data = carregar_dados()
    
    if data:
        # Processar os dados
        data_tratados = tratar_dados(data)
        
        # Salvar os dados no estado da sessão (para uso posterior)
        st.session_state['data'] = data_tratados
        
        st.success("Tratamento de dados concluído com sucesso!")
        
        # Exibir cabeçalho de uma tabela tratada como exemplo
        st.subheader("Amostra dos Dados Tratados (Orders)")
        st.dataframe(data_tratados['orders'].head())

if __name__ == "__main__":
    main()

st.divider()

#%%
dados = carregar_dados()
orders = dados["orders"]
reviews = dados["order_reviews"]

df = pd.merge(orders, reviews[["order_id", "review_score"]], on="order_id", how="inner")
df["atraso_entrega"] = df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
df["atraso_entrega"] = df["atraso_entrega"].fillna(False)
df["status_entrega"] = df["atraso_entrega"].map({True: "Com Atraso", False: "Sem Atraso"})

# Título e explicação
st.markdown("""
### Relação entre Atraso na Entrega e Avaliação do Cliente

Visualização da distribuição das notas de avaliação conforme o status da entrega. A hipótese é que entregas com atraso tendem a resultar em notas mais baixas.
""")

# Gráfico Plotly
fig = px.box(
    df,
    x="status_entrega",
    y="review_score",
    color="status_entrega",
    points=False,
    title="<b>Avaliações de Clientes por Status de Entrega</b>",
    labels={
        "status_entrega": "Entrega",
        "review_score": "Nota de Avaliação"
    },
    width=1000,
    height=500,
    color_discrete_sequence=["#00cc96", "#EF553B"]  # Verde para sem atraso, vermelho para com
)

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    margin=dict(l=50, r=50, t=80, b=50),
    font=dict(color='white')  # Fica melhor com tema escuro
)

st.plotly_chart(fig, use_container_width=True)

# Cálculo das médias
media_sem_atraso = df[df["atraso_entrega"] == False]["review_score"].mean()
media_com_atraso = df[df["atraso_entrega"] == True]["review_score"].mean()

# Exibir valores para insights
st.markdown("### Médias das Avaliações:")
st.markdown(f"- 🟩 **Sem atraso:** {media_sem_atraso:.2f}")
st.markdown(f"- 🟥 **Com atraso:** {media_com_atraso:.2f}")

st.divider()

#%%

# 1. Juntar datasets necessários
df_reclamacoes = (
    dados['order_items']
    .merge(dados['products'], on='product_id')
    .merge(dados['order_reviews'], on='order_id')
    .merge(dados['product_category'], on='product_category_name')
)

# 2. Criar flag de reclamação (notas ruins)
df_reclamacoes['reclamacao'] = df_reclamacoes['review_score'] <= 2  # 1 ou 2

# 3. Agrupar por categoria traduzida
reclamacoes_por_categoria = (
    df_reclamacoes
    .groupby('product_category_name_english')
    .agg(total_pedidos=('order_id', 'count'),
         total_reclamacoes=('reclamacao', 'sum'))
    .reset_index()
)

# 4. Calcular taxa de reclamação (%)
reclamacoes_por_categoria['taxa_reclamacao'] = (
    reclamacoes_por_categoria['total_reclamacoes'] / reclamacoes_por_categoria['total_pedidos']
) * 100

# 5. Filtrar categorias com mais de 100 pedidos (para evitar distorções)
reclamacoes_filtrado = reclamacoes_por_categoria[reclamacoes_por_categoria['total_pedidos'] > 100]

# 6. Top 10 com maior taxa de reclamação
top10_reclamacoes = (
    reclamacoes_filtrado
    .sort_values(by='taxa_reclamacao', ascending=False)
    .head(10)
)


# Gráfico
fig = px.bar(
    top10_reclamacoes,
    x='taxa_reclamacao',
    y='product_category_name_english',
    orientation='h',
    text=top10_reclamacoes['taxa_reclamacao'].apply(lambda x: f'{x:.1f}%'),
    title='<b>Top 10 Categorias com Maior Taxa de Reclamações</b>',
    labels={
        'product_category_name_english': 'Categoria',
        'taxa_reclamacao': 'Taxa de Reclamação (%)'
    },
    color='taxa_reclamacao',
    color_continuous_scale='Blues',
    height=500
)

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    margin=dict(l=50, r=50, t=80, b=50)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
*Esta visualização mostra as **categorias com maior taxa de reclamações**, 
baseando-se em avaliações com nota 1 ou 2. 
Apenas categorias com mais de 100 pedidos foram consideradas, para garantir relevância estatística.*
""")

def gerar_texto_reclamacoes(df_top):
    """Gera texto descritivo com base no top 10 de categorias com maior taxa de reclamação"""
    texto = ""

    df_top = df_top.sort_values(by='taxa_reclamacao', ascending=False).reset_index(drop=True)

    cat_top1 = df_top.loc[0, 'product_category_name_english']
    taxa_top1 = df_top.loc[0, 'taxa_reclamacao']
    total_top1 = df_top.loc[0, 'total_pedidos']

    cat_top2 = df_top.loc[1, 'product_category_name_english']
    taxa_top2 = df_top.loc[1, 'taxa_reclamacao']

    cat_top3 = df_top.loc[2, 'product_category_name_english']
    taxa_top3 = df_top.loc[2, 'taxa_reclamacao']

    media_top10 = df_top['taxa_reclamacao'].mean()

    texto = "### Análise das Categorias com Maior Taxa de Reclamações\n\n"
    texto += f"- A categoria com maior taxa de reclamações é **{cat_top1}**, com **{taxa_top1:.1f}%** dos pedidos resultando em avaliações negativas (nota 1 ou 2), considerando um total de {total_top1} pedidos.\n\n"
    texto += f"- Em seguida, destacam-se **{cat_top2}** com **{taxa_top2:.1f}%** e **{cat_top3}** com **{taxa_top3:.1f}%**.\n\n"
    texto += f"- A média de reclamações entre as 10 categorias com pior desempenho é de **{media_top10:.1f}%**.\n\n"
    texto += f"- Esses dados indicam possíveis pontos críticos nessas categorias, que podem estar relacionados à qualidade dos produtos, logística ou experiência do cliente. Recomenda-se uma análise mais detalhada para identificação das causas e definição de ações corretivas.\n"

    return texto


st.markdown(gerar_texto_reclamacoes(top10_reclamacoes))

st.divider()

#%%

# 1. Juntar datasets necessários
df_correlacao = (
    dados['order_items']
    .merge(dados['order_reviews'], on='order_id')
    .merge(dados['products'], on='product_id')
)

# 2. Calcular a média das avaliações por produto
df_media_reviews = df_correlacao.groupby('product_id').agg(
    media_avaliacao=('review_score', 'mean'),
    preco=('price', 'mean')
).reset_index()

# 3. Gerar gráfico de dispersão para visualizar a correlação
fig = px.scatter(
    df_media_reviews,
    x='preco',
    y='media_avaliacao',
    title='<b>Correlação entre Preço do Produto e Nota Média</b>',
    labels={'preco': 'Preço (R$)', 'media_avaliacao': 'Nota Média de Avaliação'},
    color='media_avaliacao',
    color_continuous_scale='Greens',
    height=500
)

# 4. Ajustes no layout para gráfico bonito e clean
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    margin=dict(l=50, r=50, t=80, b=50)
)

# Exibir gráfico
st.plotly_chart(fig, use_container_width=True)

# Calcular insights numéricos
correlacao = df_media_reviews['preco'].corr(df_media_reviews['media_avaliacao'])
preco_medio_5 = df_media_reviews[df_media_reviews['media_avaliacao'] == 5]['preco'].mean()
preco_medio_1 = df_media_reviews[df_media_reviews['media_avaliacao'] == 1]['preco'].mean()
preco_mais_bem_avaliado = df_media_reviews.sort_values('media_avaliacao', ascending=False).iloc[0]
preco_mais_mal_avaliado = df_media_reviews.sort_values('media_avaliacao').iloc[0]

# Texto com análise
st.markdown("### Análise da Correlação entre Preço e Avaliação")
st.markdown(f"""
- A correlação entre **preço** e **nota média de avaliação** é de **{correlacao:.2f}**, indicando uma relação {"positiva" if correlacao > 0 else "negativa" if correlacao < 0 else "nula"}.
- Produtos com **nota 5** têm, em média, preço de **R$ {preco_medio_5:,.2f}**.
- Já os produtos com **nota 1** têm preço médio de **R$ {preco_medio_1:,.2f}**.
- O produto com a **melhor avaliação** possui nota **{preco_mais_bem_avaliado['media_avaliacao']:.1f}** e preço médio de **R$ {preco_mais_bem_avaliado['preco']:,.2f}**.
- O produto com a **pior avaliação** tem nota **{preco_mais_mal_avaliado['media_avaliacao']:.1f}** e custa em média **R$ {preco_mais_mal_avaliado['preco']:,.2f}**.
""")

st.divider()

#%%
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px


# 1. Filtrar avaliações ruins (nota 1 ou 2)
clientes_ruins = dados['order_reviews'][dados['order_reviews']['review_score'] <= 2] \
    .merge(dados['orders'], on='order_id') \
    .merge(dados['order_items'], on='order_id') \
    .merge(dados['customers'], on='customer_id')  # Garantir que traga o customer_unique_id

# 2. Converter datas
clientes_ruins['order_purchase_timestamp'] = pd.to_datetime(clientes_ruins['order_purchase_timestamp'])

# 3. Referência temporal
referencia = clientes_ruins['order_purchase_timestamp'].max()

# 4. Calcular métricas RFM
df_rfm = clientes_ruins.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (referencia - x.max()).days,
    'order_id': 'nunique',
    'price': 'sum'
}).reset_index().rename(columns={
    'order_purchase_timestamp': 'recencia',
    'order_id': 'frequencia',
    'price': 'monetario'
})

# 5. Normalizar
scaler = StandardScaler()
rfm_normalizado = scaler.fit_transform(df_rfm[['recencia', 'frequencia', 'monetario']])

# 6. KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df_rfm['cluster'] = kmeans.fit_predict(rfm_normalizado)

# 7. Gráfico 3D
fig = px.scatter_3d(
    df_rfm,
    x='recencia',
    y='frequencia',
    z='monetario',
    color='cluster',
    title='<b>Clusterização de Clientes Insatisfeitos (RFM)</b>',
    labels={'recencia': 'Recência (dias)', 'frequencia': 'Frequência', 'monetario': 'Gasto (R$)'},
    color_discrete_sequence=px.colors.qualitative.Set1
)

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    margin=dict(l=50, r=50, t=80, b=50),
    height=600
)

st.plotly_chart(fig, use_container_width=True)


# 1. Resumo estatístico dos clusters
resumo_clusters = df_rfm.groupby('cluster').agg({
    'recencia': 'mean',
    'frequencia': 'mean',
    'monetario': 'mean',
    'customer_unique_id': 'count'
}).rename(columns={'customer_unique_id': 'qtde_clientes'}).round(2).reset_index()

# 2. Ordenar clusters por frequência ou monetário
resumo_clusters = resumo_clusters.sort_values(by='monetario', ascending=False)

# 3. Gerar texto de análise
texto = "### Análise dos Clusters de Clientes Insatisfeitos (RFM)\n\n"

for _, row in resumo_clusters.iterrows():
    texto += (
        f"- **Cluster {int(row['cluster'])}**: possui **{int(row['qtde_clientes'])} clientes**.\n"
        f"  - Recência média: **{row['recencia']:.1f} dias** desde a última compra\n"
        f"  - Frequência média: **{row['frequencia']:.1f} pedidos**\n"
        f"  - Gasto médio: **R$ {row['monetario']:.2f}**\n\n"
    )

texto += "*Esses grupos ajudam a identificar quais clientes insatisfeitos compram com frequência, gastam mais e podem ser prioritários para ações de recuperação.*"
st.markdown(texto)

st.dataframe(resumo_clusters)

# Filtro para escolher o cluster
cluster_selecionado = st.selectbox(
    "Selecione o cluster para visualizar os clientes:",
    df_rfm['cluster'].sort_values().unique()
)

# Filtrar o DataFrame com base no cluster selecionado
clientes_cluster = df_rfm[df_rfm['cluster'] == cluster_selecionado]

# Exibir tabela com os dados dos clientes
st.write(f"Clientes no cluster {cluster_selecionado}:")
st.dataframe(
    clientes_cluster[['customer_unique_id', 'recencia', 'frequencia', 'monetario']]
    .sort_values(by='recencia')
    .reset_index(drop=True)
)

# Título e descrição
st.header('Clusterização de Clientes Insatisfeitos')
st.write("""
Este sistema permite exportar os dados dos clientes de um cluster específico para um arquivo CSV ou Excel. Você pode selecionar um cluster e gerar os arquivos para análise.

Clique no botão abaixo para exportar os dados dos clientes de um determinado cluster.
""")

# Opção de selecionar o cluster
cluster_selecionado = st.selectbox(
    'Selecione o Cluster:',
    [0, 1, 2]
)

# Filtra os clientes do cluster selecionado
clientes_do_cluster = clientes_cluster[clientes_cluster['cluster'] == cluster_selecionado]

# Função para exportar os dados
def exportar_dados(format='csv'):
    if format == 'csv':
        clientes_do_cluster.to_csv(f'clientes_cluster_{cluster_selecionado}.csv', index=False)
        st.success(f'Os dados do Cluster {cluster_selecionado} foram exportados para CSV!')
    elif format == 'excel':
        clientes_do_cluster.to_excel(f'clientes_cluster_{cluster_selecionado}.xlsx', index=False)
        st.success(f'Os dados do Cluster {cluster_selecionado} foram exportados para Excel!')

# Botões para exportar os dados
col1, col2 = st.columns(2)

with col1:
    if st.button('Exportar para CSV'):
        exportar_dados(format='csv')

with col2:
    if st.button('Exportar para Excel'):
        exportar_dados(format='excel')

st.divider()

#%%
import json

st.header('Comportamento de Compra por Região')

@st.cache_data
def carregar_dados():
    data_path = "data/"
    customers = pd.read_csv(data_path + "olist_customers_dataset.csv", usecols=["customer_id", "customer_state"])
    order_items = pd.read_csv(data_path + "olist_order_items_dataset.csv", usecols=["order_id", "price"])
    orders = pd.read_csv(data_path + "olist_orders_dataset.csv", usecols=["order_id", "customer_id"])
    return customers, order_items, orders

def gerar_grafico_barras(customers, order_items, orders):
    vendas_por_estado = (
        orders
        .merge(order_items, on="order_id")
        .merge(customers, on="customer_id")
        .groupby("customer_state")["price"]
        .sum()
        .reset_index()
    )

    vendas_por_estado.columns = ["sigla", "vendas"]

    # Ordenando as vendas de menor para maior
    vendas_por_estado = vendas_por_estado.sort_values(by="vendas", ascending=True)

    fig = px.bar(
        vendas_por_estado,
        x="vendas",
        y="sigla",
        orientation="h",
        color="vendas",
        color_continuous_scale="Cividis",
        labels={"vendas": "Volume de Vendas (R$)", "sigla": "Estado"},
        title="Volume de Vendas por Estado",
        height=600
    )
    
    return fig, vendas_por_estado

# Carregar dados
customers, order_items, orders = carregar_dados()

# Gerar gráfico de barras e obter o dataframe de vendas por estado
fig, vendas_por_estado = gerar_grafico_barras(customers, order_items, orders)

# Exibir gráfico
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
O gráfico a cima mostra o total de vendas realizadas por estado brasileiro, revelando o volume financeiro movimentado em cada unidade da federação.

### Análise de Comportamento de Compra por Região
- Estados como **Roraima**, **Amapá** e **Acre** apresentam os **menores volumes de vendas** do país.
- **São Paulo** lidera com folga, somando **mais de R$ 5 milhões em vendas**.
- O valor movimentado por **SP é mais que o dobro da soma dos 10 estados com menor volume**.
- Esses dados evidenciam a **forte concentração de consumo no Sudeste**, especialmente em São Paulo.
- Há **potenciais oportunidades de crescimento** em regiões com menor movimentação, que podem ser exploradas com ações direcionadas de marketing e logística.
""")

# Exibir a tabela de vendas por estado após o gráfico
st.markdown("""
#### Tabela de Vendas por Estado
Abaixo está a tabela com o total de vendas realizadas em cada estado, que também pode ser útil para análises mais detalhadas.
""")

# Exibir a tabela de vendas
st.dataframe(vendas_por_estado, height=200)

st.divider()

#%%

# Dicionário de estados por região
estados_por_regiao = {
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"]
}

# Inverter para mapear estado -> região
estado_para_regiao = {estado: regiao for regiao, estados in estados_por_regiao.items() for estado in estados}

@st.cache_data
def carregar_dados():
    data_path = "data/"
    customers = pd.read_csv(data_path + "olist_customers_dataset.csv", usecols=["customer_id", "customer_state"])
    order_items = pd.read_csv(data_path + "olist_order_items_dataset.csv", usecols=["order_id", "price"])
    orders = pd.read_csv(data_path + "olist_orders_dataset.csv", usecols=["order_id", "customer_id"])
    return customers, order_items, orders

def calcular_ticket_medio_por_regiao(customers, order_items, orders):
    df = (
        orders
        .merge(order_items, on="order_id")
        .merge(customers, on="customer_id")
    )

    df["regiao"] = df["customer_state"].map(estado_para_regiao)

    ticket_medio = df.groupby("regiao").agg(
        total_vendas=("price", "sum"),
        total_pedidos=("order_id", "nunique")
    ).reset_index()

    ticket_medio["ticket_medio"] = ticket_medio["total_vendas"] / ticket_medio["total_pedidos"]
    ticket_medio = ticket_medio.sort_values(by="ticket_medio", ascending=True)
    return ticket_medio

# Carregar dados
customers, order_items, orders = carregar_dados()

# Calcular ticket médio
ticket_medio_df = calcular_ticket_medio_por_regiao(customers, order_items, orders)


# Gráfico
fig = px.bar(
    ticket_medio_df,
    x="ticket_medio",
    y="regiao",
    orientation="h",
    text_auto=".2s",
    labels={"ticket_medio": "Ticket Médio (R$)", "regiao": "Região"},
    color="ticket_medio",
    color_continuous_scale="Magma",
    title="Ticket Médio por Região",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Texto introdutório e insight
st.markdown("""
O gráfico mostra o **ticket médio por região**, ou seja, quanto em média os clientes gastam por pedido em cada parte do Brasil.

### Análise de Ticket Médio por Região
- Apesar de **Sudeste** e **Sul** liderarem em **volume total de vendas**, eles **não possuem os maiores tickets médios**.
- As regiões **Norte** e **Nordeste** apresentam os **maiores tickets médios do país**.
- Isso indica que, embora a **frequência de compras** nessas regiões possa ser menor, os **valores por transação são mais elevados**.
- Pode haver oportunidades para **explorar a recorrência de compra** nessas regiões com alto ticket.
""")

st.write('Abaixo temos uma tabela com maior detalhe, contendo todas as Regiões, Total de Vendas, Total de Pedidos e o Ticket Médio.')
# Exibir tabela
st.dataframe(ticket_medio_df, use_container_width=True)

st.divider()

#%%

@st.cache_data
def carregar_dados_categoria():
    data_path = "data/"
    orders = pd.read_csv(data_path + "olist_orders_dataset.csv", usecols=["order_id", "customer_id"])
    order_items = pd.read_csv(data_path + "olist_order_items_dataset.csv", usecols=["order_id", "product_id"])
    customers = pd.read_csv(data_path + "olist_customers_dataset.csv", usecols=["customer_id", "customer_state"])
    products = pd.read_csv(data_path + "olist_products_dataset.csv", usecols=["product_id", "product_category_name"])
    return orders, order_items, customers, products

def mapear_estado_para_regiao(estado):
    regioes = {
        "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
        "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "Centro-Oeste": ["DF", "GO", "MT", "MS"],
        "Sudeste": ["ES", "MG", "RJ", "SP"],
        "Sul": ["PR", "RS", "SC"],
    }
    for regiao, estados in regioes.items():
        if estado in estados:
            return regiao
    return "Outro"

def categoria_mais_vendida_por_regiao(orders, order_items, customers, products):
    df = (
        orders.merge(order_items, on="order_id")
              .merge(customers, on="customer_id")
              .merge(products, on="product_id")
    )
    df["regiao"] = df["customer_state"].apply(mapear_estado_para_regiao)
    vendas_por_categoria = (
        df.groupby(["regiao", "product_category_name"])
          .size()
          .reset_index(name="qtd_vendas")
    )
    mais_vendidas = (
        vendas_por_categoria.sort_values(["regiao", "qtd_vendas"], ascending=[True, False])
                            .groupby("regiao")
                            .first()
                            .reset_index()
    )
    return mais_vendidas

# Carregar dados
orders, order_items, customers, products = carregar_dados_categoria()

# Obter categorias mais vendidas por região
mais_vendidas_regiao = categoria_mais_vendida_por_regiao(orders, order_items, customers, products)

# Criar gráfico de barras
fig = px.bar(
    mais_vendidas_regiao,
    x="regiao",
    y="qtd_vendas",
    color="regiao",
    hover_data=["product_category_name"],
    labels={"qtd_vendas": "Quantidade de Vendas", "regiao": "Região", "product_category_name": "Categoria de Produto"},
    title="Categorias de Produtos Mais Vendidas por Região",
    height=600
)

# Exibir gráfico
st.plotly_chart(fig, use_container_width=True)

# Tabela
st.dataframe(mais_vendidas_regiao)

# Texto e análise
st.markdown("""
### Análise das Categorias por Região
- Produtos como `beleza_saude`, `cama_mesa_banho` e `moveis_decoracao` se destacam em algumas regiões.
- A região **Sudeste** apresenta um volume significativo de vendas na categoria **cama_mesa_banho**, com mais de 8.000 unidades, o que reflete uma maior demanda por esses produtos.
- **Norte** e **Centro-Oeste** têm categorias como **beleza_saude** com volumes menores, o que pode indicar mercados menos saturados para essas categorias nessas regiões.
""")

st.divider()
#%%

st.header('Eficiência da Logística (Entregas e Prazos)')

@st.cache_data
def carregar_dados_entrega():
    data_path = "data/"
    orders = pd.read_csv(data_path + "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])
    customers = pd.read_csv(data_path + "olist_customers_dataset.csv", usecols=["customer_id", "customer_state"])
    return orders, customers

def calcular_tempo_entrega(orders, customers):
    dados = (
        orders.dropna(subset=["order_delivered_customer_date"])
              .merge(customers, on="customer_id")
    )
    dados["tempo_entrega"] = (dados["order_delivered_customer_date"] - dados["order_purchase_timestamp"]).dt.days
    entrega_por_estado = dados.groupby("customer_state")["tempo_entrega"].mean().reset_index()
    entrega_por_estado = entrega_por_estado.sort_values(by="tempo_entrega", ascending=False)
    return entrega_por_estado

def plotar_tempo_entrega(df):
    fig = px.bar(
        df,
        x="tempo_entrega",
        y="customer_state",
        orientation="h",
        title="Tempo Médio de Entrega por Estado",
        labels={"tempo_entrega": "Dias", "customer_state": "Estado"},
        color="tempo_entrega",
        color_continuous_scale="Viridis",
        height=600
    )
    return fig

# Carregar e processar dados
orders, customers = carregar_dados_entrega()
entrega_por_estado = calcular_tempo_entrega(orders, customers)

# Mostrar gráfico
st.plotly_chart(plotar_tempo_entrega(entrega_por_estado), use_container_width=True)

# Tabela
st.dataframe(entrega_por_estado, height=200)

# Texto e insight
st.markdown("""
O gráfico a mostra o **tempo médio (em dias) que os pedidos levaram para serem entregues em cada estado do Brasil**.

**Análise de Tempo Médio de Entrega por Localidade:**
- Estados da região **Norte** e **Centro-Oeste** geralmente apresentam **tempos médios de entrega mais longos**, sugerindo possíveis desafios logísticos ou maior distância dos centros de distribuição.
- Em contraste, regiões como **Sudeste** e **Sul** possuem **entregas mais rápidas**, reflexo de melhor infraestrutura, maior densidade urbana e proximidade com polos logísticos.
- Esses dados indicam oportunidades de melhoria logística em áreas menos favorecidas, que podem impactar positivamente a satisfação do cliente e os prazos de entrega.
""")

st.divider()


#%%

@st.cache_data
def carregar_dados_atrasos():
    data_path = "data/"
    orders = pd.read_csv(
        data_path + "olist_orders_dataset.csv",
        parse_dates=["order_delivered_customer_date", "order_estimated_delivery_date"]
    )
    customers = pd.read_csv(
        data_path + "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_state"]
    )
    return orders, customers

def calcular_atrasos_reais(orders, customers):
    dados = (
        orders.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
              .merge(customers, on="customer_id")
    )
    dados["dias_atraso"] = (dados["order_delivered_customer_date"] - dados["order_estimated_delivery_date"]).dt.days
    dados = dados[dados["dias_atraso"] > 0]  # Considera apenas entregas com atraso
    atraso_por_estado = dados.groupby("customer_state")["dias_atraso"].mean().reset_index()
    atraso_por_estado = atraso_por_estado.sort_values(by="dias_atraso", ascending=False)
    return atraso_por_estado

def plotar_atrasos(df):
    fig = px.bar(
        df,
        x="dias_atraso",
        y="customer_state",
        orientation="h",
        title="Tempo Médio de Atraso por Estado (Real x Estimado)",
        labels={"dias_atraso": "Dias de Atraso", "customer_state": "Estado"},
        color="dias_atraso",
        color_continuous_scale="Rdylbu",
        height=600
    )
    return fig

# Execução no Streamlit
orders, customers = carregar_dados_atrasos()
df_atrasos = calcular_atrasos_reais(orders, customers)

st.plotly_chart(plotar_atrasos(df_atrasos), use_container_width=True)
st.dataframe(df_atrasos, height=200)

# Texto explicativo
st.markdown("""
O gráfico acima apresenta o **tempo médio de atraso na entrega** dos pedidos em cada estado do Brasil, comparando a data de entrega real com a data estimada originalmente.

### Análise de Atrasos Reais nas Entregas
- Estados mais distantes dos grandes centros, como os da região **Norte**, tendem a apresentar os **maiores atrasos médios**.
- Já os estados do **Sudeste** e **Sul**, com maior infraestrutura e proximidade de centros logísticos, mostram **menores índices de atraso**.
- Entender esses gargalos é fundamental para redesenhar rotas e melhorar a experiência de entrega.
""")

st.divider()

#%%
@st.cache_data
def carregar_dados_rfm():
    path = "data/"
    orders = pd.read_csv(path + "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    order_items = pd.read_csv(path + "olist_order_items_dataset.csv", usecols=["order_id", "price"])
    return orders, order_items

def calcular_rfm(orders, order_items):
    # Apenas pedidos entregues
    pedidos = orders[orders["order_status"] == "delivered"].copy()

    # Combina pedidos com valores
    dados = pedidos.merge(order_items, on="order_id")

    # Data de referência para cálculo da recência
    data_ref = dados["order_purchase_timestamp"].max()

    # Agrupamento por cliente
    rfm = dados.groupby("customer_id").agg({
        "order_purchase_timestamp": lambda x: (data_ref - x.max()).days,  # Recência
        "order_id": "nunique",  # Frequência
        "price": "sum"  # Valor monetário
    }).reset_index()

    rfm.columns = ["customer_id", "Recencia", "Frequencia", "Valor"]

    # Score RFM (dividido em 4 quantis)
    rfm["R_quartil"] = pd.qcut(rfm["Recencia"], 4, labels=[4, 3, 2, 1])
    rfm["F_quartil"] = pd.qcut(rfm["Frequencia"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    rfm["M_quartil"] = pd.qcut(rfm["Valor"], 4, labels=[1, 2, 3, 4])

    # Segmento RFM
    rfm["RFM_Score"] = rfm["R_quartil"].astype(str) + rfm["F_quartil"].astype(str) + rfm["M_quartil"].astype(str)

    # Segmentos
    def segmentar_cliente(r, f, m):
        if r >= 3 and f >= 3 and m >= 3:
            return "Cliente Premium"
        elif r >= 3 and f >= 2:
            return "Leal"
        elif r <= 2 and f <= 2:
            return "Em Risco"
        elif f >= 3:
            return "Potencial"
        else:
            return "Outros"

    rfm["Segmento"] = rfm[["R_quartil", "F_quartil", "M_quartil"]].apply(
        lambda x: segmentar_cliente(int(x[0]), int(x[1]), int(x[2])), axis=1
    )

    return rfm

orders, order_items = carregar_dados_rfm()
rfm_df = calcular_rfm(orders, order_items)

st.header("Análise de RFM (Recência, Frequência e Valor Monetário)")

segmentos = rfm_df["Segmento"].value_counts().reset_index()
segmentos.columns = ["Segmento", "count"]

fig = px.bar(
    segmentos,
    x="Segmento",
    y="count",
    labels={"Segmento": "Segmento", "count": "Número de Clientes"},
    title="Distribuição dos Segmentos de Clientes",
    color="Segmento",
    color_discrete_sequence=px.colors.qualitative.Set3
)


st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Análise por Segmento de Clientes (RFM)

A segmentação RFM classifica os clientes em grupos com base em três fatores: **Recência**, **Frequência** e **Valor Monetário**. Aqui estão os principais grupos:

- **Clientes Premium**: Compram frequentemente, gastam alto e estão ativos recentemente. Devem ser priorizados com ofertas exclusivas.
  
- **Clientes Leais**: Compram com frequência e têm alto gasto, mas podem estar inativos. A reativação com campanhas estratégicas é ideal.

- **Clientes em Potencial**: Clientes com bom valor ou frequência, mas inativos. A reativação pode ser feita com promoções e produtos personalizados.

- **Clientes em Risco**: Compraram muito, mas estão inativos. São ideais para campanhas de recuperação.

- **Clientes Recém-Chegados**: Compraram recentemente, mas com baixo valor e frequência. Reforce a experiência e incentive a recompra.

- **Clientes Inativos**: Não compram há algum tempo. Avalie a viabilidade de reengajá-los com campanhas específicas.

Esta segmentação permite estratégias direcionadas e eficientes para retenção e crescimento.
""")

st.divider()


#%%

st.header("Conclusão")

st.markdown("""
Este projeto teve como objetivo explorar dados de vendas e logística da **Olist**, com o intuito de identificar padrões de comportamento de clientes, avaliar a eficiência logística e sugerir estratégias para melhorar a performance da empresa.

### Principais análises realizadas:
- **Atrasos na entrega x avaliação dos clientes**: revelou que a pontualidade é um fator crucial na experiência do consumidor.
- **Comportamento de compra por região**: identificou variações no volume e ticket médio, sugerindo estratégias de marketing segmentadas.
- **Segmentação de clientes via RFM (Recência, Frequência, Valor Monetário)**: permitiu identificar clientes valiosos e os com maior risco de churn.

### Recomendações para a Olist:
- Melhorar a logística nas regiões com maiores atrasos (ex: Norte e Centro-Oeste) para reduzir o tempo de entrega e aumentar a satisfação dos clientes.
- Criar campanhas específicas para regiões com maior ticket médio (como Norte e Nordeste), incentivando maior frequência de compra.
- Implementar estratégias de retenção para grupos de clientes "em risco", com promoções e vantagens exclusivas.

### Limitações observadas:
- Falta de dados sobre transportadoras.
- Ausência de informações mais granulares sobre o comportamento de navegação dos clientes.

No futuro, seria interessante incluir dados de comportamento online e feedback direto dos consumidores para expandir ainda mais a análise e prever tendências futuras de vendas.

---

### Conclusão final:
Este projeto oferece uma visão detalhada sobre o comportamento do cliente e a performance logística da Olist, servindo como base sólida para **decisões mais informadas e estratégias de melhoria contínua**.
""")


#%%
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: gray;'>
    Desenvolvido por <strong>Adriano Mayco</strong> • 
    <a href='https://www.linkedin.com/in/adriano-mayco-382256221/' target='_blank'>LinkedIn</a> • 
    <a href='https://github.com/AdriMayc' target='_blank'>GitHub</a> • 
    <a href='https://github.com/AdriMayc/Analise_de_Dados.git' target='_blank'>Repositório do Projeto</a><br>
    Maio de 2025 • Versão 1.0
</div>
""", unsafe_allow_html=True)

