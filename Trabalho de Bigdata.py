# =========================================================================
# BLOCO 0: CONFIGURAÇÃO INICIAL E IMPORTAÇÕES
# =========================================================================
# Este bloco configura o ambiente de trabalho e conecta as três principais ferramentas e define
# as dependências e estabelece as variáveis chave.
import sys 

# Importações de Bibliotecas:
import findspark
# SparkSession é o ponto de entrada para toda a funcionalidade do Spark.
from pyspark.sql import SparkSession
# Funções do Spark SQL para manipulação de colunas e dados.
from pyspark.sql.functions import col, to_date, regexp_replace, avg, lit
# Módulos de Machine Learning do Spark para pré-processamento e clustering. sbc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Bibliotecas Python padrão para gestão de arquivos e caminhos.
# =========================================================================

import glob
#A biblioteca glob é usada para encontrar caminhos de arquivo que correspondam
# a um padrão específico usando caracteres wildcard (como * e **).

import os
#A biblioteca os (de Operating System) fornece uma maneira de interagir com o
# sistema operacional no qual o Python está rodando (macOS, Windows, Linux).

#Em resumo, glob e os são a camada de organização de arquivos que permite que suas bibliotecas de
# análise (Pandas e Spark) encontrem os dados para trabalhar.

# Pandas é crucial para manipulação eficiente de dados em memória e para o Gráfico 1.
import pandas as pd

# Plotly é usado para gerar gráficos interativos.
import plotly.express as px
import plotly.graph_objects as go

# =========================================================================

# Inicializa o findspark para localizar a instalação do Spark no sistema.
findspark.init()
# Cria a sessão Spark, que será usada para processamento distribuído.
spark = SparkSession.builder.appName("AnaliseCriptoML").getOrCreate()
print("Sessão Spark inicializada com sucesso.")

# =========================================================================

# --- VARIÁVEIS DE CONFIGURAÇÃO ---
# Define os parâmetros que podem ser facilmente ajustados, como nomes de arquivos e caminhos.
NOME_ARQUIVO_PRECOS = 'bitcoin.csv'
NOME_ARQUIVO_TRENDS = 'multiTimeline.csv'
CAMINHO_PASTA_PRECOS = '/Users/luiz/PycharmProjects/PythonProject/data/Top 100 Crypto Coins'
CAMINHO_PASTA_TRENDS = '/Users/luiz/PycharmProjects/PythonProject/data'
CRIPTO_ALVO = 'BTC'
K_CLUSTERS = 4

# Função auxiliar para garantir que os arquivos sejam encontrados, independente do subdiretório.
def encontrar_caminho(base_path, nome_arquivo):
    caminho_completo = os.path.join(base_path, nome_arquivo)
    if os.path.exists(caminho_completo):
        return caminho_completo
    caminhos = glob.glob(f'{base_path}/**/{nome_arquivo}', recursive=True)
    return caminhos[0] if caminhos else None



# =========================================================================
# BLOCO 1: CARREGAMENTO DE PREÇOS (PANDAS FIRST) E GERAÇÃO DO GRÁFICO 1
# =========================================================================
# Usamos o Pandas primeiro para manipular os dados de preço. Isso garante
# a tipagem correta da data, resolvendo o problema de visualização do Plotly.

print("\n--- 1. PREPARAÇÃO DE DADOS (PANDAS FIRST) E GRÁFICO 1 ---")

# --- 1.1. Carregamento e Limpeza no Pandas ---

caminho_preco = encontrar_caminho(CAMINHO_PASTA_PRECOS, NOME_ARQUIVO_PRECOS)
df_pandas_preco = pd.DataFrame()

if caminho_preco:
    try:
        # 1. Carrega o CSV diretamente no Pandas.
        df_pandas_preco = pd.read_csv(caminho_preco)

        # 2. TIPAGEM CRÍTICA: Converte 'Date' para datetime. 'errors=coerce' substitui
        # datas inválidas por NaT (Not a Time), facilitando a limpeza.
        df_pandas_preco['Date'] = pd.to_datetime(df_pandas_preco['Date'], errors='coerce')
        df_pandas_preco['Close'] = pd.to_numeric(df_pandas_preco['Close'], errors='coerce')

        # 3. LIMPEZA: Remove todas as linhas onde a data ou o preço estão nulos.
        # ISSO FOI CRUCIAL: Plotly falha ao plotar séries temporais com valores NaT.
        df_pandas_preco.dropna(subset=['Date', 'Close'], inplace=True)

        # 4. Ajustes finais de colunas
        df_pandas_preco = df_pandas_preco.rename(columns={'Close': 'Close_Price'})
        df_pandas_preco['Symbol'] = CRIPTO_ALVO
        df_pandas_preco = df_pandas_preco[['Date', 'Close_Price', 'Symbol']].drop_duplicates()

        print(f"-> Preços (Pandas) limpos para plotagem: {len(df_pandas_preco)} linhas.")

    except Exception as e:
        print(f"ERRO ao carregar/limpar dados de preço no Pandas: {e}.")

# --- 1.2. Geração do Gráfico 1 (Evolução Anual) ---

try:
    if not df_pandas_preco.empty:

        # PASSO DE FILTRAGEM: Cria um sub-DataFrame contendo apenas os dados de 1º de Janeiro de cada ano.
        df_pandas_plot_anual = df_pandas_preco[
            (df_pandas_preco['Date'].dt.month == 1) &
            (df_pandas_preco['Date'].dt.day == 1)
            ].copy()

        if not df_pandas_plot_anual.empty:
            # PRIMEIRO GRAFICO de linha com marcadores para destacar os pontos anuais.
            fig_preco = px.line(df_pandas_plot_anual, x='Date', y='Close_Price', color='Symbol',
                                title='1. Evolução de Preço de Fechamento (BTC) - PONTOS ANUAIS (1º Jan)',
                                labels={'Close_Price': 'Preço de Fechamento (USD) [Log Scale]', 'Date': 'Data'},
                                log_y=True,
                                markers=True)

            # Formatação do eixo X para uma visualização limpa:
            fig_preco.update_layout(
                xaxis=dict(
                    type='date',  # Força o tipo de eixo como data.
                    tickformat="%Y",  # Rótulos mostram apenas o ano.
                    dtick="M12",  # Ticks (marcas) a cada 12 meses (anual).
                    tickangle=0,
                    showgrid=True
                )
            )

            fig_preco.show()
            print("Gráfico 1 (Preços) gerado com sucesso com filtro anual.")
        else:
            print("Gráfico 1: Não gerado. DataFrame anual ficou vazio após filtragem.")
    else:
        print("Gráfico 1: Não gerado. DataFrame de Preço está vazio.")
except Exception as e:
    print(f"Erro ao gerar Gráfico 1: {e}")



# =========================================================================
# BLOCO 2: CARREGAMENTO SPARK E ETL PARA ML (Extração e Transformação)
# =========================================================================
# Os dados de preço (já limpos) são agora passados para o Spark.
# Os dados de Trends são carregados e transformados diretamente no Spark para ETL.

print("\n--- 2. CARREGAMENTO SPARK E ETL ---")

# --- 2.1. Conversão de Preços Pandas para Spark ---

if not df_pandas_preco.empty:
    # 1. Cria o DataFrame Spark a partir do Pandas limpo.
    df_spark_preco = spark.createDataFrame(df_pandas_preco)
    # 2. Garante que os tipos finais do Spark estejam corretos.
    df_spark_preco = df_spark_preco.select(
        to_date(col("Date")).alias("Date"),
        col("Close_Price").cast("float"),
        col("Symbol")
    )
    print(f"-> Preços convertidos para Spark: {df_spark_preco.count()} linhas.")
else:
    df_spark_preco = spark.createDataFrame([], "Date: date, Close_Price: float, Symbol: string")
    #Transfere os dados limpos do Pandas para o DataFrame Spark, preparando-os para o processamento distribuído.

# --- 2.2. Carregamento dos Dados do GOOGLE TRENDS ---

caminho_trends = encontrar_caminho(CAMINHO_PASTA_TRENDS, NOME_ARQUIVO_TRENDS)
df_spark_trends = spark.createDataFrame([], "Date: date, Interesse_0_100: float")

try:
    # Uso do Pandas para ignorar a primeira linha do CSV de Trends (metadados).
    df_trends_pandas = pd.read_csv(caminho_trends, header=1, encoding='utf-8')
    df_spark_trends_raw = spark.createDataFrame(df_trends_pandas)

    # 1. Renomeação de Colunas
    nome_coluna_interesse = df_spark_trends_raw.columns[1]
    df_spark_trends = df_spark_trends_raw.withColumnRenamed(df_spark_trends_raw.columns[0], "Date_String")
    df_spark_trends = df_spark_trends.withColumnRenamed(nome_coluna_interesse, "Interesse_String")

    # 2. Limpeza de Strings: Substitui o valor "<1" (menor que 1%) por "0" e converte para float.
    df_spark_trends = df_spark_trends.withColumn(
        "Interesse_Num",
        regexp_replace(col("Interesse_String"), "<1", "0").cast("float")
    )
    # 3. Conversão de Data: Converte a string YYYY-MM para o tipo Date do Spark.
    df_spark_trends = df_spark_trends.select(
        to_date(col("Date_String"), "yyyy-MM").alias("Date"),
        col("Interesse_Num").alias("Interesse_0_100")
    )
    print(f"-> Trends carregado: {df_spark_trends.count()} linhas.")
except Exception as e:
    print(f"AVISO/ERRO ao carregar Trends: {e}. Pulando Trends.")

# --- 2.3. Agregação e Junção de Dados (Preparação para ML) ---

print("2.3. Agregando e unindo dados...")

    # Exemplo de lazy evaluation quando o spark separa os comandos na memoria para
    # executa-los de uma vez otimizando o processo

    #O Spark planeja a junção (JOIN) dos dois DataFrames (Preço Agregado e Trends) com base na data do mês.

if df_spark_preco.count() > 0:
    # 1. Agregação Mensal de Preços: Calcula a média do preço diário para obter um preço mensal.
    df_spark_preco_btc_mensal = df_spark_preco.filter(col("Symbol") == CRIPTO_ALVO).withColumn(
        "Date_Month", to_date(col("Date"), "yyyy-MM")
    ).groupBy("Date_Month").agg(avg("Close_Price").alias("Avg_Close_Price"))
    #O Spark planeja e executa o cálculo da média de preço mensal (avg), agregando milhares
    # de linhas diárias de forma distribuída.

    # 2. Junção (JOIN): Combina os dados de preço (mensal) e trends (mensal) pela data do mês.
    df_spark_joined = df_spark_preco_btc_mensal.join(
        df_spark_trends,
        df_spark_preco_btc_mensal.Date_Month == df_spark_trends.Date,
        "inner"
    ).select(
        col("Date").alias("Mes"),
        col("Avg_Close_Price"),
        col("Interesse_0_100")
    )

    # metodo .count() é uma Ação que força o Spark a executar todas as transformações pendentes
    # (agregação e junção) e retorna o resultado final (o número de linhas) ao programa Driver.

    if df_spark_joined.count() > 0:
        print(f"-> Dados consolidados para ML: {df_spark_joined.count()} linhas.")
    else:
        print("AVISO: DataFrame consolidado está vazio. Pulando Bloco ML.")
else:
    df_spark_joined = spark.createDataFrame([], "Mes: date, Avg_Close_Price: float, Interesse_0_100: float")



# =========================================================================
# BLOCO 3: MACHINE LEARNING (K-MEANS CLUSTERING)
# =========================================================================
# Aplica um algoritmo de aprendizado de máquina não supervisionado para
# segmentar os meses com base em preço e interesse de busca.

print("\n--- 3. MACHINE LEARNING (K-MEANS) ---")

if df_spark_joined.count() > 0:
    # 3.1. Pré-processamento: VectorAssembler
    # O K-Means exige que todas as features numéricas estejam em uma única coluna VETORIAL.

    feature_cols = ["Avg_Close_Price", "Interesse_0_100"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_spark_ml = assembler.transform(df_spark_joined).dropna()
    # Converte as features (Preço e Interesse) em um vetor único.

    # 3.2. Treinamento do Modelo K-Means

    kmeans = KMeans(featuresCol="features", k=K_CLUSTERS, seed=1)
    model = kmeans.fit(df_spark_ml)
    print(f"-> Modelo K-Means treinado com K={K_CLUSTERS}.")
    #O comando fit() é a Ação mais pesada do ML. O Spark distribui o cálculo do algoritmo K-Means
    # entre seus executores para encontrar os melhores centros de cluster (K=4).

    # 3.3. Aplicação e Análise dos Clusters

    # Aplica o modelo treinado para adicionar a coluna prediction (o ID do cluster) a cada mês.
    df_spark_results = model.transform(df_spark_ml)

    # Análise de Resultados: Contagem de meses por cluster.
    print("\n--- CONTAGEM DE PERIODOS POR CLUSTER ---")
    df_spark_results.groupBy("prediction").count().orderBy("prediction").show()

    # Força a execução dos cálculos para exibir os centróides (as médias) de cada cluster,
    # permitindo a interpretação dos regimes de mercado.
    print("\n--- CENTROIDES E CARACTERÍSTICAS DOS CLUSTERS ---")
    df_cluster_summary = df_spark_results.groupBy("prediction").agg(
        avg("Avg_Close_Price").alias("Media_Preco"),
        avg("Interesse_0_100").alias("Media_Interesse"),
        lit(K_CLUSTERS).alias("K_Clusters")
    )
    df_cluster_summary.orderBy("prediction").show()
else:
    print("Pulando Bloco 3: Não há dados suficientes para o K-Means.")



# =========================================================================
# BLOCO 4: GERAÇÃO DE GRÁFICOS (RESTANTES)
# O foco volta para a visualização, exigindo que os resultados distribuídos
# do Spark sejam recolhidos para o ambiente local do Pandas.
# =========================================================================

print("\n--- 4. GERAÇÃO DE GRÁFICOS (RESTANTES) ---")

# --- 4.1. Gráfico 2: Interesse do Google Trends ---

try:
    # O Spark executa esta Ação para mover os dados processados da memória
    # distribuída para a memória local do Pandas, onde o Plotly pode acessá-los.
    df_pandas_trends = df_spark_trends.toPandas()

    if not df_pandas_trends.empty:
        df_pandas_trends['Date'] = pd.to_datetime(df_pandas_trends['Date'])
        df_pandas_trends.rename(columns={'Interesse_0_100': 'Interesse (0-100)'}, inplace=True)

       # Gera o Gráfico 2 (Trends), agora com dados recolhidos.
        fig_trends = px.line(df_pandas_trends, x='Date', y='Interesse (0-100)',
                             title='2. Interesse de Busca por "Criptomoedas" no Brasil (Dados Reais)',
                             labels={'Interesse (0-100)': 'Interesse Relativo (0-100)', 'Date': 'Data'},
                             markers=False)
        fig_trends.update_yaxes(range=[0, 105], title='Interesse Relativo (0-100)')

        # Formatação do eixo X para exibição mensal/anual limpa
        fig_trends.update_layout(xaxis=dict(type='date', tickformat="%Y-%m", dtick="M12"))

        fig_trends.show()
        print("Gráfico 2 (Trends) gerado com sucesso.")
    else:
        print("Gráfico 2: Não gerado. DataFrame de Trends está vazio.")
except Exception as e:
    print(f"Erro ao gerar Gráfico 2: {e}")

# --- 4.2. Gráfico 3: Perfil Estimado (Interpolação) ---
# Cria um DataFrame de exemplo de perfil e usa interpolação para preencher dados ausentes.

dados_perfil = {
    'Ano': [2021, 2023, 2021, 2023],
    'Perfil': ['Homens', 'Homens', 'Mulheres', 'Mulheres'],
    'Percentual (%)': [94, 91, 6, 9]
}
df_perfil_estimativa = pd.DataFrame(dados_perfil)

# Usa a funcionalidade avançada do Pandas (interpolação) para criar os dados simulados do Gráfico 3.
# Interpolação Linear: Estima o valor para 2022 com base nos dados de 2021 e 2023.
df_perfil_final = (
    df_perfil_estimativa.set_index('Ano')
    .groupby('Perfil')['Percentual (%)']
    .apply(lambda x: x.reindex(range(2021, 2024)).interpolate(method='linear'))
    .reset_index()
    .rename(columns={'level_0': 'Perfil', 'level_1': 'Ano'})
)

fig_perfil_est = px.line(df_perfil_final, x='Ano', y='Percentual (%)', color='Perfil',
                         markers=True, title='3. Evolução ESTIMADA do Perfil por Gênero (2021-2023) - INTERPOLADO')
fig_perfil_est.show()
print("Gráfico 3 (Perfil) gerado com sucesso.")

# --- 4.3. Gráfico 4: Visualização dos Clusters K-Means (SALVANDO HTML) ---
print("\n--- 4.3. Gráfico 4: Visualização dos Clusters K-Means (Scatter Plot) ---")

try:
    # 1. VERIFICAÇÃO E CONVERSÃO (Spark para Pandas)
    if df_spark_results is not None and df_spark_results.count() > 0:

        df_pandas_results = df_spark_results.toPandas()
        df_centroids_pandas = df_cluster_summary.toPandas()

        # -------------------------------------------------------------------------
        # ADIÇÃO CHAVE 1: Mapeamento de nomes para os clusters
        # -------------------------------------------------------------------------
        # Definindo as descrições dos regimes de mercado, usando os IDs de cluster (0 a 3)
        cluster_names = {
            # Baseado na análise de onde os centróides caem no gráfico e na tabela
            '0': 'Preço Alto/Interesse Moderado (Cluster 0)',
            '1': 'Baixo Preço/Baixo Interesse (Cluster 1)',
            '2': 'Crescimento/Interesse Intermediário (Cluster 2)',
            '3': 'Pico de Euforia/Máximo Preço (Cluster 3)'
        }

        # Mapeia os números do cluster para os nomes descritivos
        df_pandas_results['Regime de Mercado'] = df_pandas_results['prediction'].astype(str).map(cluster_names)
        df_centroids_pandas['Regime de Mercado'] = df_centroids_pandas['prediction'].astype(str).map(cluster_names)

        # O Plotly usará esta nova coluna como cor e na legenda

        # 2. GERAÇÃO DO GRÁFICO (Scatter Plot)
        fig_kmeans = px.scatter(df_pandas_results,
                                x='Avg_Close_Price',
                                y='Interesse_0_100',
                                color='Regime de Mercado',  # Usa o nome descritivo para colorir
                                hover_data=['Mes'],
                                title='4. Segmentação de Regimes de Mercado (K-Means) [Escala Logarítmica no Preço]',
                                labels={'Avg_Close_Price': 'Preço Médio Mensal (USD) [Log Scale]',
                                        'Interesse_0_100': 'Interesse de Busca (0-100)',
                                        'Regime de Mercado': 'Regime de Mercado'},
                                log_x=True)  # Mantém a escala logarítmica para clareza

        # 3. ADIÇÃO DOS CENTRÓIDES (Pontos Centrais)
        fig_kmeans.add_trace(go.Scatter(
            x=df_centroids_pandas['Media_Preco'],
            y=df_centroids_pandas['Media_Interesse'],
            mode='markers',
            name='Centróides',
            marker=dict(
                size=15,
                color='black',
                symbol='x',  # Usa o símbolo 'X' para destacar o centro do cluster
                line=dict(width=2, color='DarkSlateGrey')
            ),
            # Adiciona hovertext nos centróides para fácil identificação
            hovertext=df_centroids_pandas['Regime de Mercado'],
            hoverinfo='text'
        ))

        # 4. SALVAMENTO EM ARQUIVO HTML
        local_atual = os.getcwd()
        print(f"DEBUG: O arquivo HTML será salvo no diretório: {local_atual}")

        # Renomeia o arquivo para a versão mais recente
        caminho_grafico_4 = 'grafico_4_clusters_kmeans_final.html'

        fig_kmeans.write_html(caminho_grafico_4)

        print(f"Gráfico 4 (K-Means) gerado e salvo com sucesso. Procure pelo arquivo: {caminho_grafico_4}")
        print("-" * 50)

    else:
        print("Gráfico 4: Não gerado. Resultados do K-Means vazios.")
except Exception as e:
    print(f"Erro ao gerar Gráfico 4: {e}")


# =========================================================================
# --- 4.4. Encerramento ---
# Finaliza a sessão Spark para liberar todos os recursos de memória e processamento.
# O 'spark.stop()' é uma Ação que encerra o cluster e é crucial para o bom funcionamento.

spark.stop()
print("\n--- FIM DA EXECUÇÃO. Sessão Spark encerrada. ---")
sys.exit()