   # Projeto de Análise de Big Data: Mercado de Criptomoedas no Brasil

Este repositório contém o desenvolvimento de um projeto focado na análise de dados do mercado de ativos digitais. O trabalho integra diferentes fontes de informação para compreender a relação entre o comportamento de preços, o interesse do público e a demografia dos investidores no contexto brasileiro.

## Objetivos do Projeto
O estudo visa identificar padrões de correlação entre as variações de preço dos principais ativos (Bitcoin e Ethereum) e o volume de buscas registrado no Google Trends. Adicionalmente, o projeto aplica técnicas de tratamento de dados para analisar a evolução do perfil dos investidores, focando em indicadores de escolaridade e representatividade de gênero entre os anos de 2021 e 2023.

## Funcionalidades e Metodologia
A análise é estruturada através dos seguintes eixos:

1. Processamento de Séries Temporais: Limpeza e organização de dados históricos de preços.
2. Visualização de Dados: Geração de gráficos interativos utilizando a biblioteca Plotly, incluindo análises em escala logarítmica para comparação de ativos.
3. Estimativa Demográfica: Uso de métodos de interpolação para preenchimento de lacunas em dados estatísticos sobre o perfil do investidor.

## Tecnologias Utilizadas
* Linguagem de programação: Python 3.
* Ambiente de desenvolvimento: PyCharm.
* Principais bibliotecas: Pandas para manipulação de dados e Plotly para visualizações.

## Estrutura de Arquivos
* Trabalho de Bigdata.py: Código fonte principal com a lógica de análise.
* data/: Diretório contendo as bases de dados em formato CSV.
* grafico_4_clusters_kmeans_final.html: Resultado visual da análise de agrupamento.
* .gitignore: Configuração para exclusão de arquivos temporários e do ambiente virtual.

## Instruções de Uso
Para executar este projeto localmente, é necessário ter o Python instalado. Recomenda-se a criação de um ambiente virtual para a instalação das dependências (pandas e plotly). O script principal pode ser executado diretamente através do comando: python "Trabalho de Bigdata.py".