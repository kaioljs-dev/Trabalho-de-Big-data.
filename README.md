
# Análise de Dados: Cripto-Investidores e Tendências no Brasil

Este projeto foi desenvolvido como parte do **Trabalho de Big Data**. Ele consiste em uma análise integrada que cruza o comportamento de preços de criptoativos (BTC e ETH), o volume de interesse de busca no Google Trends e o perfil demográfico do investidor brasileiro.

## Objetivo do Projeto
O objetivo é identificar correlações entre as flutuações de mercado e o interesse do público geral, além de mapear a evolução do perfil do investidor (gênero e escolaridade) através de técnicas de interpolação de dados.

## Visualizações e Análises
O projeto gera três visualizações principais utilizando a biblioteca **Plotly**:

1.  **Preços Históricos (BTC/ETH):** Utilização de escala logarítmica para permitir a comparação direta da performance percentual entre ativos com valores nominais muito distantes.
2.  **Série Histórica Google Trends:** Visualização do interesse relativo de busca pelo termo "criptomoedas" no Brasil, evidenciando os picos de busca em momentos de alta do mercado.
3.  **Evolução do Perfil Demográfico:** Gráfico de linhas com interpolação linear para estimar a mudança na participação feminina e no nível de escolaridade dos investidores entre os anos de 2021 e 2023.

## Tecnologias e Ferramentas
* **Linguagem:** Python 3.x
* **IDE:** PyCharm
* **Bibliotecas de Dados:** * `Pandas`: Manipulação, limpeza e interpolação de séries temporais.
    * `Plotly`: Criação de gráficos dinâmicos e interativos.
* **Versionamento:** Git & GitHub

## Estrutura do Repositório
* `main.py`: Script principal com o código de análise e geração de gráficos.
* `trends_data.csv`: Base de dados real exportada do Google Trends.
* `.gitignore`: Arquivo de configuração para excluir pastas de ambiente virtual (`venv`) e da IDE (`.idea`).
* `LICENSE`: Licença MIT permitindo o uso acadêmico e profissional do código.

## Como Executar
1. Clone o repositório:
   ```bash
   git clone [https://github.com/kaioljs-dev/Trabalho-de-Big-data.git](https://github.com/kaioljs-dev/Trabalho-de-Big-data.git)
