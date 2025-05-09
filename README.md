# Análise de Sentimentos e Tópicos em Tweets sobre Fome
Este projeto realiza uma análise de sentimentos e tópicos em tweets relacionados ao combate à fome. Utilizando ferramentas de NLP (Processamento de Linguagem Natural), o código conecta-se a um banco MongoDB contendo tweets, limpa e processa os textos, identifica menções, analisa sentimentos com VADER e TextBlob, extrai tópicos com LDA, e apresenta os resultados por meio de gráficos e nuvens de palavras.

## Tecnologias e Bibliotecas Utilizadas
- Python;
- MongoDB + pymongo;
- nltk, spacy, textblob, vaderSentiment;
- sklearn (TF-IDF, LDA, métricas de avaliação);
- wordcloud, matplotlib, seaborn;
- numpy, re, collections.

## Funcionalidades
- Conexão com banco de dados MongoDB contendo tweets;
- Pré-processamento textual (limpeza, lematização, remoção de stopwords);
- Detecção de menções (@usuários) e contagem das mais frequentes;
- Análise de sentimentos com VADER e avaliação com TextBlob;
- Vetorização com TF-IDF;
- Extração de tópicos com LDA;
- Visualização das palavras mais relevantes por tópico;
- Geração de nuvens de palavras;
- Gráficos de distribuição de sentimentos e menções;
- Heatmap de sentimentos por tópico.

## Saídas Geradas
- Média de sentimento por tópico;
- Nuvens de palavras por tópico;
- Distribuição de polaridade dos sentimentos;
- Top 10 menções mais frequentes;
- Avaliação de acurácia da análise de sentimentos com classification_report;
- Heatmap de distribuição de sentimentos por tópico;
