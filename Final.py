import pymongo  
import nltk
import sklearn
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from textblob import TextBlob
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  

# Conexão com o MongoDB
cliente = pymongo.MongoClient("mongodb://localhost:27017/")
db = cliente['combate_a_fome']
colecao = db['tweets']
tweets = colecao.find({})

# Baixando e configurando Stopwords e modelo Spacy
nltk.download('stopwords')
palavras_parada = set(stopwords.words('portuguese'))

spacy.cli.download('pt_core_news_sm')  
nlp = spacy.load("pt_core_news_sm")

# Definindo palavras excluídas (curtas e sem relevância semântica)
palavras_excluidas = {"rt", "https", "t", "co", "ue", "vdd", "pq", "kkk", "dem", "a", "é", "ja", "e",  
                      "vc", "ir", "dar", "por", "pra", "vou", "né", "aí", "RT", "pro", "vou", "so", 
                      "mds", "mo", "mó", "lá", "la"}

# Inicializando o analisador de sentimento
analisador_sentimento = SentimentIntensityAnalyzer()

# Inicializando variáveis
corpus = []
sentimentos = []
menções = []

for dados in tweets:
    texto_tweet = dados["text"]
    
    # Capturando menções
    menções_encontradas = re.findall(r'@\w+', texto_tweet)
    menções.extend(menções_encontradas)
    
    # Limpeza do texto
    texto_tweet = re.sub(r'http\S+', '', texto_tweet)  
    texto_tweet = re.sub(r'[^\w\s@]', '', texto_tweet)  
    
    # Lematização e remoção de palavras não informativas
    doc = nlp(texto_tweet)
    tweets_filtrados = [token.lemma_ for token in doc if token.lemma_.lower() not in palavras_parada and token.pos_ != 'VERB']
    tweets_filtrados = [w for w in tweets_filtrados if not any(excluida in w for excluida in palavras_excluidas)]
    
    # Construindo o corpus
    corpus.append(' '.join(tweets_filtrados))  
    
    # Analisando sentimento
    sentimento = analisador_sentimento.polarity_scores(texto_tweet)['compound']
    sentimentos.append(sentimento)

# Vetorização usando TF-IDF
vetorizador = TfidfVectorizer(max_df=0.95, min_df=2)  
corpus_vetorizado = vetorizador.fit_transform(corpus)

# Latent Dirichlet Allocation (LDA) para extração de tópicos
num_topicos = 5  
lda = LatentDirichletAllocation(n_components=num_topicos, random_state=42)
topicos = lda.fit_transform(corpus_vetorizado)

# Agrupando sentimentos por tópico
sentimentos_por_topico = {i: [] for i in range(num_topicos)}
for idx, topico_prob in enumerate(topicos):
    topico_mais_provavel = np.argmax(topico_prob)
    sentimentos_por_topico[topico_mais_provavel].append(sentimentos[idx])

# Calculando a média dos sentimentos por tópico
media_sentimentos_por_topico = {i: np.mean(sentimentos) for i, sentimentos in sentimentos_por_topico.items()}

# Função para exibir palavras principais dos tópicos
def mostrar_palavras_topicas(model, feature_names, num_palavras=10):
    for topico_idx, topico in enumerate(model.components_):
        print(f"Tópico {topico_idx + 1} (Sentimento médio: {media_sentimentos_por_topico[topico_idx]:.2f}):")
        palavras_topicas = [feature_names[i] for i in topico.argsort()[:-num_palavras - 1:-1]]
        print(" ".join(palavras_topicas))
        print()

mostrar_palavras_topicas(lda, vetorizador.get_feature_names_out())

# Função para gerar nuvem de palavras por tópico
def gerar_nuvem_palavras_topicas(model, feature_names, num_palavras=10):
    for topico_idx, topico in enumerate(model.components_):
        palavra_peso = {feature_names[i]: topico[i] for i in topico.argsort()[:-num_palavras - 1:-1]}
        nuvem_palavras = WordCloud(width=400, height=200, background_color="white").generate_from_frequencies(palavra_peso)
        plt.figure(figsize=(5, 2.5))
        plt.imshow(nuvem_palavras, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Tópico {topico_idx + 1} (Sentimento médio: {media_sentimentos_por_topico[topico_idx]:.2f})")
        plt.show()

gerar_nuvem_palavras_topicas(lda, vetorizador.get_feature_names_out())

# Distribuição de polaridade de sentimentos
plt.hist(sentimentos, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Polaridade do Sentimento')
plt.ylabel('Frequência')
plt.title('Distribuição de Polaridade de Sentimentos nos Tweets')
plt.show()

# Contagem de menções e as mais comuns
contagem_menções = Counter(menções)
menções_mais_comuns = contagem_menções.most_common(10)

print("Menções mais frequentes:")
for menção, contagem in menções_mais_comuns:
    print(f"{menção}: {contagem}")

# Gráfico de barras das menções mais comuns
labels, values = zip(*menções_mais_comuns)
indexes = np.arange(len(labels))

plt.bar(indexes, values, color='skyblue')
plt.xticks(indexes, labels, rotation=45)
plt.xlabel('Menções')
plt.ylabel('Frequência')
plt.title('Menções Mais Frequentes nos Tweets')
plt.show()

# Calculando perplexidade dos tópicos
perplexidade = lda.perplexity(corpus_vetorizado)
print(f'Perplexidade dos Tópicos: {perplexidade}')

# Exemplo de classificação de sentimentos
tweets_rotulados = [
    {"texto": "Só pra complementar a sua fala, uma fala do Churchill: “Eu odeio indianos. Eles são um povo bestial com uma religião bestial. A fome foi culpa deles por se reproduzirem como coelhos.”", "sentimento": -0.8},
    {"texto": "Maduro se lixou e meteu mais de 34 mil garimpeiros para explorar ouro no lado venezuelano. Índios começaram a fugir da fome e vieram para o lado brasileiro", "sentimento": -0.1},
    {"texto": "Não precisa torcer…JÁ DEU ERRADO 🥹 Infelizmente talvez seja necessário isso pra que mais gente acorde…mas algumas pessoas nem se morrerem de fome irão acordar…como ocorreu com adoradores de Stálin", "sentimento": -0.3666666666666667},
    {"texto": "Eu me comovo sim não está vendo a vestimenta vermelha são índios Venezuelanos estão vindo para o  Brasil porque Maduro está matando a população de fome ou vc não sabe que Roraima é fronteira com a Venezuela para de se fazer de inocente e ficar defendendo esse governo politiqueiro", "sentimento": -0.125},
    {"texto": "é mesmo, e por que ao invés de fazer obra na argentina ele não acaba com a Fome aqui do pais. vocês todos não passam de ladrões.", "sentimento": -0.75},
    {"texto": "O cara ainda é um esquerdista homofóbico velho. Vai lá bater palmas pra o presidente que enquanto o povo de seu país passa fome vai gastar dinheiro em obras nos países dos brothers dele", "sentimento": -0.4},
]  

def calcular_polaridade(texto):
    return TextBlob(texto).sentiment.polarity

# Classificação de sentimentos usando TextBlob
y_true = [tweet['sentimento'] for tweet in tweets_rotulados]  
y_pred = []

for tweet in tweets_rotulados:
    polaridade = calcular_polaridade(tweet['texto'])
    if polaridade > 0:
        y_pred.append(1)  
    elif polaridade < 0:
        y_pred.append(-1) 
    else:
        y_pred.append(0)  

accuracy = accuracy_score(y_true, y_pred)
print(f'Acurácia da Análise de Sentimentos: {accuracy:.2f}')

# Relatório de classificação
print(classification_report(y_true, y_pred, target_names=['Negativo', 'Neutro', 'Positivo']))

# Heatmap de sentimentos por tópico
sentimentos_por_topico_matriz = np.zeros((num_topicos, 3))

for i, sentimentos in sentimentos_por_topico.items():
    positivos = sum(1 for s in sentimentos if s > 0)
    negativos = sum(1 for s in sentimentos if s < 0)
    neutros = sum(1 for s in sentimentos if s == 0)
    sentimentos_por_topico_matriz[i] = [positivos, neutros, negativos]

plt.figure(figsize=(12, 8))
sns.heatmap(sentimentos_por_topico_matriz, annot=True, fmt='d', cmap='coolwarm', 
            xticklabels=['Positivos', 'Neutros', 'Negativos'], yticklabels=[f'Tópico {i+1}' for i in range(num_topicos)])
plt.title('Distribuição de Sentimentos por Tópico')
plt.xlabel('Tipo de Sentimento')
plt.ylabel('Tópico')
plt.show()