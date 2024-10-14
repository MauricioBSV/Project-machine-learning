import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
import numpy as np

# Carregando o dataset de vinhos
wine_data = load_wine()

# Convertendo para um DataFrame do pandas
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Incluindo a variável target (classe do vinho)
df['target'] = wine_data.target

# Exibindo as primeiras linhas do dataset
print(df.head())

# Escolhendo algumas features numéricas e adicionando ruído
features = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash']]

# Somando as features e adicionando ruído aleatório para separação
noise = np.random.normal(0, 2, size=features.shape[0])  # Adicionando ruído aleatório com média 0 e desvio padrão 2
featuresAll = features.sum(axis=1).values + noise  # Adicionando o ruído à soma das features
featuresAll = featuresAll.reshape(-1, 1)  # Redimensionando para o formato correto

# A variável target é a coluna 'target'
target = df['target']

# Inicializando o modelo de regressão linear
model = LinearRegression()

# Ajustando o modelo com os dados de featuresAll (soma das features + ruído) e target
model.fit(featuresAll, target)

# Fazendo previsões
predictions = model.predict(featuresAll)

# Pegando os coeficientes e ajustando para "alinhar" melhor a linha com os pontos
coef_angular = model.coef_[0]  # Coeficiente angular (inclinação)
coef_linear = model.intercept_  # Coeficiente linear (intercepto)

# Deslocando a linha de regressão para tocar na linha central
coef_linear -= 0.2  # Ajustando o intercepto para descer a linha e tocar mais a linha do meio

# Recalculando as previsões com o novo intercepto ajustado
predictions = coef_angular * featuresAll + coef_linear

# Exibindo os coeficientes ajustados
print(f"Coeficiente angular (slope): {coef_angular}")
print(f"Coeficiente linear ajustado (intercepto): {coef_linear}")

# Definindo as cores para cada classe (target)
colors = {0: 'red', 1: 'yellow', 2: 'green'}

# Plotando o gráfico de dispersão com cores diferentes para cada classe e pontos mais espaçados
for wine_class in df['target'].unique():
    class_data = featuresAll[df['target'] == wine_class]
    plt.scatter(class_data, target[df['target'] == wine_class], 
                color=colors[wine_class], 
                alpha=0.6, 
                label=f'Classe {wine_class}')

# Adicionando a linha de regressão ajustada
plt.plot(featuresAll, predictions, color='blue', label=f'Regressão Linear Ajustada (y = {coef_angular:.2f}x + {coef_linear:.2f})')

# Customizações do gráfico
plt.rcParams['figure.figsize'] = [10, 8]
plt.title('Wine Dataset Scatter Plot com Linha de Regressão Ajustada')
plt.xlabel('Soma das Features com Ruído')
plt.ylabel('Targets')
plt.legend()

# Mostrando o gráfico
plt.show()

# Fazendo uma previsão para um novo valor
novo_valor = [[30.0]]  # Exemplo: soma das features é 30
y_pred = model.predict(novo_valor)
print(f"\nPara a soma das features {novo_valor[0][0]}, o valor previsto de y é: {y_pred[0]}")
