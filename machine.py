from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Carregando o dataset iris
iris = datasets.load_iris()
features = iris.data
targets = iris.target

# Somando as quatro features em uma única coluna
featuresAll = []
for observation in features:
    featuresAll.append([observation[0] + observation[1] + observation[2] + observation[3]])

# Inicializando o modelo de regressão linear
model = LinearRegression()

# Ajustando o modelo com os dados de featuresAll (soma das features) e targets
model.fit(featuresAll, targets)

# Fazendo previsões
predictions = model.predict(featuresAll)

# Pegando os coeficientes
coef_angular = model.coef_[0]  # Coeficiente angular (inclinação)
coef_linear = model.intercept_  # Coeficiente Linear (intercepto)

# Exibindo os coeficientes
print(f"Coeficiente angular (slope): {coef_angular}")
print(f"Coeficiente linear (intercepto): {coef_linear}")

# Plotando o gráfico de dispersão
plt.scatter(featuresAll, targets, color='red', alpha=1.0, label='Dados reais')
plt.plot(featuresAll, predictions, color='blue', label=f'Regressão Linear (y = {coef_angular}x + {coef_linear})')

# Customizações do gráfico
plt.rcParams['figure.figsize'] = [10, 8]
plt.title('Iris Dataset scatter Plot com Regressão Linear')
plt.xlabel('Soma das Features')
plt.ylabel('Targets')
plt.legend()

# Mostrando o gráfico
plt.show()

# Fazendo uma previsão para um novo valor
novo_valor = [[15.0]]
y_pred = model.predict(novo_valor)
print(f"\nPara a soma das features {novo_valor[0][0]}, o valor previsto de y é: {y_pred[0]}")
