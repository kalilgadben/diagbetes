import pandas as pd
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import shap

dados = pd.read_csv("diabetes.csv")
dados.rename(columns={'Pregnancies': 'gestacoes', 'Glucose': 'glicose', 'BloodPressure': 'pressao_arterial'}, inplace=True)
dados.rename(columns={'SkinThickness': 'espessura_pele', 'Insulin': 'insulina', 'BMI': 'imc'}, inplace=True)
dados.rename(columns={'DiabetesPedigreeFunction': 'predisposicao_genetica_diabetes', 'Age': 'idade', 'Outcome': 'diabetes'}, inplace=True)
dados.head()

dados.describe()

msno.matrix(dados)
sb.histplot(data = dados, x = "glicose")

matriz_correlacao = dados.corr().round(2)
fig, ax = plt.subplots(figsize=(8,8))
sb.heatmap(data=matriz_correlacao, annot=True, linewidths=5, ax = ax)

x = dados[["gestacoes", "glicose", "insulina", "imc", "predisposicao_genetica_diabetes","idade"]]
y = dados["diabetes"]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 13)

erros = []

for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_treino, y_treino)
    predicao_i = knn.predict(x_teste)
    erros.append(np.mean(predicao_i != y_teste))

plt.figure(figsize=(12,6))
plt.plot(range(1,10), erros, color = "red", linestyle="dashed", marker="o", markerfacecolor="blue", markersize = 10)
plt.title("Erro médio para K")
plt.xlabel("Valor de K")
plt.ylabel("Erro médio")

modelo_classificador_knn = KNeighborsClassifier(n_neighbors = 5)
modelo_classificador_knn.fit(x_treino, y_treino)
predicao_knn_y = modelo_classificador_knn.predict(x_teste)

aruracia_knn = accuracy_score(y_true = y_teste, y_pred = predicao_knn_y)
print("Acurácia KNN: ", aruracia_knn)

svm = Pipeline(
    [
        ("linear_svc", LinearSVC(C = 1))
    ]
)

svm.fit(x_treino, y_treino)

predicao_svm_y = svm.predict(x_teste)

aruracia_svm = accuracy_score(y_true = y_teste, y_pred = predicao_svm_y)
print("Acurácia SVM: ", aruracia_svm)

matriz_confusao_knn = confusion_matrix(y_teste, predicao_knn_y)
plt.figure(figsize = (8,4))
sb.heatmap(matriz_confusao_knn, annot = True, fmt = "d", cmap = "Blues")
plt.xlabel("Predição KNN")
plt.ylabel("Dados Reais")

matriz_confusao_svm = confusion_matrix(y_teste, predicao_svm_y)
plt.figure(figsize = (8,4))
sb.heatmap(matriz_confusao_svm, annot = True, fmt = "d", cmap = "Blues")
plt.xlabel("Predição SVM")
plt.ylabel("Dados Reais")

print("Resultado KNN")
print(classification_report(y_teste, predicao_knn_y))
print("Resultado SVM")
print(classification_report(y_teste, predicao_svm_y))

explainer = shap.KernelExplainer(modelo_classificador_knn.predict_proba, x[:100])
shap_values = explainer.shap_values(x_teste[:100])
shap.summary_plot(shap_values[:], x_teste[:100], feature_names=x.columns)