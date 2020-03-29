import re

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Binarizer

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from tensorflow.keras.metrics import Precision, AUC, Accuracy

"""
Função para a simplificação das equações e passos
"""
def template_equacao(equ):
    i = 0
    equ = equ.lower().replace("x", "Zx")
    for m in re.finditer(r"[\d]{1,}Zx{1}", equ):
        equ = equ.replace(m.group(0).replace("x", ""), "A", 1)
        i = i + 1
    for n in re.finditer(r"[\d]{1,}", equ):
        equ = equ.replace(n.group(0), "B", 1)
        i = i + 1

    equ = equ.replace("Z", "A")
    return equ


df = pd.read_csv("data/data.csv",delimiter=";")

df = df.drop("feedback", axis=1)

aluno = ''
answer = ''
df["passoAnt"] = 0

for (i, row) in df.iterrows():
    if (row["email"] == aluno) and (row["exercicio"] == answer):
        df.loc[i,'passoAnt'] = str(df.loc[i-1,"passo"])
    else:
        df.loc[i,'passoAnt'] = str(df.loc[i,"exercicio"])
    
    aluno = row["email"]
    answer = row["exercicio"]

df = df.drop("exercicio", axis=1)
df = df.drop("timestamp", axis=1)


correto = df.loc[:,"correct"].values
df = df.drop("correct", axis=1)

df["passo"] = df["passo"].map(template_equacao)
df["passoAnt"] = df["passoAnt"].map(template_equacao)

processor = make_column_transformer(
    ('passthrough', ["email"]),
    (Binarizer(), ["AD", "SB", "MT", "SP", "PA", "PM", "MM", "DM", "AF", "MF", "OI", "UT", "RE", "ER", "DE"]),
    (OneHotEncoder(), ["passoAnt", "passo"]),
)

previsores = processor.fit_transform(df).toarray()


"""
Criação do modelo
"""
modelo = Sequential()

modelo.add(Dense(200, activation="tanh", input_shape=(previsores.shape[1],)))
modelo.add(Dropout(0.2))

modelo.add(Dense(100, activation="tanh"))

modelo.add(Dense(1, activation="sigmoid"))

modelo.compile(loss="mean_squared_error", optimizer="Adam", metrics=[Precision(), AUC(), "accuracy"])

modelo.fit(previsores, correto, batch_size=32, epochs=100)