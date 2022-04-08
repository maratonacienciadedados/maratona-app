import pandas as pd
import streamlit as st
from PIL import Image

import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

image = Image.open("logo.png")
st.image(image)
st.title("Projeto Prático")
st.text('Predição de Custos/Campanhas de Marketing')

df = pd.read_csv("vendas_marketing.csv")

show_data = st.checkbox("Conjunto de Dados")

if show_data:
    st.write(df.head(2))

#Modelo Joblib
model = joblib.load('model.pkl')

valor_tv = st.slider("Valor Investido em TV", min_value = int(df["TV"].min()), max_value = int(df["TV"].max()))

pred = model.predict([[valor_tv]]).round(2)

#Modelo Sem Joblib
X = df[["TV"]]
y = df[["Sales"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred2 = lr.predict(X_test)

#Avaliação
r2 = r2_score(y_test,pred2)
mse = mean_squared_error(y_test,pred2)
mae = mean_absolute_error(y_test,pred2)

metrics_shows = st.checkbox("Métricas")

if metrics_shows:
    st.text("R2: {}".format(float(r2)))
    st.text("MSE: {}".format(float(mse)))
    st.text("MAE: {}".format(float(mae)))


if st.button("Realizar Predição de Vendas"):
    st.text("Predição de Vendas com base na campanha de TV: RS{}".format(float(pred)))




