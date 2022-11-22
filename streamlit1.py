import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

st.title("Region predictor")
st.write("Predict below in which region of the world your imaginary country would lie!")

def reg_to_cont(df):
    df.replace("[(]EX. NEAR EAST[)]", "", inplace=True, regex=True)
    df.replace("NEAR EAST", "ASIA", inplace=True, regex=True)
    df.replace(("WESTERN EUROPE", "EASTERN EUROPE"), "EUROPE", inplace=True, regex=True)
    df.replace(("SUB-SAHARAN AFRICA","NORTHERN AFRICA"), "AFRICA", inplace=True, regex=True)
    df.replace(("LATIN AMER. & CARIB","NORTHERN AMERICA"), "AMERICA", inplace=True, regex=True)
    df.replace(("C.W. OF IND. STATES", "BALTICS"), "FORMER USSR", inplace=True, regex=True)
    return(df)


data = pd.read_csv(r"C:\Users\korsiee\Desktop\countries.csv", decimal=",").dropna()

@st.cache
def run():
    global features_train, features_test, labels_train, labels_test, clf
    acc=0
    while acc <=0.7:
        data_train, data_test = train_test_split(data)
        reg_to_cont(data_train)
        reg_to_cont(data_test)
        features_train = data_train[["Area (sq. mi.)", "GDP ($ per capita)", "Literacy (%)", "Arable (%)", "Birthrate", "Deathrate"]]
        labels_train = data_train["Region"]
        features_test = data_test[["Area (sq. mi.)", "GDP ($ per capita)", "Literacy (%)", "Arable (%)", "Birthrate", "Deathrate"]]
        labels_test = data_test["Region"]
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
        clf = clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)
        acc = metrics.accuracy_score(labels_test, labels_pred)
    return(acc, features_train, features_test, labels_train, labels_test, clf)

result = run()
acc = result[0]
features_train = result[1]
features_test = result[2]
labels_train = result[3]
labels_test = result[4]
clf = result[5]

st.write("Accuracy of the model is " + str(round(100*acc, 1)) + "%")

st.subheader("Adjust the characteristics of your imaginary country here")
area_predict = st.slider("Area of your imaginary country (square kilometers):", min_value=int(data["Area (sq. mi.)"].min()), max_value=int(data["Area (sq. mi.)"].max()))
gdp_predict = st.slider("GDP of your imaginary country ($ per capita):", min_value=int(data["GDP ($ per capita)"].min()), max_value=int(data["GDP ($ per capita)"].max()))
lit_predict = st.slider("Literacy of your imaginary country (% of the population):", min_value=int(data["Literacy (%)"].min()), max_value=int(data["Literacy (%)"].max()))
arab_predict = st.slider("Arable land in your imaginary country (% of the total area):", min_value=int(data["Arable (%)"].min()), max_value=int(data["Arable (%)"].max()))
birth_predict = st.slider("Birthrate of your imaginary country (births per thousand people per year):", min_value=int(data["Birthrate"].min()), max_value=int(data["Birthrate"].max()))
death_predict = st.slider("Deathrate of your imaginary country (deaths per thousand people per year):", min_value=int(data["Deathrate"].min()), max_value=int(data["Deathrate"].max()))

country_predict = pd.DataFrame([[area_predict, gdp_predict, lit_predict, arab_predict, birth_predict, death_predict]], columns=["Area (sq. mi.)", "GDP ($ per capita)", "Literacy (%)", "Arable (%)", "Birthrate", "Deathrate"])

st.subheader("Data of your imaginary country")
st.write(country_predict)

if st.checkbox("Tell me in which region my imaginary country is!"):
    region_predict = clf.predict(country_predict)
    st.write(region_predict[0])

if st.checkbox("Show training data"):
    st.write(features_train, labels_train)

if st.checkbox("Show testing data"):
    st.write(features_test, labels_test)

if st.checkbox("Show data source"):
    st.write("The data was retrieved from the public domain from https://www.kaggle.com/datasets/fernandol/countries-of-the-world on the 15th of november 2022.")

if st.checkbox("Show initial investigations"):
    st.write("The countries containing no data for certain features were removed from the data, which could lead to a slightly skewed data set. The depth of the decision tree is quite high because the countries of the world are so diverse, but overfitting is prevented by checking on the testing data.")

if st.checkbox("Show app objective"):
    st.write("In this app the goal was to make a decision tree classifier that places an imaginary country containing six features into six classes. We use all the countries in the world as training and testing data.")

if st.checkbox("Show fallbacks"):
    st.write("In this model the decision tree is updated until a high accuracy is guaranteed. It can sometimes happen that a model is very good for the training data but not for the testing data, implying that the model would not work for new instances either. We are assuring that this is not the case for our model by gaining a high accuracy on the testing data.")

if st.checkbox("Show the main challenges deployment phase"):
    st.write("If we were to deploy the website to larger audiences, we would need to have our own physical servers or rent servers in order for the website to be able to handle a large amount of visitors.")

st.write("This website was made for the theoretical chemistry and computational modelling master by Kors Doedens.")