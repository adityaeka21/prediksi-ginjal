import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@st.cache_data()
def load_data():
    df = pd.read_csv('new_model.csv')
    x = df.drop('Class', axis=1)
    y = df['Class']
    return df, x, y

@st.cache_data()
def train_model(x,y):
  X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
  model = DecisionTreeClassifier(criterion="entropy", random_state=20)
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  score = accuracy_score(y_test,y_pred)
  return model, score

def predict(x, y, features):
  model, score = train_model(x,y)
  pred = model.predict(np.array(features).reshape(1,-1))
  return pred, score