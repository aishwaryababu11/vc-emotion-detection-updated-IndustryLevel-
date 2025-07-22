import pandas as pd
import numpy as np 
import pickle 
import yaml

from sklearn.ensemble import RandomForestClassifier

with open("params.yaml","r") as file:
    params = yaml.safe_load(file)

n_estimators = params['modelling']['n_estimators']
max_depth = params['modelling']['max_depth']

train_data = pd.read_csv("data/interim/train_bow.csv")

x_train = train_data.drop(columns=['sentiment']).values
y_train = train_data['sentiment'].values

model = RandomForestClassifier(n_estimators= n_estimators,max_depth = max_depth, random_state=42)
model.fit(x_train, y_train)

pickle.dump(model, open("models/random_forest_model.pkl", "wb"))
