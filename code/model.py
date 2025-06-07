import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# read data
df = pd.read_csv('./data/predictive_maintenance.csv')
df["Type"] = df["Type"].replace({"L": 0, "M": 1, "H": 2})

# split data into features and target
X = df.drop(columns=['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
y = df['Target']

# train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model to a file
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
