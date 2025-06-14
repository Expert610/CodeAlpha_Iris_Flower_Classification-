import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

iris = load_iris()

df =pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

df['species'] = df['species'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

x = df.drop('species', axis=1)
y= df['species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)

print(f"Accuracy Score",accuracy_score(y_test,y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_predict))

