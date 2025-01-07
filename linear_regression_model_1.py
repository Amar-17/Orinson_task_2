


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



df=pd.read_csv(r"C:\Users\Sasi\Downloads\archive.zip")
print(df.head())
print(df.isnull().sum())
sns.regplot(x="Hours",y="Scores",data=df,ci=95)
plt.show()

X=df[["Hours"]]
y=df[["Scores"]]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(f"Coefficient: {model.coef_}")
print(f"Intercept: {model.intercept_}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"mean squared error :{mse}")
print(f"r2 score : {r2}")
