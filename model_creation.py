import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd 
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load dataset
mpg_df = pd.read_csv(r"D:\training\regression\Car-mpg- Dataset.csv")

# Drop non-numeric column
mpg_df = mpg_df.drop('car_name', axis=1)

# Replace missing values represented as '?' with NaN
mpg_df = mpg_df.replace('?', np.nan)

# Fix 'hp' column (convert to float)
mpg_df['hp'] = mpg_df['hp'].astype('float64')

# Fill missing values with median (only numeric columns)
numeric_cols = mpg_df.select_dtypes(include=['number']).columns
mpg_df[numeric_cols] = mpg_df[numeric_cols].apply(lambda x: x.fillna(x.median()), axis=0)

# Convert origin numbers to categories
mpg_df['origin'] = mpg_df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})

mpg_df = pd.get_dummies(mpg_df, columns=['origin'], drop_first=True)
print(mpg_df)

# Features and Target
X = mpg_df.drop('mpg', axis=1)
print(X)
y = mpg_df[['mpg']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Train the model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
# y_train_pred = regression_model.predict(X_train)
# y_test_pred = regression_model.predict(X_test)
# from sklearn.metrics import root_mean_squared_error,r2_score

# rmse=root_mean_squared_error(y_train,y_train_pred)
# r2=r2_score(X_test,y_test)
# print(r2)
# print(rmse)
import pickle
# Saving model to disk
pickle.dump(regression_model, open('model.pkl','wb'))
