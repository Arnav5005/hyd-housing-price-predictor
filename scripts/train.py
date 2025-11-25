import pandas as pd
import joblib

df=pd.read_csv('data/Hyderabad.csv')

# Remove both 98th and 99th percentile outliers together
upper = df["Price"].quantile(0.98)
df = df[df["Price"] <= upper]

# Encoding

cols = [c for c in ["Location"] if c in df.columns]
df_encoded=pd.get_dummies(df,columns=cols,drop_first=True)

# split x and y

# x-->y

x=df_encoded.drop('Price',axis=1) # drop the price column
y=df_encoded['Price']

# train-test split

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# model training

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)

# model evaluation

from sklearn.metrics import root_mean_squared_error , r2_score
y_predicted=model.predict(x_test) # x_test-->y_predicted
rmse=root_mean_squared_error(y_test,y_predicted)
r2=r2_score(y_test,y_predicted)
print("Model Evaluation : ")
print(f"RMSE : {rmse:.2f}")
print(f"R2 score : {r2:.4f}")

# saving model

joblib.dump(model,"housing_model.joblib")