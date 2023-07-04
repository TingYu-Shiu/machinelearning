import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#線性回歸，可用一條線表示，cost function >> MSELoss

df_train = pd.read_csv('house_train_clean.csv')


df_train_y = df_train['單價(元/平方公尺)']
df_train = df_train.drop(['單價(元/平方公尺)'], axis=1)


x_train,x_test,y_train,y_test = train_test_split(df_train,df_train_y,test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f'coefficients係數:{model.coef_}\n')
print(f"MSE均方誤差: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f'Variance score:{r2_score(y_test,y_pred)}')

s1 = pd.Series(y_pred,index=y_test.index)
df2 = pd.DataFrame([y_test,s1],index=('實際','預測')).T
#-------------
plt.figure(figsize=(10,10))
plt.scatter(x_test['總價(元)'],y_test, color='black')
plt.scatter(x_test['總價(元)'],y_pred, color='blue')
plt.xticks(())
plt.yticks(())
plt.show()