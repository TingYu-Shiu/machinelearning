from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd


#alpha越強，懲罰力度越強(L1:Lasso, L2:Ridge, 混和:ElasticNet)

df_train = pd.read_csv('house_train_clean.csv')


df_train_y = df_train['單價(元/平方公尺)']
df_train = df_train.drop(['單價(元/平方公尺)'], axis=1)



x_train,x_test,y_train,y_test = train_test_split(df_train,df_train_y,test_size=0.2, random_state=1)

StdS =  StandardScaler()
x_train = StdS.fit_transform(x_train)
x_test = StdS.transform(x_test)

Rg = ElasticNet(alpha = 0.05, l1_ratio=0.5) #Rigde 
Rg.fit(x_train,y_train)
print("權重\n",Rg.coef_)

y_predict = Rg.predict(x_test)
print("預測結果：\n",y_predict)

print("均方誤差為：",mean_squared_error(y_test,y_predict))
