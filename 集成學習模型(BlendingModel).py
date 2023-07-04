import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

import pandas as pd

#模型都訓練結束後，blending，個別方法差異要很大，而且單一模型效果都很好，選定適合costfunction，進行權重，costfunction越大，權重越小(取倒數作加成)

df_train = pd.read_csv('house_train_clean.csv')

df_train_y = df_train['單價(元/平方公尺)']
df_train = df_train.drop(['單價(元/平方公尺)'], axis=1)

x_train,x_test,y_train,y_test = train_test_split(df_train,df_train_y,test_size=0.2, random_state=1)

lr =LinearRegression()
rf =RandomForestClassifier(max_depth=4,n_estimators=5)

model_lr =lr.fit(x_train,y_train)
lr_pred = model_lr.predict(x_test)
mse_lr = mean_squared_error(lr_pred,y_test, squared=False)
print(mse_lr)


model_rf =rf.fit(x_train,y_train)
rf_pred = model_rf.predict(x_test)
mse_rf = mean_squared_error(rf_pred,y_test, squared=False)
print(mse_rf)

model_per = Perceptron().fit(x_train,y_train)
per_pred = model_per.predict(x_test)
mse_per = mean_squared_error(per_pred,y_test, squared=False)
print(mse_per)


mse_sum = 1/mse_rf + 1/mse_lr + 1/mse_per
blending_pred = lr_pred*((1/mse_lr)/mse_sum)+rf_pred*((1/mse_rf)/mse_sum) +per_pred*((1/mse_per)/mse_sum)

print(mean_squared_error(blending_pred,y_test))



