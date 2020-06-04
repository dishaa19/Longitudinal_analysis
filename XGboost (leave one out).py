
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from scipy.stats import spearmanr
import math


# In[18]:


df_1 = pd.read_csv('ip_5.csv',header=0)
y = pd.read_csv('weeks.csv',header = 0)
df = df_1.transpose()
df = df[1:]
X = df


# In[19]:


median_pred = pd.DataFrame({"patient_id": X.index})
y = pd.concat([median_pred, y], axis=1)
y['patient_id'] = y['patient_id'].astype(str)
Y = y.set_index('patient_id')
Y = Y["weeks"]
Y


# In[20]:


X["p"] = [x.split("_")[0] for x in X.index]
y["p"] = [x.split("_")[0] for x in Y.index]


# In[21]:


#reg:linear
param_grid = {'objective':['reg:squarederror']}
prediction = []
i = 0
scaler = MinMaxScaler(feature_range=(0, 1))
sc = StandardScaler()


# In[22]:


for patient in X["p"].unique():
    X_val = X.loc[X.p==patient]
    X_train = X.loc[X.p!=patient]
    y_val = Y.loc[X.p==patient]
    y_train = Y.loc[X.p!=patient]
    X_val = X_val.drop("p", axis=1)
    X_train = X_train.drop("p", axis=1)
    pred_old = pd.DataFrame({"patient_id": X_train.index})
    X_train = sc.fit_transform(X_train)
    X_val = sc.fit_transform(X_val)
    #data_dmatrix = xgb.DMatrix(data=X,label=y)
    #X_train, y_train = make_regression(n_samples=len(X_train), n_features=len(X.columns)-1)
    #X_val, y_val = make_regression(n_samples=len(X_val), n_features=len(X.columns)-1)
        
    xgb1 = XGBRegressor()
    grid = GridSearchCV(xgb1,
                        param_grid,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

    # xgb_grid.fit(X_train,y_train)
    grid.fit(X_train,y_train)
     
    #prediction for the test set
    preds  = grid.predict(X_val)
    prediction = np.append(prediction, preds)
    i += 1
    


# In[23]:


print("Length of prediction:",len(prediction))
print("Length of y_val:",len(y_val))
print("Prediction:",prediction)

y_true = y.sort_values("patient_id")
y_true = np.array(y_true.weeks)
y_val  = y_true


# In[24]:


mse = mean_squared_error(y_val,prediction)
print('Mean squared error:', mse)
print('Root Mean Squared Error (testing set):',np.sqrt(mean_squared_error(y_val,prediction)))

corr, p = spearmanr(y_val, prediction)
print("Correlation:", corr)
print("P value:", p)

print( "%.16f" % float(p))

log = -(math.log10(p))
print("-Log(p_val):",log)


# In[ ]:


[9.98,0.13,0.089,10.19,0.90,27.17,27.79]

