
# coding: utf-8

# In[50]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from scipy.stats import spearmanr
import math
from sklearn.svm import SVR


# In[51]:


df_1 = pd.read_csv('ip_1.csv',header=0)
y = pd.read_csv('weeks.csv',header = 0)
df = df_1.transpose()
df = df[1:]
X = df


# In[52]:


median_pred = pd.DataFrame({"patient_id": X.index})
y = pd.concat([median_pred, y], axis=1)
y['patient_id'] = y['patient_id'].astype(str)
Y = y.set_index('patient_id')
Y = Y["weeks"]
Y


# In[53]:


X["p"] = [x.split("_")[0] for x in X.index]
y["p"] = [x.split("_")[0] for x in Y.index]


# In[54]:


parameters = {'kernel':['rbf', 'linear'], 'C':np.logspace(np.log10(0.01), np.log10(100), num=10), 'gamma':np.logspace(np.log10(0.001), np.log10(2), num=20)}
prm = {}
prediction = []
i = 0
scaler = MinMaxScaler(feature_range=(0, 1))
sc = StandardScaler()


# In[55]:


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
#     X_train, y_train = make_regression(n_samples=len(X_train), n_features=len(X.columns)-1)
#     X_val, y_val = make_regression(n_samples=len(X_val), n_features=len(X.columns)-1)
        
    svr = SVR()
    grid_searcher_red = GridSearchCV(svr, parameters,n_jobs=8,verbose=2, cv=2)
    grid_searcher_red.fit(X_train,y_train)

     
    #training set prediction
#     pred = grid_searcher_red.predict(X_train)
#     prm = grid_searcher_red.best_params_
    
#     #print(prm)
#     C = prm.get('C')
#     gamma = prm.get('gamma')
#     kernel = prm.get('kernel')
    
#     svr_new = SVR(C=C, gamma=gamma, kernel=kernel)
#     svr_new.fit(X_train,y_train)
    
    #score = en.score(X_val, y_val)
    
    #prediction for the test set
    preds  = grid_searcher_red.predict(X_val)
    prediction = np.append(prediction, preds)
    i += 1
    


# In[56]:


print("Length of prediction:",len(prediction))
print("Length of y_val:",len(y_val))
print("Prediction:",prediction)

y_true = y.sort_values("patient_id")
y_true = np.array(y_true.weeks)
y_val  = y_true


# In[57]:


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


[0.09,0.20,0.80,0.16,0.15,15.19,32.35]

