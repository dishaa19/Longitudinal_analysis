
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from scipy.stats import spearmanr
import math


# In[2]:


df_1 = pd.read_csv('ip_7.csv',header=0)
y = pd.read_csv('weeks.csv',header = 0)
df = df_1.transpose()
df = df[1:]
X = df
X = X.dropna()


# In[3]:


median_pred = pd.DataFrame({"patient_id": X.index})
y = pd.concat([median_pred, y], axis=1)
y['patient_id'] = y['patient_id'].astype(str)
Y = y.set_index('patient_id')
Y = Y["weeks"]
Y


# In[4]:


X["p"] = [x.split("_")[0] for x in X.index]
y["p"] = [x.split("_")[0] for x in Y.index]


# In[5]:


n_alphas = 100
# # Create the parameter grid based on the results of random search 
param_grid = {
    'alphas': [np.logspace(-10, -2, n_alphas)],
}
alphas = np.logspace(-10, -2, n_alphas) 
prediction = []
pred_df = pd.DataFrame({})
pred_val_list = []
pred_val = []
i = 0
result = []
scaler = MinMaxScaler(feature_range=(0, 1))
sc = StandardScaler()


# In[6]:


for patient in X["p"].unique():
    X_val = X.loc[X.p==patient]
    X_train = X.loc[X.p!=patient]
    y_val = Y.loc[X.p==patient]
    y_train = Y.loc[X.p!=patient]
    X_val = X_val.drop("p", axis=1)
    X_train = X_train.drop("p", axis=1)
    
    #pred_old = pd.DataFrame({"patient_id": X_train.index})
    X_train = sc.fit_transform(X_train)
    X_val = sc.fit_transform(X_val)
    #X_train, y_train = make_regression(n_samples=len(X_train), n_features=len(X.columns)-1,random_state = 0)
    #X_val, y_val = make_regression(n_samples=len(X_val), n_features=len(X.columns)-1,random_state = 0)
    
    
    en_net = ElasticNetCV(alphas = alphas)
    en_net = en_net.fit(X_train,y_train)
    
    res = en_net.predict(X_val)
    result = np.append(result, res)
    i += 1
    
    
print(result)
        
#     en = ElasticNetCV()
#     grid = GridSearchCV(en,param_grid = param_grid,cv=2)
#     # X_train: number of features in the dataset x 68 samples
#     grid.fit(X_train,y_train)
#     # output: 1 x 68 samples
     
#     #training set prediction
#     #pred = grid.predict(X_train)
#     pred_val = grid.predict(X_val)
#     # savr pred and pred_val as a column
#     #print(pred)
#     #print(grid.alpha_)
#     #pred = np.append(pred, pred)
#     #pred_df['Col'+str(i)] = list(pred.flatten())
    
#     #pred_val = np.append(pred_val,pred_val)
#     #here you create a list for the testset prediction - single omic
#     pred_val_list = np.append(pred_val_list, pred_val)
#     i += 1
    
 


# In[7]:


spearmanr(y_train, en_net.predict(X_train) )


# In[8]:


print("Length of prediction:",len(result))
print("Length of y_val:",len(y_val))

y_true = y.sort_values(["patient_id"])
print(y_true)
y_true = np.array(y_true.weeks)
y_val  = y_true
print(y_val)


# In[9]:


mse = mean_squared_error(y_true,result)
print('Mean squared error:', mse)
print('Root Mean Squared Error (testing set):',np.sqrt(mean_squared_error(y_true,result)))
corr, p = spearmanr(y_true, result)
print("Correlation", corr)
print("P value:", p)
log = -(math.log10(p))
print('-Log(p_value):',log)


# In[ ]:


[5.17,0.44,0.016,0.024,1.03,31.72,29.06]

