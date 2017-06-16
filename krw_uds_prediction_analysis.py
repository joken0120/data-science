# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 20:48:20 2017

@author: Ken
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
import date
import datetime

temp_mart_12 = pd.read_csv("temp_mart_12_conv.csv")

col_in_null = [] # null or nan을 포함하는 col list
for col in temp_mart_12.columns.values:
    if True in set(temp_mart_12[col].isnull()):
        col_in_null.append(col)
#col = col_in_null[0]

for i, col in enumerate(col_in_null):    
    null_idx = np.where(temp_mart_12[col].isnull()==True)[0]
    for n_idx in null_idx:
        is_first, is_final = False, False
        if n_idx == 0:
            is_first = True
        if n_idx == len(temp_mart_12[col])-1:
            is_final = True
        f_idx, e_idx = n_idx-1, n_idx+1
        while f_idx in null_idx:
            f_idx -= 1
        while e_idx in null_idx:
            e_idx += 1 
        if is_final is True:
            e_idx = f_idx
        if is_first is True or f_idx<0:
            f_idx = e_idx
        rep_v = np.average([temp_mart_12[col][f_idx], temp_mart_12[col][e_idx]])
        temp_mart_12[col][n_idx] = rep_v       
    print("{}/{}".format(i+1,len(col_in_null)))

temp_mart_12.to_csv("data_mart_final.csv",ignore_index = Ture)

import os
os.chdir(r'C:\Users\KEJ\Desktop\miningproject_data\merged자료')
data_mart = pd.read_csv("data_mart_final.csv")
temp_mart_12 = data_mart[data_mart.columns[1:]]

# making diff
diff = [temp_mart_12['f_krw_usd'][i] - temp_mart_12['f_krw_usd'][i-1] for i in range(1,len(temp_mart_12))]
diff.append(0)
temp_mart_12['diff'] = diff


# =========  Modeling  ========== # 

# ====================== # 
# 1. data 전처리 및 탐색  #
# ====================== #
model_data = temp_mart_12[temp_mart_12['Date']< '2016-05-01']
valid_data = temp_mart_12[temp_mart_12['Date']>= '2016-05-01']

model_data2 = model_data[model_data.columns[1:]] # 기간 나누지 않은 전체 데이터

# (1) X,y : train/test , valid 
X = model_data.ix[:,model_data.columns != 'f_krw_usd']
X = X.drop('Date',axis=1)
y = model_data['f_krw_usd']

val_x = valid_data.ix[:,valid_data.columns != 'f_krw_usd']
val_x = val_x.drop('Date',axis=1)
val_y = valid_data['f_krw_usd']


# (2)  데이터 탐색
import seaborn as sns
sns.set(style = 'whitegrid',context='notebook')
cols =["f_krw_usd","s_kospi","u_index","c_oil_breant","s_nasdaq",'y_us_1m','y_us_1yr','y_us_5yr']
sns.pairplot(model_data2[cols],kind="reg")
plt.show()

# heat_map
cm = np.corrcoef(model_data2[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',
                 annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',
                 annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
plt.show()

# (3) data scailing
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale
scaled_X = pd.DataFrame(minmax_scale(X),columns = X.columns)
scaled_y = y

val_x = minmax_scale(val_x)
val_x = pd.DataFrame(val_x,columns = X.columns)
val_y = val_y

# (4) data split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X,scaled_y,
                                                    test_size=0.3,random_state=100)

# ============ # 
# 2. Modeling  #
# ============ #

###  (1) Ordinary Linear Regression  ###
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

y_train_pred = reg.predict(X_train) # train
y_test_pred = reg.predict(X_test) # test
y_val_pred = reg.predict(val_x) # valid

vrasid = y_val_pred-val_y

###  (2) Ransac Regression  ###
import numpy as np
from sklearn.linear_model import RANSACRegressor
reg2 = RANSACRegressor(LinearRegression(),max_trials = 100,min_samples=100,
                       residual_metric = lambda x: np.sum(np.abs(x), axis=1),
                       residual_threshold=30.0,random_state=100)

reg2.fit(X_train,y_train)

y_train_pred = reg2.predict(X_train)
y_test_pred = reg2.predict(X_test)
y_val_pred = reg2.predict(y_val_pred)


###  (3) Ridge Regression  ###
from sklearn.linear_model import Ridge
rdge = Ridge(alpha=0.01)
rdge.fit(X_train,y_train)

y_train_pred = rdge.predict(X_train)
y_test_pred = rdge.predict(X_test)
y_val_pred = rdge.predict(val_x)

###  (4) Lasso Regression  ###
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.001)
lasso.fit(X_train,y_train)

y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
y_val_pred = lasso.predict(val_x)

###  (5) Elastic Regression  ###
from sklearn.linear_model import ElasticNet
ela = ElasticNet(alpha=0.001, l1_ratio = 0.9)
ela.fit(X_train,y_train)

y_train_pred = ela.predict(X_train)
y_test_pred = ela.predict(X_test)
y_val_pred = ela.predict(val_x)

###  (6) Forward selection Regression  ###
import statsmodels.formula.api as smf

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

model = forward_selected(model_data2, 'f_krw_usd') # 전체 데이터를 가지고 모델링
model.summary()

print(model.model.formula)

# 유의미하게 선택된 변수
col = ["f_krw_cny","f_cny_usd","diff","y_hk_3yr","c_gas","y_fr_3yr","f_krw_aud","s_krx_100","s_kerx_autos","s_kospi","y_uk_1yr","s_nikkei","s_nyse","f_eur_usd","s_krx_build","u_index","f_jpy_usd","c_gold","s_krx_stock","s_krx_energy","y_us_5yr"]
print(model.rsquared_adj)


# ======================= # 
# 3. Modeling 검증 및 성능 #
# ======================= # 

# (1) Residual plot
plt.scatter(y_train_pred, y_train_pred-y_train,c='blue',marker='o',
            label = 'train data')
plt.scatter(y_test_pred, y_test_pred-y_test,c='lightgreen',marker='s',
            label = 'test data')
plt.scatter(y_val_pred, y_val_pred-val_y,c='black',marker='x',
            label = 'valid data')

plt.xlabel('prediction')
plt.ylabel('residual')
plt.hlines(y=0,xmin=950,xmax=1250,lw=2,color='red')
plt.xlim([950,1250])
plt.title('[krw_usd prediction residual plot]')
plt.legend(loc=2)
plt.show()


# (2) r-square and MSE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mse_valid = mean_squared_error(val_y, y_val_pred)

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
rmse_valid = np.sqrt(mse_valid)

r2_train = r2_score(y_train,y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_v = r2_score(val_y, y_val_pred)

SSE = sum((y_val_pred - val_y)**2)
SSR = sum((y_val_pred - np.mean(val_y))**2) 
SST = SSE + SSR
r2_val = SSR/SST

tr_n = len(X_train)
te_n = len(X_test)
vl_n = len(val_x)
p = len(val_x.columns)

adj_trian_r2 = 1-((tr_n-1)/(tr_n-p-1))*(1-r2_train)
adj_test_r2 = 1-((te_n-1)/(te_n-p-1))*(1-r2_test)
adj_valid_r2 =  1-((vl_n-1)/(vl_n-p-1))*(1-r2_val)
print('RMSE - train data : %.6f, test data: %.6f, valid data: %.6f' %(rmse_train,rmse_test,rmse_valid))
print('adj_R2 - train data : %.6f, test data: %.6f, valid data: %.6f' %(adj_trian_r2,adj_test_r2,adj_valid_r2))

# (3) plot 확인 
import datetime
date_list = []
for i in range(len(temp_mart_12['Date'])):
    # date_list.append(time.mktime(datetime.datetime.strptime(temp_mart_12['Date'][i], "%Y-%m-%d").timetuple()))    
    date_list.append(datetime.datetime.strptime(temp_mart_12['Date'][i],"%Y-%m-%d"))

plt.plot(date_list,temp_mart_12['f_krw_usd']) # 전체
plt.plot(date_list[0:len(y)],y) # train/test
plt.plot(date_list[len(y):],val_y) # valid
plt.plot(date_list[len(y):],y_val_pred) # y_valid
