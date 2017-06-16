# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 20:53:29 2017

@author: Ken
"""

"""
Dataminig project
"""
import os
os.chdir(r'C:\Users\KEJ\Desktop\miningproject_data\merged자료')
os.getcwd()

######## 1. data gathering ########
import quandl
# BGUn7HThyZTfkBbrvJyv
quandl.ApiConfig.api_key = 'BGUn7HThyZTfkBbrvJyv'

import pandas as pd

def Quandl_get(get_quandl):
    get_q = quandl.get(get_quandl,start_date="2012-01-01", end_date="2017-05-31")
    get_q.reset_index(level=0, inplace=True)  # index를 column으로 
    return get_q

###### 1) target : 원달러 
fx_krw_usd = quandl.get("CURRFX/USDKRW",start_date="2012-01-01", end_date="2017-05-31")
fx_krw_usd = fx_krw_usd[[0]]  #rate만 사용
fx_krw_usd.reset_index(level=0, inplace=True) # index를 column으로 
fx_krw_usd.columns = ['Date','f_krw_usd']  

###### 2) Trade Weighted US Dollar Index: Broad
usd_index = Quandl_get("FRED/DTWEXB")
usd_index.columns = ['Date','u_index']

###### 3) Commodity Index (International)
"""
# cmd_copper
cmd_copper<-quandl.get("WSJ/COPPER", start_date="2012-01-01", end_date="2017-05-31")
str(cmd_copper)
names(cmd_copper)[2]<-"c_copper"
cmd_copper<-cmd_copper[order(cmd_copper$Date),]
head(cmd_copper,3);tail(cmd_copper,3)

# cmd_corn
cmd_corn<-quandl.get("WSJ/CORN_FEED", start_date="2012-01-01", end_date="2017-05-31")
str(cmd_corn)
names(cmd_corn)[2]<-"c_corn"
cmd_corn<-cmd_corn[order(cmd_corn$Date),]
head(cmd_corn,3);tail(cmd_corn,3)
"""

# gold
cmd_gold = Quandl_get("LBMA/GOLD")
cmd_gold = cmd_gold[[0,2]] # usd PM
cmd_gold.columns  = ['Date','c_gold']

# brent oil
cmd_oil_brent = Quandl_get("FRED/DCOILBRENTEU")
cmd_oil_brent.columns = ['Date','c_oil_breant']

# wti_oil
cmd_oil_wti = Quandl_get("FRED/DCOILWTICO")
cmd_oil_wti.columns = ['Date','c_oil_wti']

# cmd_gas
cmd_gas = Quandl_get("YAHOO/INDEX_XNG")
cmd_gas = cmd_gas[[0,4]]
cmd_gas.columns = ['Date','c_gas']

# cmd_silver
cmd_silver = Quandl_get("LBMA/SILVER")
cmd_silver = cmd_silver[[0,1]]
cmd_silver.columns = ['Date','c_silver']


# ====================================== #
# 23. Stock Market Index (International) #
# ====================================== #

# stck_cac40
stck_cac40 = Quandl_get("YAHOO/INDEX_FCHI")
stck_cac40 = stck_cac40[[0,4]]
stck_cac40.columns = ['Date','s_cac40']

# stck_dax
stck_dax = Quandl_get("YAHOO/INDEX_GDAXI")
stck_dax = stck_dax[[0,4]]
stck_dax.columns = ["Date",'s_dax']

# stck_nasdaq
stck_nasdaq = Quandl_get('NASDAQOMX/NDX')
stck_nasdaq = stck_nasdaq[[0,1]]
stck_nasdaq.columns = ["Date",'s_nasdaq']

# stck_nikkei
stck_nikkei = Quandl_get('YAHOO/INDEX_N225')
stck_nikkei = stck_nikkei[[0,4]]
stck_nikkei.columns = ["Date",'s_nikkei']

# stck_nyse
stck_nyse = Quandl_get('YAHOO/INDEX_NYA')
stck_nyse = stck_nyse[[0,4]]
stck_nyse.columns = ["Date",'s_nyse']

# stck_snp500
stck_snp500 = Quandl_get('YAHOO/INDEX_GSPC')
stck_snp500 = stck_snp500[[0,4]]
stck_snp500.columns = ["Date",'s_snp500']

# stck_ssec
stck_ssec = Quandl_get('YAHOO/INDEX_SSEC')
stck_ssec = stck_ssec[[0,4]]
stck_ssec.columns = ["Date",'s_sser']

# ================================= #
# 24. Interest Rate (International) #
# ================================= #

# yield_ca
yield_ca_1y = pd.read_excel("bond_can_1y.xlsx")
yield_ca_1y = yield_ca_1y[[0,1]]
yield_ca_3y = pd.read_excel("bond_can_3y.xlsx")
yield_ca_3y = yield_ca_3y[[0,1]]

yield_ca = pd.merge(yield_ca_1y,yield_ca_3y,how='outer',on='date')
yield_ca.columns = ['Date',"y_ca_1yr","y_ca_3yr"]

# yield_jp 
yield_jp = pd.read_excel("bond_japan_3y.xlsx") 
yield_jp.columns = ["Date","y_jp_3yr"]

# yield_nz 
yield_nz = Quandl_get('YC/NZL') 
yield_nz = yield_nz[[0,1,2,3,4,6,7]]
yield_nz.columns = ["Date","y_nz_1m","y_nz_3m","y_nz_6m","y_nz_1yr","y_nz_5yr","y_nz_10yr"]

yield_nz_2017 = pd.read_csv("yield_nz_2015_2017.csv",skiprows = 4)
yield_nz_2017 = yield_nz_2017[[0,3,4,5,6,8,9]]
yield_nz_2017.columns = ["Date","y_nz_1m","y_nz_3m","y_nz_6m","y_nz_1yr","y_nz_5yr","y_nz_10yr"]

yield_nz = yield_nz.append(yield_nz_2017)
cnv_dt = pd.to_datetime(yield_nz['Date'])
yield_nz['Date'] = cnv_dt

# yield_us 
yield_us = Quandl_get("YC/USA")
yield_us = yield_us[[0,1,2,3,4,6,7,9]]
yield_us.columns = ["Date","y_us_1m","y_us_3m","y_us_6m","y_us_1yr","y_us_3yr","y_us_5yr","y_us_10yr"]

yield_us = Quandl_get("USTREASURY/YIELD")
yield_us = yield_us[[0,1,2,3,4,6,7,9]]
yield_us.columns = ["Date","y_us_1m","y_us_3m","y_us_6m","y_us_1yr","y_us_3yr","y_us_5yr","y_us_10yr"]

# yield_uk
yield_uk = pd.read_csv("yield_uk_2012_2017.csv")
yield_uk.columns = ["Date","y_uk_1yr","y_uk_5yr","y_uk_10yr"]

cnv_dt = pd.to_datetime(yield_uk['Date'])
yield_uk['Date'] = cnv_dt

# yield_chn
yield_chn = pd.read_excel("bond_china_3y.xlsx")
yield_chn.columns = ["Date","y_chn_3yr"]

# yield_fr
yield_fr = pd.read_excel("bond_france_3y.xlsx")
yield_fr.columns =  ["Date","y_fr_3yr"]

# yield_gr
yield_gr = pd.read_excel("bond_germany_3y.xlsx")
yield_gr.columns =  ["Date","y_gr_3yr"]

# yield_hk
yield_hk = pd.read_excel("bond_hongkong_3y.xlsx")
yield_hk.columns =  ["Date","y_hk_3yr"]

# yield_china
# ========================= #
# 25. Forex (International) #
# ========================= #

# fx_aud_usd
fx_aud_usd = Quandl_get("CURRFX/USDAUD")
fx_aud_usd = fx_aud_usd[[0,1]]
fx_aud_usd.columns = ["Date","f_aud_usd"]

# fx_cad_usd
fx_cad_usd = Quandl_get("CURRFX/USDCAD")
fx_cad_usd = fx_cad_usd[[0,1]]
fx_cad_usd.columns = ["Date","f_cad_usd"]

# fx_cny_usd
fx_cny_usd = Quandl_get("CURRFX/USDCNY")
fx_cny_usd = fx_cny_usd[[0,1]]
fx_cny_usd.columns = ["Date","f_cny_usd"]

# fx_eur_usd
fx_eur_usd = Quandl_get("CURRFX/USDEUR")
fx_eur_usd = fx_eur_usd[[0,1]]
fx_eur_usd.columns = ["Date","f_eur_usd"]

# fx_gbp_usd
fx_gbp_usd = Quandl_get("CURRFX/USDGBP")
fx_gbp_usd = fx_gbp_usd[[0,1]]
fx_gbp_usd.columns = ["Date","f_gbp_usd"]

# fx_jpy_usd
fx_jpy_usd = Quandl_get("CURRFX/USDJPY")
fx_jpy_usd = fx_jpy_usd[[0,1]]
fx_jpy_usd.columns = ["Date","f_jpy_usd"]

# fx_nzd_usd
fx_nzd_usd = Quandl_get("CURRFX/USDNZD")
fx_nzd_usd = fx_nzd_usd[[0,1]]
fx_nzd_usd.columns = ["Date","f_nzd_usd"]


# ================================ #
# 11. Interst Rate (Domestic: KRX) #
# ================================ #


# yield_kr
yield_kr_y1 = pd.read_excel("bond_kr_1y.xlsx")
yield_kr_y3 = pd.read_excel("bond_kr_3y.xlsx")

yield_kr = pd.merge(yield_kr_y1,yield_kr_y3,how='outer',on='date')
yield_kr.columns = ['Date',"y_kr_1yr","y_kr_3yr"]

# ================================================= #
# 12. Stock Market Industrial Index (Domestic: KRX) #
# ================================================= #

# stck_kospi
stck_kospi = Quandl_get("YAHOO/INDEX_KS11")
stck_kospi = stck_kospi[[0,4]]
stck_kospi.columns = ["Date","s_kospi"]


# stck_krx_100
stck_krx_100 = pd.read_excel("krx_100.xlsx")
stck_krx_100 = stck_krx_100[[0,1]]
stck_krx_100.columns = ["Date","s_krx_100"]


# stck_krx_autos
stck_krx_autos = pd.read_excel("krx_autos.xlsx")
stck_krx_autos = stck_krx_autos[[0,1]]
stck_krx_autos.columns = ["Date","s_kerx_autos"]

# stck_krx_energy
stck_krx_energy = pd.read_excel("krx_energy.xlsx")
stck_krx_energy = stck_krx_energy[[0,1]]
stck_krx_energy.columns = ["Date","s_krx_energy"]

# stck_krx_it
stck_krx_IT = pd.read_excel("krx_IT.xlsx")
stck_krx_IT = stck_krx_IT[[0,1]]
stck_krx_IT.columns = ["Date","s_krx_it"]

# stck_krx_semicon
stck_krx_semicon = pd.read_excel("krx_semicon.xlsx")
stck_krx_semicon = stck_krx_semicon[[0,1]]
stck_krx_semicon.columns = ["Date","s_krx_semicon"]

# stck_krx_bank
stck_krx_bank = pd.read_excel("krx_bank.xlsx")
stck_krx_bank = stck_krx_bank[[0,1]]
stck_krx_bank.columns = ["Date","s_krx_bank"]

# stck_krx_broadcast
stck_krx_broadcast = pd.read_excel("krx_broadcast.xlsx")
stck_krx_broadcast = stck_krx_broadcast[[0,1]]
stck_krx_broadcast.columns = ["Date","s_krx_broadcast"]

# stck_krx_build
stck_krx_build = pd.read_excel("krx_build.xlsx")
stck_krx_build = stck_krx_build[[0,1]]
stck_krx_build.columns = ["Date","s_krx_build"]

# stck_krx_healthcare
stck_krx_healthcare = pd.read_excel("krx_healthcare.xlsx")
stck_krx_healthcare = stck_krx_healthcare[[0,1]]
stck_krx_healthcare.columns = ["Date","s_krx_healthcare"]

# stck_krx_insurance
stck_krx_insurance = pd.read_excel("krx_insurance.xlsx")
stck_krx_insurance = stck_krx_insurance[[0,1]]
stck_krx_insurance.columns = ["Date","s_krx_insurance"]

# stck_krx_machinery
stck_krx_machinery = pd.read_excel("krx_machinery.xlsx")
stck_krx_machinery = stck_krx_machinery[[0,1]]
stck_krx_machinery.columns = ["Date","s_krx_machinery"]

# stck_krx_steel
stck_krx_steel = pd.read_excel("krx_steel.xlsx")
stck_krx_steel = stck_krx_steel[[0,1]]
stck_krx_steel.columns = ["Date","s_krx_steel"]

# stck_krx_stock
stck_krx_stock = pd.read_excel("krx_stock.xlsx")
stck_krx_stock = stck_krx_stock[[0,1]]
stck_krx_stock.columns = ["Date","s_krx_stock"]

# stck_krx_transportation
stck_krx_transportation = pd.read_excel("krx_transportation.xlsx")
stck_krx_transportation = stck_krx_transportation[[0,1]]
stck_krx_transportation.columns = ["Date","s_krx_transportation"]

# ==================== #
# 13. Forex (Domestic) #
# ==================== #

# fx_krw_aud
fx_krw_aud = Quandl_get("CURRFX/AUDKRW")
fx_krw_aud = fx_krw_aud[[0,1]]
fx_krw_aud.columns = ["Date","f_krw_aud"]

# fx_krw_cny
fx_krw_cny = Quandl_get("CURRFX/CNYKRW")
fx_krw_cny = fx_krw_cny[[0,1]]
fx_krw_cny.columns = ["Date","f_krw_cny"]

# fx_krw_gbp
fx_krw_gbp = Quandl_get("CURRFX/GBPKRW")
fx_krw_gbp = fx_krw_gbp[[0,1]]
fx_krw_gbp.columns = ["Date","f_krw_gbp"]

# fx_krw_eur
fx_krw_eur = Quandl_get("CURRFX/EURKRW")
fx_krw_eur = fx_krw_eur[[0,1]]
fx_krw_eur.columns = ["Date","f_krw_eur"]


######################################### Data merging #########################

### 1. date_list
import datetime
sdate= datetime.datetime.strptime("2012-01-01", "%Y-%m-%d")
edate = datetime.datetime.strptime("2017-05-31", "%Y-%m-%d")
date_delta = edate-sdate

date_list = [edate - datetime.timedelta(days=x) for x in range(0, date_delta.days+1)]

date_df = pd.DataFrame(pd.to_datetime(date_list),columns=['Date'])


### 2. Merge Date with krw/usd 

# temp_mart_01 = pd.merge(date_df,fx_krw_usd,how='outer',on='Date') 
temp_mart_01 = pd.merge(date_df,fx_krw_usd,how='inner',on='Date') # y값이 없는 날은 제거
len(temp_mart_01)

### 2. Merge Korean bond yield
temp_mart_02 = temp_mart_01
temp_mart_02 = pd.merge(temp_mart_02,yield_kr,how='left',on='Date') 

len(temp_mart_02)

### 3. merge mart_tmp_04 with Korean stock index
temp_mart_03 = temp_mart_02
temp_mart_03 = pd.merge(temp_mart_03,stck_kospi,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_100,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_autos,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_energy,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_IT,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_semicon,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_bank,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_build,how='left',on='Date')
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_healthcare,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_insurance,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_machinery,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_steel,how='left',on='Date')
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_stock,how='left',on='Date') 
temp_mart_03 = pd.merge(temp_mart_03,stck_krx_transportation,how='left',on='Date') 

len(temp_mart_03)

### 4. merge mart_tmp_14 with forex variable
temp_mart_04 = temp_mart_03
temp_mart_04 = pd.merge(temp_mart_04,fx_krw_aud,how='left',on='Date') 
temp_mart_04 = pd.merge(temp_mart_04,fx_krw_cny,how='left',on='Date') 
temp_mart_04 = pd.merge(temp_mart_04,fx_krw_gbp,how='left',on='Date') 
temp_mart_04 = pd.merge(temp_mart_04,fx_krw_eur,how='left',on='Date') 

len(temp_mart_04)
temp_mart_04.head(3)
temp_mart_04.tail(3)

### 5. merge mart_tmp_24 with US Dollar index variables
temp_mart_05 = temp_mart_04
temp_mart_05 = pd.merge(temp_mart_05,usd_index,how='left',on='Date') 

len(temp_mart_05)
temp_mart_05.head(3)
temp_mart_05.tail(3)

### 6. merge mart_tmp_31 with commodity price variables
temp_mart_06 = temp_mart_05
temp_mart_06 = pd.merge(temp_mart_06,cmd_gold,how='left',on='Date') 
temp_mart_06 = pd.merge(temp_mart_06,cmd_oil_brent,how='left',on='Date') 
temp_mart_06 = pd.merge(temp_mart_06,cmd_oil_wti,how='left',on='Date') 
temp_mart_06 = pd.merge(temp_mart_06,cmd_gas,how='left',on='Date') 
temp_mart_06 = pd.merge(temp_mart_06,cmd_silver,how='left',on='Date') 

len(temp_mart_06)
temp_mart_06.head(3)
temp_mart_06.tail(3)


# 7. merge mart_tmp_47 with Foreign stock index variables 
temp_mart_07 = temp_mart_06
temp_mart_07 = pd.merge(temp_mart_07,stck_cac40,how='left',on='Date') 
temp_mart_07 = pd.merge(temp_mart_07,stck_dax,how='left',on='Date') 
temp_mart_07 = pd.merge(temp_mart_07,stck_nasdaq,how='left',on='Date') 
temp_mart_07 = pd.merge(temp_mart_07,stck_nikkei,how='left',on='Date') 
temp_mart_07 = pd.merge(temp_mart_07,stck_nyse,how='left',on='Date') 
temp_mart_07 = pd.merge(temp_mart_07,stck_snp500,how='left',on='Date') 
temp_mart_07 = pd.merge(temp_mart_07,stck_ssec,how='left',on='Date') 

len(temp_mart_07)
temp_mart_07.head(3)
temp_mart_07.tail(3)


# 8. merge mart_tmp_57 with interest rate variables #
temp_mart_08 = temp_mart_07

temp_mart_08 = pd.merge(temp_mart_08,yield_jp,how='left',on='Date') 
temp_mart_08 = pd.merge(temp_mart_08,yield_fr,how='left',on='Date') 
# temp_mart_08 = pd.merge(temp_mart_08,yield_nz,how='left',on='Date') 
temp_mart_08 = pd.merge(temp_mart_08,yield_uk,how='left',on='Date') 

temp_mart_08 = pd.merge(temp_mart_08,yield_us,how='left',on='Date') 
temp_mart_08 = pd.merge(temp_mart_08,yield_gr,how='left',on='Date') 
temp_mart_08 = pd.merge(temp_mart_08,yield_hk,how='left',on='Date') 
temp_mart_08 = pd.merge(temp_mart_08,yield_chn,how='left',on='Date') 


# 9. merge mart_tmp_66 with forex variables #
temp_mart_09 = temp_mart_08
temp_mart_09 = pd.merge(temp_mart_09,fx_aud_usd,how='left',on='Date') 
temp_mart_09 = pd.merge(temp_mart_09,fx_cad_usd,how='left',on='Date') 
temp_mart_09 = pd.merge(temp_mart_09,fx_cny_usd,how='left',on='Date') 
temp_mart_09 = pd.merge(temp_mart_09,fx_eur_usd,how='left',on='Date') 
temp_mart_09 = pd.merge(temp_mart_09,fx_gbp_usd,how='left',on='Date') 
temp_mart_09 = pd.merge(temp_mart_09,fx_jpy_usd,how='left',on='Date') 
temp_mart_09 = pd.merge(temp_mart_09,fx_nzd_usd,how='left',on='Date') 

temp_mart_09.to_csv('data_mart.csv')


# =============================================================== #
# 01. One day shift to use the available data at the predict time #
# =============================================================== #
# 1. date shift
temp_mart_09 = temp_mart_09.sort(['Date'], ascending=[1])
temp_mart_09.head()
temp_mart_09.tail(3)
temp_mart_10 = temp_mart_09.reset_index(level=0, inplace=False).drop('index',axis=1) 
temp_mart_10.head(2)
temp_mart_10.tail(3)

shift_x = temp_mart_10[temp_mart_10.columns[2:]].iloc[1:]
shift_x = shift_x.reset_index(level=0, inplace=False).drop('index',axis=1)  
shift_x.head(2)

temp_mart_10[temp_mart_10.columns[2:]]= shift_x

temp_mart_10 = temp_mart_10.iloc[0:len(temp_mart_10 )-1]
temp_mart_10.head(2)

# 2. missing data



# ============================================ #
# 02. Fill the missing with the previous value #
# ============================================ #
len(temp_mart_10)
null_col_chek = temp_mart_10.isnull().sum()


temp_mart_11 = temp_mart_10[temp_mart_10['Date']< '2017-05-01']
len(temp_mart_11)
temp_mart_11.isnull().sum()
temp_mart_11.to_csv('data_mart2.csv')


drop_list =[]
for i in range(len(temp_mart_11)):
    if temp_mart_11.iloc[i].isnull().sum() > len(temp_mart_11.columns)*0.6:
        drop_list.append(False)
    else :
        drop_list.append(True)

temp_mart_11['is_keep'] = drop_list
temp_mart_12 = temp_mart_11[temp_mart_11['is_keep']==True]
temp_mart_12 = temp_mart_12.drop('is_keep', axis=1)

temp_mart_12.head()
temp_mart_12.to_csv('data_mart3.csv')



"""
# nan 값 
for col in temp_mart_12.columns.values:
    if temp_mart_12[col].isnull().sum() > 1 :
        print(col)
"""
import numpy as np
import pandas as pd
import math

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


# making diff
diff = [temp_mart_12['f_krw_usd'][i] - temp_mart_12['f_krw_usd'][i-1] for i in range(1,len(temp_mart_12))]
diff.append(0)
temp_mart_12['diff'] = diff

# model, valid
model_data = temp_mart_12[temp_mart_12['Date']< '2016-01-01']

valid_data = temp_mart_12[temp_mart_12['Date']>= '2016-01-01']

len(model_data) # 1036 , 75%
len(valid_data) # 347, 25%

X = temp_mart_12.ix[:,temp_mart_12.columns != 'f_krw_usd']
X = X.drop('Date',axis=1)
y = temp_mart_12['f_krw_usd']


# train,test 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3, 
                                                    random_state=100)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(scaled_X,scaled_y)

coeff = reg.coef_

model_data.columns

# import statsmodels.formula.api as sm
import statsmodels.api as sm

X_train = sm.add_constant(X_train)
model = sm.GLS(y_train, X_train)
results = model.fit()
print(results.summary())


import seaborn as sns
sns.lmplot(x='f_krw_usd',y='f_cad_usd',data = model_data)

from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale

model_data2 = model_data.drop('Date',axis=1)
scaled_X = pd.DataFrame(robust_scale(X_train),columns = X_train.columns)
scaled_y = robust_scale(y_train)

# scaled_data fit
scaled_X = sm.add_constant(scaled_X)
model = sm.GLS(scaled_y, scaled_X)
model = sm.GLS(scaled_y, scaled_X)
results = model.fit()
print(results.summary())

####
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

model = forward_selected(model_data2, 'f_krw_usd')

print(model.model.formula)
# sl ~ rk + yr + 1

print(model.rsquared_adj)
# 0.835190760538