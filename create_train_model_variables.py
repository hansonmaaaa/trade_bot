import pandas as pd
import os 
from google.cloud import bigquery, bigquery_storage
import numpy as np
import math 
from datetime import datetime

from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types
from google.cloud import bigquery_storage

import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from functions import * 

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
bqstorageclient = bigquery_storage.BigQueryReadClient()
client = bigquery.Client()

### config ##############################

# get model params

#model_result_table_id = 'boreal-pride-417020.train.model_results_classifier'
#start_model_id = 132  ### vary
#Y_field, xgb_params = get_model_tbl_from_id(model_result_table_id, start_model_id)

Y_field = 'is_top_100_next_month' ### vary 

xgb_params = {
    'n_estimators': 200,
    'max_depth': 3,
    'learning_rate': .15,
    'gamma':  40,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "scale_pos_weight": 15
}

# backtest configs
cutoff = 240 ### vary

# read/write config
project_id = "boreal-pride-417020"
dataset_id = "training_data"

read_table_prefix = "training_data_top60pct_ipo_delay"
postfix = datetime.today().strftime("%Y%m")
read_table_id = read_table_prefix + '_' + postfix

# save location
model_result_table_id = 'boreal-pride-417020.train.model_results_classifier'
model_variable_table_id = 'boreal-pride-417020.train.model_variables_classifier'
### config ends ##############################

raw_data = read_table(bqstorageclient, project_id, dataset_id, read_table_id)

query_job = client.query("""
   SELECT distinct variable
   FROM train.model_variables_classifier
   where model_id = 6  
   """)

incl = query_job.result().to_dataframe()
incl = incl.variable.values

incl = pd.read_csv('this_model_id_xvars.csv')
incl = incl.loc[incl.incl == 1, 'variable'].values

# create correlation table
corr_data = raw_data.loc[:, incl]

corr_tbl = corr_data.corr().reset_index()
corr_tbl= corr_tbl.rename(columns={'index':'Var1'})

corr_tbl = pd.melt(corr_tbl, id_vars = 'Var1')
corr_tbl.columns = ['Var1', 'Var2', 'correl']

corr_tbl = corr_tbl.loc[corr_tbl.Var1 != corr_tbl.Var2, :]
corr_tbl = corr_tbl.set_index(['Var1','Var2']).stack().drop_duplicates().unstack().reset_index()

# Figure out importance

all_data = pd.concat([raw_data.loc[:, ['ticker', 'date', 'next_month_return']], raw_data.loc[:, Y_field]], axis = 1)
all_data = pd.concat([all_data, raw_data.loc[:, incl]], axis = 1)
all_data = all_data.dropna(subset = ['date', 'ticker'])

date_list = all_data.loc[:, 'date'].sort_values().unique()
cutoff = len(date_list) - 8*12
train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
test_date = pd.DataFrame(date_list[cutoff:len(date_list)], columns = ['date'])

train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)
X_train = train.loc[:, incl]
Y_train = train.loc[:, Y_field]

test = all_data.merge(test_date, how = 'inner', on ='date').reset_index(drop=True)
X_test = test.loc[:, incl]
Y_test = test.loc[:, Y_field]

model = xgb.XGBClassifier(**xgb_params)
model.fit(X_train, Y_train)
pred = model.predict(X_test) 
score = f1_score(Y_test, pred)

# Importance calc

importance = model.get_booster().get_score(importance_type = 'gain')
gain_tbl = pd.DataFrame([importance], columns = importance.keys()).transpose()
gain_tbl = gain_tbl.reset_index()
gain_tbl.columns = ['variable', 'importance']

# add importance to corr
corr_tbl = corr_tbl.merge(gain_tbl, how = 'left', left_on = 'Var1', right_on = 'variable')
corr_tbl = corr_tbl.merge(gain_tbl, how = 'left', left_on = 'Var2', right_on = 'variable')
corr_tbl = corr_tbl.drop(columns = ['variable_x', 'variable_y'])

corr_tbl.columns = ['Var1', 'Var2', 'correl', 'importance_Var1', 'importance_Var2']

# remove high correlations
high_corr = corr_tbl.loc[abs(corr_tbl.correl) >= .90, :]
high_corr = high_corr.fillna(0)
high_corr.loc[:, 'keep_Var1'] = 1
high_corr.loc[high_corr.importance_Var2 > high_corr.importance_Var1, 'keep_Var1'] = 0

high = pd.concat([high_corr.loc[high_corr.keep_Var1 == 1, 'Var1'], high_corr.loc[high_corr.keep_Var1 == 0, 'Var2']], axis = 0)
high = high.drop_duplicates()

remove = pd.concat([high_corr.loc[:, 'Var1'], high_corr.loc[:, 'Var2']], axis = 0)
remove = remove.drop_duplicates()

idx = np.in1d(incl, remove)
low = incl[~idx]

var_list = low.tolist() + high.tolist()

### Write results
incl = var_list.copy()

all_data = pd.concat([raw_data.loc[:, ['ticker', 'date', 'next_month_return']], raw_data.loc[:, Y_field]], axis = 1)
all_data = pd.concat([all_data, raw_data.loc[:, incl]], axis = 1)
all_data = all_data.dropna(subset = ['date', 'ticker'])

date_list = all_data.loc[:, 'date'].sort_values().unique()
cutoff = len(date_list) - 8*12
train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
test_date = pd.DataFrame(date_list[cutoff:len(date_list)], columns = ['date'])

train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)
X_train = train.loc[:, incl]
Y_train = train.loc[:, Y_field]

test = all_data.merge(test_date, how = 'inner', on ='date').reset_index(drop=True)
X_test = test.loc[:, incl]
Y_test = test.loc[:, Y_field]

model = xgb.XGBClassifier(**xgb_params)
model.fit(X_train, Y_train)
pred = model.predict(X_test) 

# Importance calc
score = f1_score(Y_test, pred)
importance = model.get_booster().get_score(importance_type = 'gain')


model_id = write_model_to_db(Y_field, incl, xgb_params, score, importance, model_result_table_id, model_variable_table_id)

