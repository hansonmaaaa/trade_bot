import pandas as pd
pd.options.mode.chained_assignment = None
import os
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd
import numpy as np

from google.cloud import bigquery, bigquery_storage
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types
from functions import read_table
import itertools

from functions import *

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()

query_job = client.query("""
    select * from training_data.training_data_top60pct_ipo_delay_202407      
    """)
raw_data = query_job.result().to_dataframe()

var_tbl = pd.read_csv('variables.csv')
all_vars = var_tbl.loc[:, 'variable']

## transform
vars_list = var_tbl.loc[var_tbl.reverse == 1, 'variable'].tolist()

X = raw_data.loc[:, vars_list] * -1
Y = raw_data.drop(columns = vars_list)

all_data = pd.concat([Y, X], axis = 1)

query_job = client.query("""
    select * from train.model_variables_combo_5var_short
    """)
combos = query_job.result().to_dataframe()

keys = all_data.loc[:, ['ticker', 'date']]
backtest_result_table_id = 'train.combo_5var_backtest_performance_short'

results = pd.DataFrame()
for i in range(0, len(combos)):
    this_combo = combos.loc[i, ['var1', 'var2', 'var3', 'var4', 'var5']].tolist()

    values = all_data.loc[:, this_combo]

    this_data = pd.concat([keys, values], axis = 1)
    this_data = this_data.dropna()

    ranked_data = this_data.set_index(['ticker', 'date']).groupby(['date'])[this_combo].rank(ascending = False)
    new_col = [s + "_rank" for s in this_combo]
    ranked_data.columns = new_col
    ranked_data = ranked_data.reset_index()

    ranked_data.loc[:, 'ComboRank'] = ranked_data.loc[:, new_col].mean(axis = 1)

    ranked_data = ranked_data.sort_values(by=['date', 'ComboRank'])
    #ranked_data = ranked_data.merge(this_data, how = 'left', on = ['ticker', 'date'])

    ptf = create_combo_factor_ptf(ranked_data, 50)

    all_portfolio = ptf.merge(raw_data.loc[:, ['ticker', 'date', 'next_month_return']], how = 'left', on = ['ticker', 'date'])

    cost_per_trade = .01
    monthly_return = create_monthly_return(all_portfolio, cost_per_trade)
    end = len(monthly_return)-2
    start = end - 5*12
    sharpe_5y = monthly_return.loc[start:end, 'total_return'].mean()*12 / (monthly_return.loc[start:end, 'total_return'].std() * np.sqrt(12))

    stat_tbl = create_performance_stats(monthly_return, start, end)
    to_add =  pd.DataFrame(zip([ 'var_1', 'var_2', 'var_3', 'var_4', 'var_5'], this_combo))
    stat_tbl = pd.concat([stat_tbl, to_add])

    fnl_stat_tbl =stat_tbl.set_index(0).transpose()

    results = pd.concat([results, fnl_stat_tbl])
    results.loc[:, 'model_id'] = combos.loc[i, ['model_id']][0]

    if i % 1000 == 0:
        print(datetime.now())
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = client.load_table_from_dataframe(
                    results, backtest_result_table_id, job_config=job_config
                )
        results = pd.DataFrame()