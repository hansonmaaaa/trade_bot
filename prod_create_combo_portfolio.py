import pandas as pd
pd.options.mode.chained_assignment = None
import os
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd
import numpy as np

from google.cloud import bigquery, bigquery_storage
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types
import itertools

from functions import *

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()

### Configs
# get training dataset file
project_id = "boreal-pride-417020"
dataset_id = "training_data"
read_table_prefix = "training_data_top40pct_ipo_delay"  ### vary
postfix = datetime.today().strftime("%Y%m")
read_table_id = dataset_id+'.'+read_table_prefix # read table

# save location
backtest_table_id = 'boreal-pride-417020.prod.backtest_ptfs_combo'  # save table
monthly_return_table_id = 'boreal-pride-417020.prod.backtest_ptfs_combo_monthly_return' 
performance_table_id = 'boreal-pride-417020.prod.backtest_ptfs_combo_performance'

### Configs end

query_job = client.query("""
    select * from {}     
    """.format(read_table_id))
raw_data = query_job.result().to_dataframe()

var_tbl = pd.read_csv('variables.csv')
all_vars = var_tbl.loc[:, 'variable']

## transform
vars_list = var_tbl.loc[var_tbl.reverse == 1, 'variable'].tolist()

X = raw_data.loc[:, vars_list] * -1
Y = raw_data.drop(columns = vars_list)

all_data = pd.concat([Y, X], axis = 1)

keys = all_data.loc[:, ['ticker', 'date']]

### clear tables
query_job = client.query("""
    truncate table {}     
    """.format(backtest_table_id))
query_job.result()

query_job = client.query("""
    truncate table {}     
    """.format(monthly_return_table_id))
query_job.result()

query_job = client.query("""
    truncate table {}     
    """.format(performance_table_id))
query_job.result()


### Run Backtest
query_job = client.query("""
    select * from train.combo_list
    """)
combo_list = query_job.result().to_dataframe()

for i in range(0, len(combo_list)):
    this_combo = combo_list.loc[i, ['var_1', 'var_2', 'var_3', 'var_4', 'var_5']]
    this_combo = this_combo[this_combo != 'NA'].tolist()

    model_id = combo_list.loc[i, 'model_id']

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

    if len(this_combo) == 3:
        all_portfolio.columns = ['ticker', 'date', 'var_1', 'var_2', 'var_3', 'ComboRank', 'next_month_return']
        all_portfolio.loc[:,"var_4"] = 0
        all_portfolio.loc[:,"var_5"] = 0

    elif len(this_combo) == 5:
        all_portfolio.columns = ['ticker', 'date', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'ComboRank', 'next_month_return']

    else: 
        print('error: check number of variables in combo')

    cost_per_trade = .01
    monthly_return = create_monthly_return(all_portfolio, cost_per_trade)

    # save backtest

    description = ','.join(this_combo)
    all_portfolio.loc[:, 'model_id'] = model_id
    all_portfolio.loc[:, 'description'] = description
    all_portfolio.loc[:, 'train_dataset'] = read_table_id
    all_portfolio['run_date'] = datetime.now()

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(
                    all_portfolio, backtest_table_id, job_config=job_config
                )

    # save monthly return
    monthly_return.loc[:, 'model_id'] = model_id
    monthly_return.loc[:, 'description'] = description
    monthly_return.loc[:, 'train_dataset'] = read_table_id
    monthly_return['run_date'] = datetime.now()

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(
                    monthly_return, monthly_return_table_id, job_config=job_config
                )

    # save backtest performance
    stat_tbl1 = create_performance_stats_v2(monthly_return, 3)
    stat_tbl2 = create_performance_stats_v2(monthly_return, 5)
    stat_tbl3 = create_performance_stats_v2(monthly_return, 10)

    stat_tbl = pd.concat([stat_tbl1, stat_tbl2])
    stat_tbl = pd.concat([stat_tbl, stat_tbl3])

    stat_tbl.loc[:, 'model_id'] =  model_id
    stat_tbl.loc[:, 'description'] = description

    stat_tbl = stat_tbl.loc[stat_tbl.field != 'drawdown_dt', :]

    job_config = bigquery.LoadJobConfig(
        #schema = [ \
        #bigquery.SchemaField("5y_drawdown_dt", bigquery.enums.SqlTypeNames.DATE), \
        #bigquery.SchemaField("drawdown_dt", bigquery.enums.SqlTypeNames.DATE)], \
        write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(
                    stat_tbl, performance_table_id, job_config=job_config
                )

## Figure out best ptf by recent history
lookback_period = 20
query_job = client.query("""
        with tb1 as (
    select 
    *, 
    lag(total_return_idx,1) over (partition by model_id order by date) / lag(total_return_idx, {period}) over (partition by model_id order by date)-1 as period_return
    from boreal-pride-417020.prod.backtest_ptfs_combo_monthly_return
    order by model_id, date
    )
    select
    date,
    model_id,
    description,
    total_return,
    period_return
    from tb1
    QUALIFY ROW_NUMBER() over (partition by date order by period_return desc) <= 1
    order by date     
        """.format(period = lookback_period))
return_tbl = query_job.result().to_dataframe()

meta_return_table_id = 'boreal-pride-417020.prod.meta_combo_ptfs'
job_config = bigquery.LoadJobConfig(
        #schema = [ \
        #bigquery.SchemaField("5y_drawdown_dt", bigquery.enums.SqlTypeNames.DATE), \
        #bigquery.SchemaField("drawdown_dt", bigquery.enums.SqlTypeNames.DATE)], \
        write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(
                    return_tbl, meta_return_table_id, job_config=job_config
)

