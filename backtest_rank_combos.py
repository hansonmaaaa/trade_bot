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
import plotly.express as px

from functions import *

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()

all_returns = pd.DataFrame()
for j in range(5, 61):
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
    total_return,
    period_return
    from tb1
    QUALIFY ROW_NUMBER() over (partition by date order by period_return desc) <= 1
    order by date     
        """.format(period = j))
    return_tbl = query_job.result().to_dataframe()

    return_tbl.loc[:, 'total_return_idx'] = 100.0

    for i in range(1, len(return_tbl)):
        return_tbl.loc[i, 'total_return_idx'] =  return_tbl.loc[i-1, 'total_return_idx'] * (1+return_tbl.loc[i, 'total_return'])

    return_tbl.loc[:, 'lookback_months'] = j
    all_returns = pd.concat([all_returns, return_tbl])

fig_tbl = all_returns.loc[all_returns.lookback_months.isin([20]), :]
fig = px.line(return_tbl, x="date", y="model_id",
              labels={'y': 'Total Return', 'x': 'Date'},
              hover_data = ['model_id', 'description', 'period_return'])

fig.update_layout(title='', xaxis_title='Date', yaxis_title='Total Return')

fig.show()