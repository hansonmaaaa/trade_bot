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


### subset variables
query_job = client.query("""
    select * from train.combo_3var_backtest_performance
    order by sharp_5y desc
    limit 10 
    """)
combo_3var = query_job.result().to_dataframe()

var_tbl = pd.read_csv('variables.csv')
all_vars = var_tbl.loc[:, 'variable']

### create combos
all_combo_tbl = pd.DataFrame()

for i in range(0, len(combo_3var)):
    this_combo_3var = combo_3var.loc[i, ['var_1', 'var_2', 'var_3']].tolist()

    idx = np.in1d(all_vars, this_combo_3var)
    all_incl = all_vars[~idx]

    combos = [list(x) for x in itertools.combinations(all_incl.tolist(), 2)]

    first_list = [this_combo_3var] * len(combos)

    combo_tbl = pd.concat([pd.DataFrame(first_list), pd.DataFrame(combos)], axis = 1)
    combo_tbl.columns = ['var1', 'var2', 'var3', 'var4', 'var5']
    all_combo_tbl = pd.concat([all_combo_tbl, combo_tbl], axis = 0)

all_combo_tbl = all_combo_tbl.reset_index(drop=True)
all_combo_tbl = all_combo_tbl.reset_index()
all_combo_tbl = all_combo_tbl.rename(columns = {'index': 'model_id'})

backtest_result_table_id = 'train.model_variables_combo_5var_short'
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(
            all_combo_tbl, backtest_result_table_id, job_config=job_config
        )
