import pandas as pd
pd.options.mode.chained_assignment = None
import os
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd
import numpy as np
import argparse

from google.cloud import bigquery, bigquery_storage
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types
from functions import read_table

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()


### create training data set: raw_training_data_top60pct

def raw_training_data_query(new_table_id, percentage):
    query = '''
    CREATE OR REPLACE table {write_table} as (
    with all_data as (
    select
    p.*,
    f.* EXCEPT(ticker, date, dimension, lastupdated)
    from transformed.monthly_px_technicals p
    left join transformed.monthly_financials_filled f on p.ticker = f.ticker and p.date = f.date
    ),
    filtered_fundamentals as (
    select
    ticker,
    count(*) as num_records
    from financials.ART_fundamentals_cleaned
    group by ticker
    having num_records > 3
    )
    select
    d.*
    from all_data d 
    inner join filtered_fundamentals f on d.ticker = f.ticker
    left join `transformed.marketcap_volume_rank` t on t.ticker = d.ticker and d.date = t.date 
    where d.date > DATE_ADD(t.first_date, INTERVAL 90 DAY) and d.date <= t.last_date 
    and t.volume_12m_pct_rank < 0.8 and t.mcap_pct_rank <= {percentage}
    order by d.ticker, d.date
    )
    '''.format(write_table = new_table_id, percentage = percentage/100)

    query_job = client.query(
        query
    )  
    query_job.result()

def biotech_query(new_table_id):
        query = '''
        CREATE OR REPLACE table {} as (
        select
        d.*
        from training_data.training_data_top60pct_ipo_delay d
        left join `prices.tickers_tbl` t on d.ticker = t.ticker
        where t.famaindustry = 'Pharmaceutical Products'
        )
        '''.format(new_table_id)

        query_job = client.query(
            query
        )  
        query_job.result()

def create_training_data(project_id, dataset_id, read_table_id, write_table_id, append_history):
    
    bqstorageclient = bigquery_storage.BigQueryReadClient()
    raw_data = read_table(bqstorageclient, project_id, dataset_id, read_table_id)

    # Get default variables
    client = bigquery.Client()
    query_job = client.query("""
    SELECT distinct variable
    FROM train.default_model_variables
    where incl = 1                     
    """)

    incl = query_job.result().to_dataframe()
    incl = incl.variable.values

    all_data = pd.concat([raw_data.loc[:, ['date','ticker', 'next_month_return']], raw_data.loc[:, incl]],  axis = 1)
    if 'px_pct_2m' not in incl:
        all_data = pd.concat([all_data, raw_data.loc[:, 'px_pct_2m']], axis = 1)
    if 'px_pct_3m' not in incl:
        all_data = pd.concat([all_data, raw_data.loc[:, 'px_pct_3m']], axis = 1)
    if 'px_pct_6m' not in incl:
        all_data = pd.concat([all_data, raw_data.loc[:, 'px_pct_6m']], axis = 1)

    all_data = all_data.sort_values(by=['ticker', 'date'])

    all_data = all_data.replace([np.inf, -np.inf], np.nan)
    all_data = all_data.drop_duplicates(subset = ['ticker', 'date'])

    ### count NAs
    for x in incl:
        this_col = all_data.loc[all_data[x].isnull(), x]
        pct = len(this_col) / len(all_data)
        if pct > .15:
            print(str(x), ': ', str(pct))

    all_data.loc[:, 'next_2m_return'] = all_data.groupby('ticker')['px_pct_2m'].shift(-2)
    all_data.loc[:, 'next_3m_return'] = all_data.groupby('ticker')['px_pct_3m'].shift(-3)
    all_data.loc[:, 'next_6m_return'] = all_data.groupby('ticker')['px_pct_6m'].shift(-6)

    all_data.loc[:, 'return_rank'] = all_data.groupby('date')['next_month_return'].rank(ascending = False, method = 'dense')

    all_data.loc[:, 'is_top_50_next_month'] = 0
    all_data.loc[all_data.return_rank <= 50, 'is_top_50_next_month'] = 1

    all_data.loc[:, 'is_top_100_next_month'] = 0
    all_data.loc[all_data.return_rank <= 100, 'is_top_100_next_month'] = 1

    all_data.loc[:, 'is_top_200_next_month'] = 0
    all_data.loc[all_data.return_rank <= 200, 'is_top_200_next_month'] = 1

    all_data.loc[:, 'is_top_300_next_month'] = 0
    all_data.loc[all_data.return_rank <= 300, 'is_top_300_next_month'] = 1

    all_data.loc[:, 'return_rank'] = all_data.groupby('date')['next_2m_return'].rank(ascending = False, method = 'dense')

    all_data.loc[:, 'is_top_50_next_2month'] = 0
    all_data.loc[all_data.return_rank <= 50, 'is_top_50_next_2month'] = 1

    all_data.loc[:, 'is_top_100_next_2month'] = 0
    all_data.loc[all_data.return_rank <= 100, 'is_top_100_next_2month'] = 1

    all_data.loc[:, 'return_rank'] = all_data.groupby('date')['next_3m_return'].rank(ascending = False, method = 'dense')

    all_data.loc[:, 'is_top_50_next_3month'] = 0
    all_data.loc[all_data.return_rank <= 50, 'is_top_50_next_3month'] = 1

    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(
                    all_data, write_table_id, job_config=job_config
                )
    
    ## append to historical table
    if append_history == True: 
        all_data.loc[:, 'run_date'] = datetime.now()

        historical_table_id = write_table_id+'_historical'
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = client.load_table_from_dataframe(
                        all_data, historical_table_id, job_config=job_config
                    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script that adds append history from CMD"
    )
    parser.add_argument('--percentage', default=60, type=int, help = "run over top X percentage in terms of marketcap")
    parser.add_argument('--append_history', default=False, action='store_true')
    args = parser.parse_args()
    append_history_flag = args.append_history
    percentage = args.percentage

    ### create raw training data set: raw_training_data_top60pct_ipo_delay
    project_id = "boreal-pride-417020"
    dataset_id = "training_data"
    new_table_prefix ="raw_training_data_top" + str(percentage) +"pct_ipo_delay"

    new_table_id = dataset_id + '.'+ new_table_prefix 
    
    raw_training_data_query(new_table_id, percentage)

    ### create training data set: training_data_top60pct_ipo_delay
    read_table_id = new_table_prefix 
    write_table_id = dataset_id+'.'+ new_table_prefix[4:] 

    create_training_data(project_id, dataset_id, read_table_id, write_table_id, append_history_flag)

    ## biotech only
    #new_table_prefix ="training_data_top60pct_ipo_delay_biotech"
    #new_table_id = dataset_id + '.'+ new_table_prefix 

    #biotech_query(new_table_id)
    
    #### create training data set: raw_training_data
    #new_table_prefix ="raw_training_data"
    #new_table_id = dataset_id + '.'+ new_table_prefix 
    #raw_training_data_query(new_table_id)

    ## create training data set: training_data
    #read_table_id = new_table_prefix 
    #write_table_id = dataset_id+'.'+ new_table_prefix[4:] 

    #create_training_data(project_id, dataset_id, read_table_id, write_table_id, True)
