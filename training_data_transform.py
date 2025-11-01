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

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()


def create_training_data(project_id, dataset_id, read_table_id, write_table_id):
    
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

    all_data.loc[:, 'is_top_200_next_2month'] = 0
    all_data.loc[all_data.return_rank <= 200, 'is_top_200_next_2month'] = 1

    all_data.loc[:, 'return_rank'] = all_data.groupby('date')['next_3m_return'].rank(ascending = False, method = 'dense')

    all_data.loc[:, 'is_top_50_next_3month'] = 0
    all_data.loc[all_data.return_rank <= 50, 'is_top_50_next_3month'] = 1

    all_data.loc[:, 'is_top_100_next_3month'] = 0
    all_data.loc[all_data.return_rank <= 100, 'is_top_100_next_3month'] = 1

    all_data.loc[:, 'return_rank'] = all_data.groupby('date')['next_6m_return'].rank(ascending = False, method = 'dense')

    all_data.loc[:, 'is_top_50_next_6month'] = 0
    all_data.loc[all_data.return_rank <= 50, 'is_top_50_next_6month'] = 1

    all_data.loc[:, 'is_top_100_next_6month'] = 0
    all_data.loc[all_data.return_rank <= 100, 'is_top_100_next_6month'] = 1


    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(
                    all_data, write_table_id, job_config=job_config
                )
    
def create_training_data_removeNA(project_id, dataset_id, read_table_id, write_table_id):
    
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

    #remove rows with > 50% NAs
    threshold = round(.5 * len(incl), 0)
    row_nas = all_data.apply(lambda x: x.isna().sum(), axis=1)
    row_nas = pd.DataFrame(row_nas, columns = ['nan_count']) 
    row_nas.loc[:, 'keep'] = True
    row_nas.loc[row_nas.nan_count > threshold, 'keep'] = False

    all_data = all_data.loc[row_nas.keep.tolist(), :]  

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

    all_data.loc[:, 'is_top_200_next_2month'] = 0
    all_data.loc[all_data.return_rank <= 200, 'is_top_200_next_2month'] = 1

    all_data.loc[:, 'return_rank'] = all_data.groupby('date')['next_3m_return'].rank(ascending = False, method = 'dense')

    all_data.loc[:, 'is_top_50_next_3month'] = 0
    all_data.loc[all_data.return_rank <= 50, 'is_top_50_next_3month'] = 1

    all_data.loc[:, 'is_top_100_next_3month'] = 0
    all_data.loc[all_data.return_rank <= 100, 'is_top_100_next_3month'] = 1

    all_data.loc[:, 'return_rank'] = all_data.groupby('date')['next_6m_return'].rank(ascending = False, method = 'dense')

    all_data.loc[:, 'is_top_50_next_6month'] = 0
    all_data.loc[all_data.return_rank <= 50, 'is_top_50_next_6month'] = 1

    all_data.loc[:, 'is_top_100_next_6month'] = 0
    all_data.loc[all_data.return_rank <= 100, 'is_top_100_next_6month'] = 1


    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(
                    all_data, write_table_id, job_config=job_config
                )

## set postfix
postfix =  datetime.today() + timedelta(days = 1)
postfix = postfix.strftime("%Y%m")

### create training data set: raw_training_data_top60pct
project_id = "boreal-pride-417020"
dataset_id = "training_data"
new_table_prefix ="raw_training_data_top60pct"

new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
with all_data as (
select
p.*,
f.* EXCEPT(ticker, date, dimension, lastupdated)
from transformed.monthly_px_technicals p
left join transformed.monthly_financials_filled f on p.ticker = f.ticker and p.date = f.date
)
select
d.*
from all_data d 
left join `transformed.marketcap_volume_rank` t on t.ticker = d.ticker and d.date = t.date 
where d.date <t.last_date and  t.volume_12m_pct_rank < 0.8 and t.mcap_pct_rank <= 0.6
order by d.ticker, d.date
)
'''.format(new_table_id)

query_job = client.query(
    query
)  
query_job.result()

# read/write config
read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data(project_id, dataset_id, read_table_id, write_table_id)

# biotech only
new_table_prefix ="training_data_top60pct_biotech"
new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
select
d.*
from training_data.training_data_top60pct_{} d
left join `prices.tickers_tbl` t on d.ticker = t.ticker
where t.famaindustry = 'Pharmaceutical Products'
)
'''.format(new_table_id, postfix)

query_job = client.query(
    query
)  
query_job.result()


# biotech exclude recalc
new_table_prefix ="raw_training_data_top60pct_ex_biotech"
new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
select
d.*
from `training_data.raw_training_data_top60pct_{}` d
left join `prices.tickers_tbl` t on d.ticker = t.ticker
where t.famaindustry != 'Pharmaceutical Products'
)
'''.format(new_table_id, postfix)

query_job = client.query(
    query
)  
query_job.result()

read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data_removeNA(project_id, dataset_id, read_table_id, write_table_id)




### create training data set: raw_training_data
project_id = "boreal-pride-417020"
dataset_id = "training_data"
new_table_prefix ="raw_training_data"

new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
with all_data as (
select
p.*,
f.* EXCEPT(ticker, date, dimension, lastupdated)
from transformed.monthly_px_technicals p
left join transformed.monthly_financials_filled f on p.ticker = f.ticker and p.date = f.date
)
select
d.*
from all_data d 
left join `transformed.marketcap_volume_rank` t on t.ticker = d.ticker and d.date = t.date 
where d.date <t.last_date and  t.volume_12m_pct_rank < 0.8 and t.mcap_pct_rank <= 0.9
order by d.ticker, d.date
)
'''.format(new_table_id)

query_job = client.query(
    query
)  
query_job.result()


# read/write config
read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data(project_id, dataset_id, read_table_id, write_table_id)


### create training data set: raw_training_data_top60pct_ipo_delay
project_id = "boreal-pride-417020"
dataset_id = "training_data"
new_table_prefix ="raw_training_data_top60pct_ipo_delay"

new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
with all_data as (
select
p.*,
f.* EXCEPT(ticker, date, dimension, lastupdated)
from transformed.monthly_px_technicals p
left join transformed.monthly_financials_filled f on p.ticker = f.ticker and p.date = f.date
)
select
d.*
from all_data d 
left join `transformed.marketcap_volume_rank` t on t.ticker = d.ticker and d.date = t.date 
where d.date > DATE_ADD(t.first_date, INTERVAL 90 DAY) and d.date <t.last_date 
and t.volume_12m_pct_rank < 0.8 and t.mcap_pct_rank <= 0.6
order by d.ticker, d.date
)
'''.format(new_table_id)

query_job = client.query(
    query
)  
query_job.result()


# read/write config
read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data(project_id, dataset_id, read_table_id, write_table_id)


# biotech only
new_table_prefix ="training_data_top60pct_ipo_delay_biotech"
new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
select
d.*
from `training_data.training_data_top60pct_ipo_delay_{}` d
left join `prices.tickers_tbl` t on d.ticker = t.ticker
where t.famaindustry = 'Pharmaceutical Products'
)
'''.format(new_table_id, postfix)

query_job = client.query(
    query
)  
query_job.result()


# biotech exclude
new_table_prefix ="training_data_top60pct_ipo_delay_ex_biotech_no_recalc"
new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
select
d.*
from `training_data.training_data_top60pct_ipo_delay_{}` d
left join `prices.tickers_tbl` t on d.ticker = t.ticker
where t.famaindustry != 'Pharmaceutical Products'
)
'''.format(new_table_id, postfix)

query_job = client.query(
    query
)  
query_job.result()

# biotech exclude recalc
new_table_prefix ="raw_training_data_top60pct_ipo_delay_ex_biotech"
new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
select
d.*
from `training_data.raw_training_data_top60pct_ipo_delay_{}` d
left join `prices.tickers_tbl` t on d.ticker = t.ticker
where t.famaindustry != 'Pharmaceutical Products'
)
'''.format(new_table_id, postfix)

query_job = client.query(
    query
)  
query_job.result()

read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data(project_id, dataset_id, read_table_id, write_table_id)




### create training data set: raw_training_data_top70pct_ipo_delay
project_id = "boreal-pride-417020"
dataset_id = "training_data"
new_table_prefix ="raw_training_data_top70pct_ipo_delay"

new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
with all_data as (
select
p.*,
f.* EXCEPT(ticker, date, dimension, lastupdated)
from transformed.monthly_px_technicals p
left join transformed.monthly_financials_filled f on p.ticker = f.ticker and p.date = f.date
)
select
d.*
from all_data d 
left join `transformed.marketcap_volume_rank` t on t.ticker = d.ticker and d.date = t.date 
where d.date > DATE_ADD(t.first_date, INTERVAL 90 DAY) and d.date <t.last_date 
and t.volume_12m_pct_rank < 0.8 and t.mcap_pct_rank <= 0.7
order by d.ticker, d.date
)
'''.format(new_table_id)

query_job = client.query(
    query
)  
query_job.result()


# read/write config
read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data(project_id, dataset_id, read_table_id, write_table_id)

### create training data set: raw_training_data_top80pct_ipo_delay
project_id = "boreal-pride-417020"
dataset_id = "training_data"
new_table_prefix ="raw_training_data_top80pct_ipo_delay"

new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
with all_data as (
select
p.*,
f.* EXCEPT(ticker, date, dimension, lastupdated)
from transformed.monthly_px_technicals p
left join transformed.monthly_financials_filled f on p.ticker = f.ticker and p.date = f.date
)
select
d.*
from all_data d 
left join `transformed.marketcap_volume_rank` t on t.ticker = d.ticker and d.date = t.date 
where d.date > DATE_ADD(t.first_date, INTERVAL 90 DAY) and d.date <t.last_date 
and t.volume_12m_pct_rank < 0.8 and t.mcap_pct_rank <= 0.8
order by d.ticker, d.date
)
'''.format(new_table_id)

query_job = client.query(
    query
)  
query_job.result()


# read/write config
read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data(project_id, dataset_id, read_table_id, write_table_id)

# biotech only
new_table_prefix ="training_data_top80pct_ipo_delay_biotech"
new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
select
d.*
from `training_data.training_data_top80pct_ipo_delay_{}` d
left join `prices.tickers_tbl` t on d.ticker = t.ticker
where t.famaindustry = 'Pharmaceutical Products'
)
'''.format(new_table_id, postfix)

query_job = client.query(
    query
)  
query_job.result()


# biotech exclude
new_table_prefix ="training_data_top80pct_ipo_delay_ex_biotech_no_recalc"
new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
select
d.*
from `training_data.training_data_top80pct_ipo_delay_{}` d
left join `prices.tickers_tbl` t on d.ticker = t.ticker
where t.famaindustry != 'Pharmaceutical Products'
)
'''.format(new_table_id, postfix)

query_job = client.query(
    query
)  
query_job.result()

# biotech exclude recalc
new_table_prefix ="raw_training_data_top80pct_ipo_delay_ex_biotech"
new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
select
d.*
from `training_data.raw_training_data_top80pct_ipo_delay_{}` d
left join `prices.tickers_tbl` t on d.ticker = t.ticker
where t.famaindustry != 'Pharmaceutical Products'
)
'''.format(new_table_id, postfix)

query_job = client.query(
    query
)  
query_job.result()

read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data(project_id, dataset_id, read_table_id, write_table_id)



### create training data set: raw_training_data_ipo_delay
project_id = "boreal-pride-417020"
dataset_id = "training_data"
new_table_prefix ="raw_training_data_ipo_delay"

new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
with all_data as (
select
p.*,
f.* EXCEPT(ticker, date, dimension, lastupdated)
from transformed.monthly_px_technicals p
left join transformed.monthly_financials_filled f on p.ticker = f.ticker and p.date = f.date
)
select
d.*
from all_data d 
left join `transformed.marketcap_volume_rank` t on t.ticker = d.ticker and d.date = t.date 
where d.date > DATE_ADD(t.first_date, INTERVAL 90 DAY) and d.date <t.last_date 
and t.volume_12m_pct_rank < 0.8 and t.mcap_pct_rank <= 0.9
order by d.ticker, d.date
)
'''.format(new_table_id)

query_job = client.query(
    query
)  
query_job.result()


# read/write config
read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data(project_id, dataset_id, read_table_id, write_table_id)


### create training data set: raw_training_data_top60pct_ipo_delay
project_id = "boreal-pride-417020"
dataset_id = "training_data"
new_table_prefix ="raw_training_data_top60pct_9m_delay"

new_table_id = dataset_id + '.'+ new_table_prefix + '_' + postfix

query = '''
CREATE OR REPLACE table {} as (
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
where d.date > DATE_ADD(t.first_date, INTERVAL 90 DAY) and d.date <t.last_date 
and t.volume_12m_pct_rank < 0.8 and t.mcap_pct_rank <= 0.6
order by d.ticker, d.date
)
'''.format(new_table_id)

query_job = client.query(
    query
)  
query_job.result()


# read/write config
read_table_id = new_table_prefix + '_' + postfix
write_table_id = dataset_id+'.'+ new_table_prefix[4:] + '_' + postfix

create_training_data(project_id, dataset_id, read_table_id, write_table_id)
