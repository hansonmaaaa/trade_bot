import json
import sys
import time
import nasdaqdatalink
import requests
import pandas as pd
import os
from datetime import datetime, timedelta

from google.cloud import bigquery, bigquery_storage
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types
from dotenv import load_dotenv
load_dotenv('./.env')

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()

nasdaqdatalink.ApiConfig.api_key = os.getenv['nasdaq_api_key']

start_day = 1

### fundamentals
last_update = datetime.today() - timedelta(days = start_day)
last_fundamental_dt = datetime.today() - timedelta(days = 365)
fundamentals = nasdaqdatalink.get_table('SHARADAR/SF1', \
    datekey={'gte':last_fundamental_dt.strftime("%Y-%m-%d")}, lastupdated={'gte':last_update.strftime("%Y-%m-%d")}, paginate=True)
fundamentals = fundamentals.reset_index(drop = True)

table_id = 'boreal-pride-417020.financials.fundamentals_update'
job_config = bigquery.LoadJobConfig(
    #schema = [
    #bigquery.SchemaField("filingDate", bigquery.enums.SqlTypeNames.DATE),
    #bigquery.SchemaField("periodEndDate", bigquery.enums.SqlTypeNames.DATE)],
    write_disposition="WRITE_TRUNCATE",
)
job = client.load_table_from_dataframe(
                fundamentals, table_id, job_config=job_config
            )
job.result()

query = '''
delete from financials.sharadar_fundamentals p
where concat(p.ticker, '-', p.dimension,'-', FORMAT_DATE('%Y-%m-%d', p.datekey)) 
in 
(select concat(u.ticker, '-', u.dimension,'-', FORMAT_DATE('%Y-%m-%d', u.datekey)) 
from financials.fundamentals_update u )
'''
query_job = client.query(
    query
)  
query_job.result()

query = '''
CREATE OR REPLACE table `boreal-pride-417020.financials.sharadar_fundamentals` as (
select * from financials.sharadar_fundamentals
union all
select f.* EXCEPT(fiscalperiod) from financials.fundamentals_update f
)

'''
query_job = client.query(
    query
)  
query_job.result()

query = ''' truncate table boreal-pride-417020.financials.fundamentals_update'''
query_job = client.query(
    query
)  

### prices
for i in range(start_day, 0, -1):
    print(i)
    last_update = datetime.today() - timedelta(days = i)
    if last_update.weekday() > 4: # Mon-Fri are 0-4
        continue

    px_tbl = nasdaqdatalink.get_table('SHARADAR/SEP', \
        lastupdated=last_update.strftime("%Y-%m-%d"), paginate=True)
    px_tbl = px_tbl.reset_index(drop = True)

    #nasdaqdatalink.export_table('SHARADAR/SEP', \
    #    lastupdated=last_update.strftime("%Y-%m-%d"))
    #px_tbl = pd.read_csv('SHARADAR_SEP_2_23de0ee850267b69fc756a9b6a67acea.csv')
    #px_tbl.date = pd.to_datetime(px_tbl.date)
    #px_tbl.lastupdated = pd.to_datetime(px_tbl.lastupdated)

    table_id = 'boreal-pride-417020.prices.px_update'
    job_config = bigquery.LoadJobConfig(
        #schema = [
        #bigquery.SchemaField("filingDate", bigquery.enums.SqlTypeNames.DATE),
        #bigquery.SchemaField("periodEndDate", bigquery.enums.SqlTypeNames.DATE)],
        write_disposition="WRITE_TRUNCATE",
    )
    job = client.load_table_from_dataframe(
                    px_tbl, table_id, job_config=job_config
                )
    job.result()

    query = '''
    delete from prices.px p
    where concat(p.ticker, '-', FORMAT_DATE('%Y-%m-%d', p.date)) 
    in 
    (select concat(u.ticker, '-', FORMAT_DATE('%Y-%m-%d', u.date)) 
    from prices.px_update u )

    '''
    query_job = client.query(
        query
    )  
    query_job.result()

    query = '''
    CREATE OR REPLACE table `boreal-pride-417020.prices.px` as (
    select * from prices.px
    union all
    select * from prices.px_update
    )'''
    query_job = client.query(
        query
    )  
    query_job.result()

    query = ''' truncate table boreal-pride-417020.prices.px_update'''
    query_job = client.query(
        query
    )  
    #query_job.result()


### daily valuations
for i in range(start_day, 0, -1):
    last_update = datetime.today() - timedelta(days = i)

    if last_update.weekday() > 4: # Mon-Fri are 0-4
        continue

    valuation_tbl = nasdaqdatalink.get_table('SHARADAR/DAILY', \
        lastupdated=last_update.strftime("%Y-%m-%d"), paginate=True)
    valuation_tbl = valuation_tbl.reset_index(drop = True)


    table_id = 'boreal-pride-417020.prices.daily_valuations_update'
    job_config = bigquery.LoadJobConfig(
        #schema = [
        #bigquery.SchemaField("filingDate", bigquery.enums.SqlTypeNames.DATE),
        #bigquery.SchemaField("periodEndDate", bigquery.enums.SqlTypeNames.DATE)],
        write_disposition="WRITE_TRUNCATE",
    )
    job = client.load_table_from_dataframe(
                    valuation_tbl, table_id, job_config=job_config
                )
    job.result()

    query = '''
    delete from prices.daily_valuations p
    where concat(p.ticker, '-', FORMAT_DATE('%Y-%m-%d', p.date)) 
    in 
    (select concat(u.ticker, '-', FORMAT_DATE('%Y-%m-%d', u.date)) 
    from prices.daily_valuations_update u )

    '''
    query_job = client.query(
        query
    )  
    query_job.result()

    query = '''
    CREATE OR REPLACE table `boreal-pride-417020.prices.daily_valuations` as (
    select * from prices.daily_valuations
    union all
    select * from prices.daily_valuations_update
    )'''
    query_job = client.query(
        query
    )  
    query_job.result()

    query = ''' truncate table boreal-pride-417020.prices.daily_valuations_update'''
    query_job = client.query(
        query
    )  
    #query_job.result()

message = pd.DataFrame([datetime.now(), "nasdaq_data_download.py run success"]).transpose()
message.columns = ['datetime', 'message']
table_id = 'boreal-pride-417020.monitor.script_log'
job_config = bigquery.LoadJobConfig(
    #schema = [
    #bigquery.SchemaField("filingDate", bigquery.enums.SqlTypeNames.DATE),
    #bigquery.SchemaField("periodEndDate", bigquery.enums.SqlTypeNames.DATE)],
    write_disposition="WRITE_APPEND",
)
job = client.load_table_from_dataframe(
                message, table_id, job_config=job_config
            )
job.result()
