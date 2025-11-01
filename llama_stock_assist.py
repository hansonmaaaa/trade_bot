import os
import pandas as pd
import time
import random
from datetime import datetime, timedelta
import numpy as np

from google.cloud import bigquery, bigquery_storage
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types

from meta_ai_api import MetaAI

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()


def source_to_text(source):

    if len(source) > 0:
        link = [x.get('link') for x in source]
        return ', '.join(link)
    else:
        return 'NA'

latest_date = datetime.today() + pd.offsets.MonthEnd(-1)
query = '''
select 
t.name as company_name,
t.location,
m.marketcap,
t.famaindustry,
t.sector,
t.industry,
ptf.*,
d.px_avg_mom,
d.px_pct_3m,
d.px_pct_6m,
d.px_yoy,
d.px_vol_3m,
d.volume_yoy,
d.revenue_yoy,
d.opmargin,
d.opmargin_dyoy,
d.ps,
d.interest_ebit,
d.capex_asset,
d.capex_asset_dyoy

from prod.backtest_ptfs_combo ptf
left join prices.tickers_tbl t on ptf.ticker = t.ticker
left join transformed.marketcap_volume_rank m on ptf.ticker = m.ticker and ptf.date = m.date
left join training_data.training_data_top40pct_ipo_delay d on ptf.ticker = d.ticker and ptf.date = d.date
where ptf.date = '{date}'
order by ptf.date desc, ptf.ComboRank asc

'''.format(date = latest_date.strftime("%Y-%m-%d"))

query_job = client.query(query)
raw_data = query_job.result().to_dataframe()

cutoff_date = datetime.today() + pd.offsets.MonthEnd(-3)
query = """
select distinct ticker 
from prod.llama_stock_summary
where run_date >= '{date}'
""".format(date = cutoff_date.strftime("%Y-%m-%d"))

query_job = client.query(query)
existing_data = query_job.result().to_dataframe()

df = raw_data.drop_duplicates(subset = 'ticker')

idx = np.in1d(df.ticker, existing_data.ticker)
df = df.loc[~idx, :]
df = df.reset_index(drop = True)

stock_summary = pd.DataFrame()

for i in range(0, 50):
    try: 
        company = df.loc[i, 'company_name']
        ticker = df.loc[i, 'ticker']

        ai = MetaAI()
        system_prompt = """
        Can you provide a brief overview of the {company} with ticker {ticker} 
        and state what are its main products that drive revenue?
        """.format(company = company, ticker = ticker)

        response = ai.prompt(message = system_prompt, new_conversation = True)
        output1 = response['message']
        source1 = source_to_text(response['sources'])

        time.sleep(random.randint(3,10))

        system_prompt = """
        You are an expert at stock analysis.
        Can you look through fundamental data and recent news in the past 1-3 years
        and give the top 3 reasons why {company}'s stock price has outperformed?
        """.format(company = company)

        response = ai.prompt(message = system_prompt)
        output2 = response['message']
        source2 = source_to_text(response['sources'])

        time.sleep(random.randint(3,10))

        system_prompt = """
        Who are the main competitors to {company}?
        """.format(company = company)

        response = ai.prompt(message = system_prompt)
        output3 = response['message']
        source3 = source_to_text(response['sources'])

        this_stock_summary = pd.DataFrame([company, ticker, output1, source1, output2, source2, output3, source3]).transpose()
        this_stock_summary.columns = ['Company_Name', 'Ticker', 'Overview', 'Overview_source', 'News', 'News_source', 'Competitors', 'Competitors_source']

        stock_summary = pd.concat([stock_summary, this_stock_summary])

        print(company)
        time.sleep(5)

    except Exception as e:
        print(e)
        #time.sleep(30)


stock_summary.loc[:, 'run_date'] = datetime.now()

historical_table_id = 'prod.llama_stock_summary'
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
job = client.load_table_from_dataframe(
                    stock_summary, historical_table_id, job_config=job_config
    )