import os
import pandas as pd
import time
import random
from datetime import datetime, timedelta
import numpy as np

from google.cloud import bigquery, bigquery_storage
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types

import google.generativeai as genai

#### Configs ####
credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()
genai.configure(api_key='AIzaSyCOPhWj20fa2r1NMOfzSltRwoEIzUWTkM0')
#### Configs ####

# Tickers to get info for
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
from prod.gemini_stock_summary
where run_date >= '{date}'
""".format(date = cutoff_date.strftime("%Y-%m-%d"))

query_job = client.query(query)
existing_data = query_job.result().to_dataframe()

df = raw_data.drop_duplicates(subset = 'ticker')

idx = np.in1d(df.ticker, existing_data.ticker)
df = df.loc[~idx, :]
df = df.reset_index(drop = True)


#### LLM Model
# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
)

stock_summary = pd.DataFrame()

for i in range(0, len(df)):
    try: 
        company = df.loc[i, 'company_name']
        ticker = df.loc[i, 'ticker']

        chat_session = model.start_chat()

        system_prompt = """
        Can you provide a brief overview of the {company} with ticker {ticker} 
        and state what are its main products that drive revenue? 
        Next, under a separate section called 'Competitors', please list the main competitors to {company}. 
        Include the names of the competitors where possible.
        """.format(company = company, ticker = ticker)

        response = chat_session.send_message(system_prompt)
        output1 = response.text
        #print(response.text)

        system_prompt = """
        Next, under a separate section called 'News', please search through recent news in the past 18 months for {company} and give the top 3 reasons why {company}'s stock price has outperformed. 
        Only include reasons specific to the company and do not reference overall market conditions
        or general investor disclaimers 
        """.format(company = company, ticker = ticker)

        response = chat_session.send_message(system_prompt, tools='google_search_retrieval')
        output2 = response.text
        #print(response.text)

        this_stock_summary = pd.DataFrame([company, ticker, output1, output2]).transpose()
        this_stock_summary.columns = ['Company_Name', 'Ticker', 'Overview', 'News']

        stock_summary = pd.concat([stock_summary, this_stock_summary])
        
        print(company)

    except Exception as e:
        print(e)

stock_summary.loc[:, 'run_date'] = datetime.now()

historical_table_id = 'prod.gemini_stock_summary'
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
job = client.load_table_from_dataframe(
                    stock_summary, historical_table_id, job_config=job_config
    )