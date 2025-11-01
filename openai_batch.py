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
from openai import OpenAI
import json

def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False
    
credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()

OPENAI_API_KEY = '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'

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

### Open AI calls
openai_client = OpenAI(
  organization='org-6IBLKTLIT7pUUdmMOCIpNQZB',
  project='proj_pBXaC0uM8dSzlPS9V9gzyfNx',
)

def main_openai_call(system_content, user_content):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    )
    return response.choices[0].message.content

i = 0
company = df.loc[i, 'company_name']
ticker = df.loc[i, 'ticker']
system_content = """ 
    You are a stock analysis expert. 
    You will be given a company name and its stock ticker. 
    Your job is to return the information that has the highest impact on the company's future stock price. 
    """
user_content =  """
    Can you provide a brief overview of the {company} with ticker {ticker} 
    and state what are its main products that drive revenue? 
    Next, under a separate section called 'Competitors', please list the main competitors to {company}. 
    Include the names of the competitors where possible.
    """.format(company = company, ticker = ticker)

response = main_openai_call(system_content, user_content)


#### Batch
system_content = """
    Role: 
        -You are a stock analysis expert. Your job is to evaluate each company's fundamental data and recent news
          and pick out the information that has the highest impact on the company's future stock price. 
    
    Context:
        -Below in the section **Data** are a pipe-separated list of company identifying data. The field names are as follows: Ticker, Company Name
    Task:
        -For each company identified by the Ticker and Company Name, provide a company overview. The company overview should summarize what the company does and what are its main products that drove revenue in the recent 1-3 years. 
        -Provide a competitors list which is a bulleted list of the company's main competitors.


    Output:
        -The output should be the following JSON structure
        ```json
            {'output':
                [
                    {'ticker': <Ticker>,
                     'company_name': <Company Name>
                     'summary': response
                    }               
                ]
            }

        ```

    Data:
"""

def create_user_content(this_data):
    user_content = """
    Ticker: {ticker}
    Company Name: {company_name}
    """.format(ticker = this_data.ticker, company_name = this_data.company_name)
    return user_content

BATCH_SIZE = 10
i = 0

text_df = df.iloc[i:i+BATCH_SIZE]
text_df['data'] = text_df.apply(lambda x: create_user_content(x), axis = 1)

user_content_input = text_df['data'].str.cat(sep='|')
prompt_with_data = system_content + user_content_input

thread = openai_client.beta.threads.create()
message = openai_client.beta.threads.messages.create(
    thread_id=thread.id,
    role = "user",
    content =prompt_with_data
)

run = openai_client.beta.threads.runs.create(
    thread_id = thread.id,
    assistant_id = assistant.id)

### parse runs

def parse_run_response(run):
    messages = openai_client.beta.threads.messages.list(thread_id=run.thread_id)
    for message in messages:
        response_raw = message.content[0].text.value
        start_index = response_raw.find("```json") + len("```json")
        end_index =response_raw.rfind("```")
        response_substr = response_raw[start_index:end_index]
        if start_index >=0 and end_index >= 0 and is_valid_json(response_substr) and 'output' in json.loads(response_substr):
            response = json.loads(response_substr)['output']
            df = pd.DataFrame(response)
            return(df)
        return(None)

response_df = parse_run_response(run)
