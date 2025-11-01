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

### prices join
query = '''
CREATE OR REPLACE table `boreal-pride-417020.transformed.monthly_avg_px_pt1` as (

with ordered_px as (
select * from `prices.px` p
where p.date < date(2010, 01, 01)
order by p.ticker, p.date desc
)
select
p.ticker,
extract(year from p.date) as year,
extract(month from p.date) as month,
LAST_DAY(max(p.date), month) as date,
round(avg(p.closeadj), 2) as px_close_avg,
round(avg(p.volume), 0) as volume,
ARRAY_AGG(closeadj)[OFFSET(0)] as px_close
from ordered_px p
group by ticker, year, month
order by ticker, year, month

)
'''
query_job = client.query(
    query
)  
query_job.result()

query = '''
CREATE OR REPLACE table `boreal-pride-417020.transformed.monthly_avg_px_pt2` as 
(with ordered_px as (
select * from `prices.px` p
where p.date >= date(2010, 01, 01) and p.date < date(2020, 01, 01)
order by p.ticker, p.date desc
)

select
p.ticker,
extract(year from p.date) as year,
extract(month from p.date) as month,
LAST_DAY(max(p.date), month) as date,
round(avg(p.closeadj), 2) as px_close_avg,
round(avg(p.volume), 0) as volume,
ARRAY_AGG(closeadj)[OFFSET(0)] as px_close
from ordered_px p
group by ticker, year, month
order by ticker, year, month

)
'''
query_job = client.query(
    query
)  
query_job.result()

query = '''
CREATE OR REPLACE table `boreal-pride-417020.transformed.monthly_avg_px_pt3` as 
(with ordered_px as (
select * from `prices.px` p
where p.date >= date(2020, 01, 01)
order by p.ticker, p.date desc
)

select
p.ticker,
extract(year from p.date) as year,
extract(month from p.date) as month,
LAST_DAY(max(p.date), month) as date,
round(avg(p.closeadj), 2) as px_close_avg,
round(avg(p.volume), 0) as volume,
ARRAY_AGG(closeadj)[OFFSET(0)] as px_close
from ordered_px p
group by ticker, year, month
order by ticker, year, month

)
'''
query_job = client.query(
    query
)  
query_job.result()


query = '''
CREATE OR REPLACE table `boreal-pride-417020.transformed.monthly_avg_px` as (
with tb1 as (
select * from `transformed.monthly_avg_px_pt1`
union all
select * from `transformed.monthly_avg_px_pt2`
union all
select * from `transformed.monthly_avg_px_pt3`
)
select
*
from tb1
order by ticker, date 
)
'''
query_job = client.query(
    query
)  
query_job.result()


### prices join

query = '''
CREATE OR REPLACE table `boreal-pride-417020.transformed.monthly_px_returns` as (
with tb2 as (
select
p.*,
lag(px_close, 1) over (partition by ticker order by year, month) as px_l1,
lag(px_close, 2) over (partition by ticker order by year, month) as px_l2,
lag(px_close, 3) over (partition by ticker order by year, month) as px_l3,
lag(px_close, 6) over (partition by ticker order by year, month) as px_l6,
lag(px_close, 12) over (partition by ticker order by year, month) as px_l12,
lag(px_close, 18) over (partition by ticker order by year, month) as px_l18,
lag(px_close_avg, 1) over (partition by ticker order by year, month) as px_avg_l1,
lag(px_close_avg, 3) over (partition by ticker order by year, month) as px_avg_l3,
lag(px_close_avg, 6) over (partition by ticker order by year, month) as px_avg_l6,
lag(px_close_avg, 12) over (partition by ticker order by year, month) as px_avg_l12,
lag(px_close_avg, 18) over (partition by ticker order by year, month) as px_avg_l18,
lag(volume, 1) over (partition by ticker order by year, month) as volume_l1,
lag(volume, 3) over (partition by ticker order by year, month) as volume_l3,
lag(volume, 6) over (partition by ticker order by year, month) as volume_l6,
lag(volume, 12) over (partition by ticker order by year, month) as volume_l12,
lag(volume, 18) over (partition by ticker order by year, month) as volume_l18
from `transformed.monthly_avg_px` p
)
select
ticker,
year,
month,
date,
px_close,
px_close_avg,
volume,
avg(volume) over (partition by ticker order by year, month rows between 5 preceding and current row) as avg_volume_6m,
avg(volume) over (partition by ticker order by year, month rows between 11 preceding and current row) as avg_volume_12m,
round(px_close / nullif(px_l1, 0) - 1, 6) as px_mom,
round(px_close / nullif(px_l2, 0) - 1, 6) as px_pct_2m,
round(px_close / nullif(px_l3, 0) - 1, 6) as px_pct_3m,
round(px_close / nullif(px_l6, 0) - 1, 6) as px_pct_6m,
round(px_close / nullif(px_l12,0) - 1, 6) as px_yoy,
round(px_close / nullif(px_l18,0) - 1, 6) as px_pct_18m,
round(px_close_avg / nullif(px_avg_l1, 0) - 1, 6) as px_avg_mom,
round(px_close_avg / nullif(px_avg_l3, 0) - 1, 6) as px_avg_pct_3m,
round(px_close_avg / nullif(px_avg_l6, 0) - 1, 6) as px_avg_pct_6m,
round(px_close_avg / nullif(px_avg_l12,0) - 1, 6) as px_avg_yoy,
round(px_close_avg / nullif(px_avg_l18,0) - 1, 6) as px_avg_pct_18m,
round(volume / nullif(volume_l1, 0) - 1, 6) as volume_mom,
round(volume / nullif(volume_l3, 0) - 1, 6) as volume_pct_3m,
round(volume / nullif(volume_l6, 0) - 1, 6) as volume_pct_6m,
round(volume / nullif(volume_l12,0) - 1, 6) as volume_yoy,
round(volume / nullif(volume_l18,0) - 1, 6) as volume_pct_18m
from tb2 p
order by ticker, date 
)

'''
query_job = client.query(
    query
)  
query_job.result()



query = '''
CREATE OR REPLACE table `boreal-pride-417020.transformed.monthly_px_technicals` as (
with tb1 as (
select
p.*,
lead(px_mom) over (partition by ticker order by date) as next_month_return,

avg(px_mom) over (partition by ticker order by date rows between 2 preceding and current row) as px_momentum_3m,
avg(px_mom) over (partition by ticker order by date rows between 5 preceding and current row) as px_momentum_6m,
avg(px_mom) over (partition by ticker order by date rows between 8 preceding and current row) as px_momentum_9m,
avg(px_avg_mom) over (partition by ticker order by date rows between 2 preceding and current row) as px_avg_momentum_3m,
avg(px_avg_mom) over (partition by ticker order by date rows between 5 preceding and current row) as px_avg_momentum_6m,
avg(px_avg_mom) over (partition by ticker order by date rows between 8 preceding and current row) as px_avg_momentum_9m,
stddev(px_mom) over (partition by ticker order by date rows between 2 preceding and current row) as px_vol_3m,
stddev(px_mom) over (partition by ticker order by date rows between 5 preceding and current row) as px_vol_6m,
stddev(px_mom) over (partition by ticker order by date rows between 8 preceding and current row) as px_vol_9m,
stddev(px_mom) over (partition by ticker order by date rows between 11 preceding and current row) as px_vol_12m,
stddev(px_mom) over (partition by ticker order by date rows between 17 preceding and current row) as px_vol_18m,
stddev(px_avg_mom) over (partition by ticker order by date rows between 2 preceding and current row) as px_avg_vol_3m,
stddev(px_avg_mom) over (partition by ticker order by date rows between 5 preceding and current row) as px_avg_vol_6m,
stddev(px_avg_mom) over (partition by ticker order by date rows between 8 preceding and current row) as px_avg_vol_9m,
stddev(px_avg_mom) over (partition by ticker order by date rows between 11 preceding and current row) as px_avg_vol_12m,
stddev(px_avg_mom) over (partition by ticker order by date rows between 17 preceding and current row) as px_avg_vol_18m
from transformed.monthly_px_returns p
), 
tb2 as (
select
*,
px_pct_3m / nullif(px_vol_12m,0) as px_chg_3m_vol_12m,
px_pct_6m / nullif(px_vol_12m,0) as px_chg_6m_vol_12m,
px_yoy / nullif(px_vol_12m,0) as px_chg_12m_vol_12m,

px_momentum_3m / nullif(px_vol_12m,0) as px_momentum_3m_vol_12m,
px_momentum_6m / nullif(px_vol_12m,0) as px_momentum_6m_vol_12m,
px_momentum_9m / nullif(px_vol_12m,0) as px_momentum_9m_vol_12m,

px_avg_momentum_3m / nullif(px_avg_vol_12m,0) as px_avg_momentum_3m_vol_12m,
px_avg_momentum_6m / nullif(px_avg_vol_12m,0) as px_avg_momentum_6m_vol_12m,
px_avg_momentum_9m / nullif(px_avg_vol_12m,0) as px_avg_momentum_9m_vol_12m,

lag(px_momentum_3m, 3) over (partition by ticker order by date) as px_momentum_3m_l3,
lag(px_momentum_6m, 3) over (partition by ticker order by date) as px_momentum_6m_l3,
lag(px_momentum_9m, 3) over (partition by ticker order by date) as px_momentum_9m_l3,

lag(px_avg_momentum_3m, 3) over (partition by ticker order by date) as px_avg_momentum_3m_l3,
lag(px_avg_momentum_6m, 3) over (partition by ticker order by date) as px_avg_momentum_6m_l3,
lag(px_avg_momentum_9m, 3) over (partition by ticker order by date) as px_avg_momentum_9m_l3

from tb1 t
),
tb3 as (
select
*,
px_momentum_3m - px_momentum_3m_l3 as px_momentum_3m_chg_3m,
px_momentum_6m - px_momentum_6m_l3 as px_momentum_6m_chg_3m,
px_momentum_9m - px_momentum_9m_l3 as px_momentum_9m_chg_3m,

px_avg_momentum_3m - px_avg_momentum_3m_l3 as px_avg_momentum_3m_chg_3m,
px_avg_momentum_6m - px_avg_momentum_6m_l3 as px_avg_momentum_6m_chg_3m,
px_avg_momentum_9m - px_avg_momentum_9m_l3 as px_avg_momentum_9m_chg_3m,

lag(px_momentum_3m_vol_12m, 3) over (partition by ticker order by date) as px_momentum_3m_vol_12m_l3,
lag(px_momentum_6m_vol_12m, 3) over (partition by ticker order by date) as px_momentum_6m_vol_12m_l3,
lag(px_momentum_9m_vol_12m, 3) over (partition by ticker order by date) as px_momentum_9m_vol_12m_l3

from tb2 t
)

select
*,
px_momentum_3m_vol_12m - px_momentum_3m_vol_12m_l3 as px_momentum_3m_vol_12m_chg_3m,
px_momentum_6m_vol_12m - px_momentum_6m_vol_12m_l3 as px_momentum_6m_vol_12m_chg_3m,
px_momentum_9m_vol_12m - px_momentum_9m_vol_12m_l3 as px_momentum_9m_vol_12m_chg_3m

from tb3 t

)
'''
query_job = client.query(
    query
)  
query_job.result()


### valuations 

query = '''
CREATE OR REPLACE table `boreal-pride-417020.transformed.monthly_avg_valuation` as (
select
ticker,
extract(year from date) as year,
extract(month from date) as month,
LAST_DAY(max(date), month) as date,
round(avg(ev), 2) as ev,
round(avg(evebit), 2) as evebit,
round(avg(marketcap), 2) as marketcap,
round(avg(pb), 2) as pb,
round(avg(pe), 2) as pe,
round(avg(ps), 2) as ps
from prices.daily_valuations
group by ticker, year, month
)
'''
query_job = client.query(
    query
)  
query_job.result()

### marketcap
query = '''
CREATE OR REPLACE table `boreal-pride-417020.transformed.marketcap_volume_rank` as (
with delist as (
select ticker,
max(date) as delist_date 
from financials.actions 
where action = 'delisted'
group by ticker
),
pricing as (select
ticker,
min(date) as first_date
from transformed.monthly_avg_px
group by ticker)

select 
v.*,
percent_rank() over (partition by v.date order by v.marketcap desc) as mcap_pct_rank,
p.volume,
p.avg_volume_12m,
x.first_date,
coalesce(d.delist_date, current_date) as last_date,
percent_rank() over (partition by v.date order by p.avg_volume_12m desc) as volume_12m_pct_rank,
from transformed.monthly_avg_valuation v
left join transformed.monthly_px_returns p on v.ticker = p.ticker and v.date = p.date
left join delist d on v.ticker = d.ticker
left join pricing x on v.ticker = x.ticker
order by v.date, mcap_pct_rank 
)
'''
query_job = client.query(
    query
)  
query_job.result()

### fundamentals
def clean_fundamentals(raw_data, dimension, write_table_id):

    client = bigquery.Client()
    query_job = client.query("""
    SELECT *
    FROM financials.{}_fundamentals_data_quality
    """.format(dimension))

    data_quality = query_job.result().to_dataframe()

    client = bigquery.Client()
    query_job = client.query("""
    SELECT *
    FROM financials.{}_fundamentals_data_quality_by_ticker
    """.format(dimension))

    data_quality_by_ticker = query_job.result().to_dataframe()

    all_data = raw_data.copy()
    all_data = all_data.drop(columns = ['dimension', 'calendardate', 'reportperiod', 'lastupdated'])
    all_data.loc[(all_data.ticker == 'AAPL') & (all_data.liabilities <= 0), 'liabilities'] = np.nan

    #### revenue, revenueusd
    idx = all_data.revenue < 0.0
    all_data.loc[idx, 'revenue'] = 10

    idx = all_data.revenueusd < 0.0
    all_data.loc[idx, 'revenue'] = 10

    ### <.0002 fields
    # zeros
    this_field = data_quality.loc[data_quality.pct_zero < 0.0002, 'indicator']

    metrics_list =  all_data.loc[:, this_field]
    X = metrics_list == 0.0
    Y = X.any(axis = 1)
    drop_tickers = all_data.ticker[Y].unique()

    idx = np.in1d(all_data.ticker, drop_tickers)
    all_data = all_data.loc[~idx, :]

    # neg
    this_field = data_quality.loc[data_quality.pct_neg < 0.0002, 'indicator']

    metrics_list =  all_data.loc[:, this_field]
    X = metrics_list < 0.0
    Y = X.any(axis = 1)

    drop_tickers = all_data.ticker[Y].unique()

    idx = np.in1d(all_data.ticker, drop_tickers)
    all_data = all_data.loc[~idx, :]

    # small pct is zeros
    this_field = data_quality_by_ticker.loc[(data_quality_by_ticker.pct_zero > 0.0) & (data_quality_by_ticker.pct_zero < 0.04), ['ticker', 'indicator']]
    this_field.columns = ['ticker', 'variable']
    this_field.loc[:, 'replace'] = 1

    long_data = pd.melt(all_data, id_vars = ['ticker', 'datekey'])
    long_data = long_data.merge(this_field, how = 'left', on = ['ticker', 'variable'])

    long_data.loc[long_data['replace'] == 1, 'value'] = long_data.loc[long_data['replace'] == 1, 'value'].replace(0, np.nan)
    long_data = long_data.drop(columns = 'replace')

    fnl_data = pd.pivot_table(long_data, index = ['ticker', 'datekey'], columns = 'variable', values = 'value')

    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(
        fnl_data, write_table_id, job_config=job_config
    )

dimension = 'ART'
query_job = client.query("""
        SELECT 
        f.* EXCEPT (ev, evebit, pb,pe,ps),                                               
        FROM financials.sharadar_fundamentals f
        where f.dimension = '{}'
        order by f.ticker, f.datekey              
        """.format(dimension))
raw_data = query_job.result().to_dataframe()

write_table_id = 'financials.ARQ_fundamentals_cleaned'
clean_fundamentals(raw_data, dimension, write_table_id)

def fundamentals_transform(credentials_path, transform_map_path,dimension, read_tbl, write_tbl):
 
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    ### gather ticker universe
    client = bigquery.Client()
    query_job = client.query("""
    SELECT distinct ticker
    FROM {}
    """.format(read_tbl))

    ticker_list = query_job.result().to_dataframe()

    sub_tickers = np.array_split(ticker_list.ticker, 20)

    # get data quality
    client = bigquery.Client()
    query_job = client.query("""
    SELECT *
    FROM financials.{}_fundamentals_data_quality
    """.format(dimension))

    data_quality = query_job.result().to_dataframe()

    ## make indicator transform map
    indicator_list = pd.read_csv(transform_map_path)

    ### clear old table
    query = ''' truncate table ''' + write_tbl
    query_job = client.query(
        query
    ) 

    #### run transformations 

    for j in range(0, 20):

        client = bigquery.Client()
        this_list = sub_tickers[j]
        table_id = 'boreal-pride-417020.transformed.temp_tickers'
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = client.load_table_from_dataframe(
                        pd.DataFrame(this_list), table_id, job_config=job_config
                    )
        job.result()

        query_job = client.query("""
        SELECT 
        f.*,                                               
        FROM {} f
        inner join transformed.temp_tickers t on f.ticker = t.ticker
        order by ticker, datekey              
        """.format(read_tbl))
        raw_data = query_job.result().to_dataframe()

        raw_data.loc[:, 'date'] = raw_data.datekey + MonthEnd(0)
        raw_data.loc[:, 'date'] = raw_data.loc[:, 'date'].dt.tz_localize(None)

        financials = raw_data.copy()
        indx = financials.ticker
        financials = financials.groupby('ticker', as_index = False).ffill()
        financials = pd.concat([indx, financials], axis = 1)

        financials = financials.sort_values(by=['ticker', 'datekey'], ascending = False).reset_index(drop=True)
        financials = financials.drop_duplicates(subset = ['ticker', 'date'])
        financials = financials.sort_values(by=['ticker', 'date'], ascending = True).reset_index(drop=True)

        ### Data Quality post fill NA
        ### 0.0002 < fields < 0.01 
        # negatives
        this_field = data_quality.loc[(data_quality.pct_neg > 0.0002) & (data_quality.pct_neg < 0.01), 'indicator']
        metrics_list =  financials.loc[:, this_field]
        filled = metrics_list.clip(lower=-0.1)
        filled = filled.replace(-0.1, np.nan)

        financials.loc[:, this_field] = filled

        # zeros
        this_field = data_quality.loc[(data_quality.pct_zero > 0.0002) & (data_quality.pct_zero < 0.01), 'indicator']
        metrics_list =  financials.loc[:, this_field]
        filled = metrics_list.replace(0, np.nan)

        financials.loc[:, this_field] = filled

        query_job = client.query("""
        SELECT
        v.ticker,
        v.date, 
        v.ev,
        v.evebit,
        v.pb,
        v.pe,
        v.ps                    
        from transformed.monthly_avg_valuation v
        inner join transformed.temp_tickers t on v.ticker = t.ticker 
        order by ticker, date          
        """)
        metrics = query_job.result().to_dataframe()

        metrics.loc[:, 'date'] = pd.to_datetime(metrics.loc[:, 'date'])

        all_data = financials.merge(metrics, how = 'right', on = ['ticker', 'date'])
        indx = all_data.ticker
        all_data = all_data.groupby('ticker', as_index = False).ffill()
        all_data = pd.concat([indx, all_data], axis = 1)

        # additional ratios
        all_data.loc[:, 'ebitmargin'] = round(all_data.loc[:, 'ebit'] / all_data.loc[:, 'revenue'], 3)
        all_data.loc[:, 'opmargin'] = round(all_data.loc[:, 'opinc'] / all_data.loc[:, 'revenue'], 3)
        all_data.loc[:, 'fcfmargin'] = round(all_data.loc[:, 'fcf'] / all_data.loc[:, 'revenue'], 3)

        all_data.loc[:, 'netdebt'] = all_data.loc[:, 'debt'] - all_data.loc[:, 'cashneq']
        all_data.loc[:, 'debt_ebit'] = round(all_data.loc[:, 'debt'] / all_data.loc[:, 'ebit'], 3)
        all_data.loc[:, 'debt_ebitda'] = round(all_data.loc[:, 'debt'] / all_data.loc[:, 'ebitda'], 3)
        all_data.loc[:, 'netdebt_ebit'] = round(all_data.loc[:, 'netdebt'] / all_data.loc[:, 'ebit'], 3)
        all_data.loc[:, 'netdebt_ebitda'] = round(all_data.loc[:, 'netdebt'] / all_data.loc[:, 'ebitda'], 3)
        all_data.loc[:, 'interest_ebit'] = round(all_data.loc[:, 'intexp'].fillna(0) / all_data.loc[:, 'ebit'], 3)

        all_data.loc[:, 'capex_asset'] = round(all_data.loc[:, 'capex'] / all_data.loc[:, 'assets'], 3)
        all_data.loc[:, 'equity_asset'] = round(all_data.loc[:, 'equity'] / all_data.loc[:, 'assets'], 3)
        all_data.loc[:, 'retearn_asset'] = round(all_data.loc[:, 'retearn'] / all_data.loc[:, 'assets'], 3)

        all_metrics = pd.DataFrame()
        for this_ticker in this_list:
            
            data = all_data.loc[all_data.ticker == this_ticker, :]               
            # transformations

            this_fields = indicator_list.loc[indicator_list.dyoy == 1, 'indicator'].tolist()
            original = data.loc[:, this_fields]
            lag = original.shift(12)
            transform = original - lag
            transform.columns = original.columns.values + '_dyoy'
            data = pd.concat([data, transform], axis = 1)

            this_fields = indicator_list.loc[indicator_list.yoy == 1, 'indicator'].tolist()
            original = data.loc[:, this_fields]
            lag = original.shift(12)
            transform = original / lag - 1
            transform.columns = original.columns.values + '_yoy'
            data = pd.concat([data, transform], axis = 1)

            this_fields = indicator_list.loc[indicator_list['3mom'] == 1, 'indicator'].tolist()
            original = data.loc[:, this_fields]
            lag = original.shift(3)
            transform = original / lag - 1
            transform.columns = original.columns.values + '_3mom'
            data = pd.concat([data, transform], axis = 1)

            this_fields = indicator_list.loc[indicator_list.d3mom == 1, 'indicator'].tolist()
            original = data.loc[:, this_fields]
            lag = original.shift(3)
            transform = original - lag
            transform.columns = original.columns.values + '_d3mom'
            data = pd.concat([data, transform], axis = 1)

            # other
            data.loc[:, 'peg'] = data.loc[:, 'pe'] / (data.loc[:, 'revenue_yoy']*100)
            data.loc[:, 'psg'] = data.loc[:, 'ps'] /(data.loc[:, 'revenue_yoy']*100)
            
            # append table
            all_metrics = pd.concat([all_metrics, data])

        # write to database
    
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = client.load_table_from_dataframe(
                        all_metrics, write_tbl, job_config=job_config
                    )
        print('written chunk: '+ str(j))

credentials_path = 'gcp-bigquery-privatekey.json'
transform_map_path = 'field_transform_map.csv'
read_tbl = 'financials.ART_fundamentals_cleaned'
write_tbl = 'transformed.monthly_financials_filled'
fundamentals_transform(credentials_path, transform_map_path, dimension, read_tbl, write_tbl)

