import requests
import json
import pandas as pd
import os 
from google.cloud import bigquery
import numpy as np
from datetime import datetime, timedelta

from ib_insync import *

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()

snapshot_table_id = 'boreal-pride-417020.monitor.ibkr_ptf_snapshot'

def get_latest_px(ticker_list, reqType = 3):
    ib.reqMarketDataType(reqType)
    ### first pass
    px_tbl = list()
    for this_symbol in ticker_list:
        this_contract = Stock(symbol = this_symbol, exchange = 'SMART', currency='USD')
        ib.qualifyContracts(this_contract)
        Ticker=ib.reqMktData(this_contract, '', True, False)
        ib.sleep(3)  
        this_px = [this_symbol, Ticker.last]
        px_tbl.append(this_px)

    px_tbl = pd.DataFrame(px_tbl, columns = ['ticker', 'px'])

    return px_tbl

def get_latest_px_retries(ticker_list):
    
    px_tbl = get_latest_px(ticker_list)
    ### retry for nulls
    retry1 = px_tbl.loc[px_tbl.px.isnull(), 'ticker']
    if len(retry1) > 0: 

        retry1_list = get_latest_px(retry1)
        px_tbl = pd.concat([px_tbl.loc[~px_tbl.px.isnull(), :], retry1_list])

    ### 2nd retry for nulls
    retry2 = px_tbl.loc[px_tbl.px.isnull(), 'ticker']
    if len(retry2) > 0:
        retry2_list = get_latest_px(retry2, reqType = 2)
        px_tbl = pd.concat([px_tbl.loc[~px_tbl.px.isnull(), :], retry2_list])

    return px_tbl

def create_ptf_snapshot(account_id, snap_type):
    snapshot = ib.positions(account = account_id)
    ticker = []
    for pos in snapshot:
        this_contract = pos.contract.symbol
        ticker.append(this_contract)
    snapshot = pd.concat([pd.DataFrame(snapshot), pd.DataFrame(ticker, columns = ['ticker'])], axis = 1)

    snapshot = snapshot.drop(columns = ['contract'])
    px_tbl = get_latest_px_retries(snapshot.ticker)

    snapshot = snapshot.merge(px_tbl, how = 'left', on='ticker')
    snapshot.loc[:, 'snapshot_type'] = snap_type
    snapshot.loc[:, 'as_of_date'] = datetime.now() 
    return snapshot

# connect to TWS
util.startLoop()
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=2)

# get existing positions
#account_id = 'U14466554' #brokerage
#account_id = 'U14784906' #roth ira
account_id = 'U14784949' #traditional ira


latest_date = datetime.today() + pd.offsets.MonthEnd(-1)

query_job = client.query("""
SELECT 
model_id
FROM boreal-pride-417020.prod.meta_combo_ptfs p
where date = '{}'
""".format(latest_date.strftime('%Y-%m-%d')))

model_id = query_job.result().to_dataframe().model_id

query_job = client.query("""
SELECT 
p.*
FROM prod.backtest_ptfs_combo p
where p.model_id = {} and date = '{}'
""".format(model_id[0], latest_date.strftime('%Y-%m-%d')))

stock_list = query_job.result().to_dataframe()

query_job = client.query("""
select
ticker,
close as px
from prices.px  
where date = (select max(date) from prices.px)
""".format(latest_date.strftime('%Y-%m-%d')))

px_list = query_job.result().to_dataframe()

new_ptf = stock_list.merge(px_list, how = 'left', on = 'ticker')

# get price
trade_list = new_ptf.ticker

px_tbl = get_latest_px_retries(trade_list)
px_df = pd.DataFrame(px_tbl, columns = ['ticker', 'px'])

new_ptf = new_ptf.drop(columns = ['px'])
new_ptf = new_ptf.merge(px_df, how = 'left', on = 'ticker')

new_ptf.loc[:, 'px'] = new_ptf.px.fillna(10000)

## calculate new position
# get cash
acct_tbl = ib.accountValues(account = account_id)
acct_tbl = pd.DataFrame(acct_tbl)

cash = acct_tbl.loc[acct_tbl.tag == 'NetLiquidation', 'value'].values[0]
cash = float(cash)

# create orders
size = cash/50
new_ptf.loc[:, 'target'] = round(size, 0)
new_ptf.loc[:, 'quantity'] = round(new_ptf.target / new_ptf.px, 0)
new_ptf.loc[:, 'quantity_frac'] = round(new_ptf.target / new_ptf.px, 2)

## get existing positions
pos_tbl = create_ptf_snapshot(account_id, 'end')

# write snapshot db

pos_tbl = pos_tbl.rename(columns = {'old_ticker':'ticker'})

job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
job = client.load_table_from_dataframe(
                pos_tbl, snapshot_table_id, job_config=job_config
            )
job.result()

# create master table
all_tbl = pos_tbl.merge(new_ptf, how = 'outer', on = 'ticker')

all_tbl.loc[:, 'position'] = all_tbl.position.fillna(0)
all_tbl.loc[:, 'quantity'] = all_tbl.quantity.fillna(0)
all_tbl.loc[:, 'quantity_frac'] = all_tbl.quantity_frac.fillna(0)

all_tbl.loc[:, 'trade_quantity'] = all_tbl.quantity - all_tbl.position
all_tbl.loc[:, 'px'] = all_tbl.px_x.combine_first(all_tbl.px_y)

order_tbl = all_tbl.loc[(all_tbl.trade_quantity != 0) & (all_tbl.px != 10000) , ['ticker', 'trade_quantity']]

manual = new_ptf.loc[new_ptf.quantity == 0, ['ticker', 'quantity_frac']]
sell_tbl = order_tbl.loc[order_tbl.trade_quantity < 0, :].reset_index(drop=True)
buy_tbl = order_tbl.loc[order_tbl.trade_quantity > 0, :].reset_index(drop=True)

# sell positions
for i in range(0, len(sell_tbl)):
    this_contract = Stock(symbol = sell_tbl.loc[i, 'ticker'], exchange = 'SMART', currency='USD')
    order = MarketOrder('SELL', sell_tbl.loc[i, 'trade_quantity']*-1)
    trade = ib.placeOrder(this_contract, order)

# buy positions
for i in range(0, len(buy_tbl)):
    this_contract = Stock(symbol = buy_tbl.loc[i, 'ticker'], exchange = 'SMART', currency='USD')
    order = MarketOrder('BUY', buy_tbl.loc[i, 'trade_quantity'])
    trade = ib.placeOrder(this_contract, order)

### Check trades
new_pos = ib.positions(account = account_id)

old_ticker = []
for pos in new_pos:
    this_contract = pos.contract.symbol
    old_ticker.append(this_contract)

new_pos = pd.concat([pd.DataFrame(new_pos), pd.DataFrame(old_ticker, columns = ['old_ticker'])], axis = 1)

# create master table
new_pos = new_pos.merge(new_ptf, how = 'outer', left_on = 'old_ticker', right_on = 'ticker')

clean = new_pos.loc[new_pos.ticker.isnull(), :]

# view open orders
#orders = ib.orders()

# cancel all open orders
ib.reqGlobalCancel()

# current ptf snapshot

snapshot = create_ptf_snapshot(account_id, 'start')

# write to db
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",
)
job = client.load_table_from_dataframe(
                snapshot, snapshot_table_id, job_config=job_config
            )
job.result()

ib.disconnect()

