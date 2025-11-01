import os
import pandas as pd
from datetime import datetime
import numpy as np

from google.cloud import bigquery, bigquery_storage
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types

import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import plotly.express as px

# Set Page Defaults
st.set_page_config(layout="wide")

credentials_path = 'gcp-bigquery-privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
client = bigquery.Client()

def get_ptf_constituents(schema, model_id, vars, get_date):
    
    if len(vars) == 5:
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
            d.ps,
            d.{var1},
            d.{var2},
            d.{var3},
            d.{var4},
            d.{var5}

        from {schema}.backtest_ptfs_combo ptf
        left join prices.tickers_tbl t on ptf.ticker = t.ticker
        left join transformed.marketcap_volume_rank m on ptf.ticker = m.ticker and ptf.date = m.date
        left join training_data.training_data_top60pct_ipo_delay d on ptf.ticker = d.ticker and ptf.date = d.date
        where ptf.date = '{date}' and ptf.model_id = {model_id}
        order by ptf.date desc, ptf.ComboRank asc

        '''.format(schema = schema, date = get_date.strftime('%Y-%m-%d'),model_id=model_id, var1=vars[0], var2=vars[1], var3=vars[2],var4=vars[3],var5=vars[4])
    elif len(vars) == 3:
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
            d.ps,
            d.{var1},
            d.{var2},
            d.{var3}

        from {schema}.backtest_ptfs_combo ptf
        left join prices.tickers_tbl t on ptf.ticker = t.ticker
        left join transformed.marketcap_volume_rank m on ptf.ticker = m.ticker and ptf.date = m.date
        left join training_data.training_data_top60pct_ipo_delay d on ptf.ticker = d.ticker and ptf.date = d.date
        where ptf.date = '{date}' and ptf.model_id = {model_id}
        order by ptf.date desc, ptf.ComboRank asc

        '''.format(schema = schema, date = get_date.strftime('%Y-%m-%d'),model_id=model_id, var1=vars[0], var2=vars[1], var3=vars[2])
    else:
        print("error in retrieving ptf")
        return
    
    query_job = client.query(query)
    ptf_constituents = query_job.result().to_dataframe()

    return ptf_constituents

def get_columnDefs(vars):
    if len(vars) == 5:
        col_defs = [
            {
                "field": "company_name",
                "headerName": "Name",
                "minWidth": 120,
                "pinned": "left",
                "cellStyle": {"fontWeight": "bold"},
            },
            {
                "field": "marketcap",
                "headerName": "Mcap",
                "maxWidth": 80,
                "type": ["numericColumn", "customCurrencyFormat"],
                "custom_currency_symbol": "$",
                "precision":0,
                "cellStyle": {"fontWeight": "bold"},
            },
            {
                "field": "sector_industry",
                "headerName": "Industry",
                "maxWidth": 250,
            },
            {"field": "ComboRank", "headerName": "Rank", "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"], "precision": 0},
            {"field": "ps", "headerName": "P/S", "maxWidth": 80, "type": ["numericColumn", "customNumericFormat"], "precision": 2},
            {"field": "px_pct_3m", "headerName": "PX %3M", "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"], "precision": 0},
            {"field": "px_pct_6m", "headerName": "PX %6M", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "px_yoy", "headerName": "PX %YoY", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "px_vol_3m", "headerName": "PX VOL 3M", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "volume_yoy", "headerName": "VOL %YoY", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "revenue_yoy", "headerName": "Revenue %YoY", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "opmargin", "headerName": "OP Margin", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": vars[0], "headerName":vars[0].replace("_", " "), "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"],"precision": 2},
            {"field": vars[1], "headerName":vars[1].replace("_", " "),  "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"],"precision": 2},
            {"field": vars[2], "headerName":vars[2].replace("_", " "),  "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"],"precision": 2},
            {"field": vars[3], "headerName":vars[3].replace("_", " "),  "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"],"precision": 2},
            {"field": vars[4], "headerName":vars[4].replace("_", " "),  "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"],"precision": 2},
            {"field": "ticker", "headerName": "Ticker", "maxWidth": data_maxWidth}
        ]
    elif len(vars) == 3:
        col_defs = [
            {
                "field": "company_name",
                "headerName": "Name",
                "minWidth": 120,
                "pinned": "left",
                "cellStyle": {"fontWeight": "bold"},
            },
            {
                "field": "marketcap",
                "headerName": "Mcap",
                "maxWidth": 80,
                "type": ["numericColumn", "customCurrencyFormat"],
                "custom_currency_symbol": "$",
                "precision":0,
                "cellStyle": {"fontWeight": "bold"},
            },
            {
                "field": "sector_industry",
                "headerName": "Industry",
                "maxWidth": 250,
            },
            {"field": "ComboRank", "headerName": "Rank", "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"], "precision": 0},
            {"field": "ps", "headerName": "P/S", "maxWidth": 80, "type": ["numericColumn", "customNumericFormat"], "precision": 2},
            {"field": "px_pct_3m", "headerName": "PX %3M", "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"], "precision": 0},
            {"field": "px_pct_6m", "headerName": "PX %6M", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "px_yoy", "headerName": "PX %YoY", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "px_vol_3m", "headerName": "PX VOL 3M", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "volume_yoy", "headerName": "VOL %YoY", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "revenue_yoy", "headerName": "Revenue %YoY", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": "opmargin", "headerName": "OP Margin", "maxWidth": data_maxWidth,  "type": ["numericColumn", "customNumericFormat"],"precision": 0},
            {"field": vars[0], "headerName":vars[0].replace("_", " "), "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"],"precision": 2},
            {"field": vars[1], "headerName":vars[1].replace("_", " "),  "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"],"precision": 2},
            {"field": vars[2], "headerName":vars[2].replace("_", " "),  "maxWidth": data_maxWidth, "type": ["numericColumn", "customNumericFormat"],"precision": 2},
            {"field": "ticker", "headerName": "Ticker", "maxWidth": data_maxWidth}
        ]
    else:
        print("error returning columnDefs")
        return
    
    return col_defs

### data download

latest_date = datetime.today() + pd.offsets.MonthEnd(-1)
schema = 'prod'

query = '''
select
    model_id,
    description, 
    date,
    next_month_return,
    turnover,
    trade_costs,
    total_return,
    train_dataset,
    run_date
from {schema}.backtest_ptfs_combo_monthly_return
order by model_id, date

'''.format(schema = schema)

query_job = client.query(query)
monthly_return = query_job.result().to_dataframe()

query = 'select * from prod.llama_stock_summary'
query_job = client.query(query)
stock_summary = query_job.result().to_dataframe()

### dashboard

## Time selector
def ChngDt(wch_option):
    latest_date = datetime.now() + pd.offsets.MonthEnd(-1)

    if wch_option == "3m":
        st.session_state.ksd = latest_date + pd.offsets.MonthEnd(-3)
    elif wch_option == "6m":
        st.session_state.ksd = latest_date + pd.offsets.MonthEnd(-6)
    elif wch_option == "YTD":
        st.session_state.ksd = latest_date + pd.offsets.YearEnd(-1)
    elif wch_option == "1Y":
        st.session_state.ksd = latest_date + pd.offsets.MonthEnd(-12)
    elif wch_option == "2Y":
        st.session_state.ksd = latest_date + pd.offsets.MonthEnd(-24)
    elif wch_option == "3Y":
        st.session_state.ksd = latest_date + pd.offsets.MonthEnd(-36)
    elif wch_option == "5Y":
        st.session_state.ksd = latest_date + pd.offsets.MonthEnd(-60)

# set date defaults
latest_date = datetime.now() + pd.offsets.MonthEnd(-1)

if 'ksd' not in st.session_state:
    st.session_state.ksd  = latest_date + pd.offsets.YearEnd(-1)

sc11, sc12 = st.columns(2)
stdt = sc11.date_input("Start Date:", key="ksd")
nddt = sc12.date_input("End Date:",  value = latest_date)

button1, button2, button3, button4, button5, button6, button7 = st.columns(7)
if button1.button("3m", on_click=ChngDt, args=("3m",)):
    wch_option = "3m:"

if button2.button("6m", on_click=ChngDt, args=("6m",)):
    wch_option = "6m:"

if button3.button("YTD", on_click=ChngDt, args=("YTD",)):
    wch_option = "YTD:"

if button4.button("1Y", on_click=ChngDt, args=("1Y",)):
    wch_option = "1Y:"

if button5.button("2Y", on_click=ChngDt, args=("2Y",)):
    wch_option = "2Y:"

if button6.button("3Y", on_click=ChngDt, args=("3Y",)):
    wch_option = "3Y:"

if button7.button("5Y", on_click=ChngDt, args=("5Y",)):
    wch_option = "5Y:"

## Table 1

date_start = stdt + pd.offsets.MonthEnd(0)
date_end = nddt + pd.offsets.MonthEnd(0)

graph_data = monthly_return.copy()
graph_data.loc[:, 'date'] = graph_data.loc[:, 'date'] + pd.offsets.MonthEnd(1)
graph_data = graph_data.loc[(graph_data.date >= date_start) & (graph_data.date <= date_end), :].reset_index(drop = True)

graph_data.loc[:, 'multiplier'] = graph_data.total_return + 1
X = graph_data.groupby('model_id')['multiplier'].apply(lambda x: x.cumprod()).reset_index(drop = True)
X.name = 'return_index'
graph_data = pd.concat([graph_data, X], axis = 1)

graph_data.loc[:, 'return_index'] = graph_data.loc[:, 'return_index'] * 100

table1 = graph_data.loc[:, ['model_id', 'date', 'return_index', 'description']]

legend_order = graph_data.loc[graph_data.date == date_end, ['model_id', 'return_index']].reset_index(drop = True)
legend_order = legend_order.sort_values(by=['return_index'], ascending = False)

top = legend_order.iloc[0:25, ]

idx = np.in1d(table1.model_id, top.model_id)
table1 = table1.loc[idx, :]

fig = px.line(table1, x="date", y="return_index", color="model_id", 
              labels={'y': 'Total Return', 'x': 'Date'},
              hover_data = ['model_id', 'description', 'return_index'],
              category_orders = {'model_id': legend_order.model_id})

fig.update_layout(title='', xaxis_title='Date', yaxis_title='Total Return')

header_text = 'Portfolio Returns: ' + date_start.strftime('%Y-%m-%d') + ' - ' + date_end.strftime('%Y-%m-%d') 

st.header(header_text)
st.plotly_chart(fig, use_container_width=True)

## Table 2
st.header('Portfolio Constituents')

# Controls
max_model = monthly_return.model_id.max()
sc21, sc22 = st.columns(2)

this_model = sc21.number_input("Model ID", value=12, min_value = 1, max_value = max_model)
ptf_date = sc22.date_input("Ptf As Of Date:", value =latest_date)

# Text
factors = monthly_return.loc[monthly_return.model_id == this_model, "description"].unique()[0]
st.text("Model Factors: " + factors)

vars = factors.split(',')

# Get Ptf Constituents

ptf_constituents = get_ptf_constituents(schema, this_model,vars, ptf_date)

# Table
ptf_constituents.loc[:, 'sector_industry'] = ptf_constituents.sector + ': ' + ptf_constituents.industry
columns = ['company_name', 'marketcap', 'sector_industry', 'ComboRank', 'ps', 'px_pct_3m', 'px_pct_6m', 'px_yoy', 'px_vol_3m', 'volume_yoy', 'revenue_yoy', 'opmargin','ticker'] + vars
columns = list(set(columns))

table2 = ptf_constituents.loc[:, columns]

pct_columns = ['px_pct_3m', 'px_pct_6m', 'px_yoy', 'px_vol_3m', 'volume_yoy', 'revenue_yoy', 'opmargin']
table2.loc[:, pct_columns] = table2.loc[:, pct_columns] * 100
table2.loc[:, "marketcap"] = round(table2.loc[:, "marketcap"], 0)

gb = GridOptionsBuilder.from_dataframe(table2)
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
gb.configure_selection(selection_mode = "single", pre_selected_rows = ["0"])
gb.configure_default_column(
        floating_filter=True,
        headerComponentParams={
            "template":
                '<div class="ag-cell-label-container" role="presentation">' +
                '  <span ref="eMenu" class="ag-header-icon ag-header-cell-menu-button"></span>' +
                '  <div ref="eLabel" class="ag-header-cell-label" role="presentation">' +
                '    <span ref="eSortOrder" class="ag-header-icon ag-sort-order"></span>' +
                '    <span ref="eSortAsc" class="ag-header-icon ag-sort-ascending-icon"></span>' +
                '    <span ref="eSortDesc" class="ag-header-icon ag-sort-descending-icon"></span>' +
                '    <span ref="eSortNone" class="ag-header-icon ag-sort-none-icon"></span>' +
                '    <span ref="eText" class="ag-header-cell-text" role="columnheader" style="white-space: normal;text-align: right;"></span>' +
                '    <span ref="eFilter" class="ag-header-icon ag-filter-icon"></span>' +
                '  </div>' +
                '</div>'
        }   
    )
gridOptions = gb.build()
### custom columns
data_maxWidth = 70
gridOptions['headerHeight'] = 80


gridOptions["columnDefs"] = get_columnDefs(vars)

grid_return = AgGrid(table2, 
                     gridOptions=gridOptions,
                     height=650,
                     width = '100%',
                     update_mode = GridUpdateMode.SELECTION_CHANGED)

selected = grid_return.selected_rows
selected_ticker = selected['ticker'][0]

## Table 4

st.header('Sector Breakdown')

pie_start_date = latest_date + pd.offsets.MonthEnd(-12)
pie_end_date = latest_date

query = '''
select 
    t.sector,
    ptf.date,
    count(*)/50 * 100 as percentage

from {schema}.backtest_ptfs_combo ptf
left join prices.tickers_tbl t on ptf.ticker = t.ticker
where ptf.date >= '{start_date}' and ptf.date <= '{end_date}' and ptf.model_id = {model_id}
group by t.sector, ptf.date
order by ptf.date, percentage desc
'''.format(schema = schema, start_date = pie_start_date.strftime("%Y-%m-%d"), end_date = pie_end_date.strftime("%Y-%m-%d"), model_id = this_model)

query_job = client.query(query)
pie_chart = query_job.result().to_dataframe()


fig2 = px.bar(pie_chart, x="date", y="percentage", color="sector", 
              labels={'y': 'Pct', 'x': 'Date'},
              hover_data = ['date', 'sector', 'percentage'])

fig2.update_layout(title='', xaxis_title='Date', yaxis_title='Percent of Portfolio')
st.plotly_chart(fig2, use_container_width=True)

## Table 3
with st.sidebar: 
    st.header('Stock Summary')

    table3 = stock_summary.loc[stock_summary.Ticker == selected_ticker, :].reset_index(drop=True)

    text1 = table3['Overview']
    text2 = table3['News']
    text3 = table3['Competitors']
    text4 = table3['run_date']

    st.subheader("Overview")
    st.markdown(table3.loc[0, 'Overview'])

    st.subheader("News")
    st.markdown(table3.loc[0, 'News'])

    st.subheader("Competitors")
    st.markdown(table3.loc[0, 'Competitors'])
    st.caption("Data as of " + table3.loc[0, 'run_date'].strftime("%Y-%m-%d"))

