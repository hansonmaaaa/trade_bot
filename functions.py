import pandas as pd
import os 
from google.cloud import bigquery, bigquery_storage
import numpy as np
import math 
from datetime import datetime
import itertools

import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import shap 

from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types
from google.cloud import bigquery_storage


def print_runtime(time_start):
    time_stop = datetime.now()
    runtime = time_stop - time_start
    print(runtime)
    
def read_table(bqstorageclient, project_id, dataset_id, table_id):

    table = f"projects/{project_id}/datasets/{dataset_id}/tables/{table_id}"

    read_options = types.ReadSession.TableReadOptions(
        selected_fields=["country_name", "region_name"] 
    )

    parent = "projects/{}".format(project_id)

    requested_session = types.ReadSession(
        table=table,
        data_format=types.DataFormat.ARROW,
        #read_options=read_options,
    )
    read_session = bqstorageclient.create_read_session(
        parent=parent,
        read_session=requested_session,
        max_stream_count=1,
    )

    stream = read_session.streams[0] #read every stream from 0 to 3
    reader = bqstorageclient.read_rows(stream.name)

    frames = []
    for message in reader.rows().pages:
        frames.append(message.to_dataframe())
    dataframe = pd.concat(frames)
    #print(dataframe.head())

    return dataframe

def get_model_tbl_from_id(model_result_table_id, starting_model_id):
    client = bigquery.Client()
    query_job = client.query("""
    SELECT *
    FROM {id1} where model_id = {id2}
    """.format(id1=model_result_table_id, id2=starting_model_id))

    model_tbl = query_job.result().to_dataframe()
    Y_field = model_tbl.loc[0, 'dependent_var']

    params = model_tbl.loc[model_tbl.model_id == starting_model_id, ['parameter', 'value']]
    params['value'] = params['value'].astype(float)
    xgb_params = dict(zip(params.parameter, params.value))
    xgb_params['n_estimators'] = int(xgb_params['n_estimators'])
    xgb_params['max_depth'] = int(xgb_params['max_depth'])

    return Y_field, xgb_params

### backtests
def backtest_classifier_base_removeNAs(all_data, Y_field, incl, xgb_params, cutoff): 
    
    #remove rows with > 50% NAs
    threshold = round(.5 * len(incl), 0)
    row_nas = all_data.apply(lambda x: x.isna().sum(), axis=1)
    row_nas = pd.DataFrame(row_nas, columns = ['nan_count']) 
    row_nas.loc[:, 'keep'] = True
    row_nas.loc[row_nas.nan_count > threshold, 'keep'] = False

    all_data = all_data.loc[row_nas.keep.tolist(), :]    
    
    date_list = all_data.loc[:, 'date'].sort_values().unique()
    train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
    train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)

    all_portfolio = pd.DataFrame()
    start = cutoff
    for k in range(start,len(date_list)):
        try:
            latest_month = date_list[k]

            latest_data = all_data.loc[all_data.date == latest_month, :].reset_index(drop = True)
            
            X_train = train.loc[:, incl]
            Y_train = train.loc[:, Y_field]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, Y_train)

            # make predictions
            
            pred_class = pd.DataFrame(model.predict(latest_data.loc[:, incl]), columns = ['pred_label']) 
            score = f1_score(latest_data.loc[:, Y_field], pred_class)
            #print(str(latest_month), ': ', str(score))

            pred = pd.DataFrame(model.predict_proba(latest_data.loc[:, incl])[:, 1], columns = ['pred']) 
            to_add = latest_data.loc[:, ['ticker', 'next_month_return']]
            pred = pd.concat([pred, to_add], axis = 1)

            to_add = latest_data.loc[:, Y_field]
            to_add.name = 'dependent_var'
            pred = pd.concat([pred, to_add], axis = 1)

            top_predict = pred.sort_values(by = 'pred', ascending = False).reset_index(drop=True)

            ## top 50, full replacement
            top_cutoff = 50
            portfolio = top_predict.iloc[0:top_cutoff, :]

            portfolio.loc[:, 'date'] = latest_month
            portfolio.loc[:, 'score'] = score 

            all_portfolio = pd.concat([all_portfolio, portfolio])
            
            # add latest month to train set
            train = pd.concat([train, latest_data])

        except Exception as e:
            print("error on date ", str(latest_month))
            print(e)

    return all_portfolio

def run_backtest_pred_threshold_removeNAs(all_data, Y_field, incl, xgb_params, pred_threshold, cutoff):
    #remove rows with > 50% NAs
    threshold = round(.5 * len(incl), 0)
    row_nas = all_data.apply(lambda x: x.isna().sum(), axis=1)
    row_nas = pd.DataFrame(row_nas, columns = ['nan_count']) 
    row_nas.loc[:, 'keep'] = True
    row_nas.loc[row_nas.nan_count > threshold, 'keep'] = False

    all_data = all_data.loc[row_nas.keep.tolist(), :]  
    
    date_list = all_data.loc[:, 'date'].sort_values().unique() 
    train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
    train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)

    all_portfolio = pd.DataFrame()

    start = cutoff
    for k in range(start,len(date_list)):
        try:
            prev_month = date_list[k-1]
            latest_month = date_list[k]

            latest_data = all_data.loc[all_data.date == latest_month, :].reset_index(drop = True)

            X_train = train.loc[:, incl]
            Y_train = train.loc[:, Y_field]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, Y_train)

            # make predictions
            
            pred_class = pd.DataFrame(model.predict(latest_data.loc[:, incl]), columns = ['pred_label']) 
            score = f1_score(latest_data.loc[:, Y_field], pred_class)
            print(str(latest_month), ': ', str(score))

            pred = pd.DataFrame(model.predict_proba(latest_data.loc[:, incl])[:, 1], columns = ['pred']) 
            to_add = latest_data.loc[:, ['ticker', 'next_month_return']]
            pred = pd.concat([pred, to_add], axis = 1)

            to_add = latest_data.loc[:, Y_field]
            to_add.name = 'dependent_var'
            pred = pd.concat([pred, to_add], axis = 1)

            top_predict = pred.sort_values(by = 'pred', ascending = False).reset_index(drop=True)

            ## top 50, full replacement
            top_cutoff = 50
            if k == start:
                portfolio = top_predict.iloc[0:top_cutoff, :]

            else:
            # top 50, replace according to rules
                prev_portfolio = all_portfolio.loc[all_portfolio.date == prev_month, ['ticker', 'pred']]
                prev_portfolio['is_prev'] = 1
                threshold = min(pred_threshold, top_predict.pred[50])
                new_ptf = top_predict.loc[top_predict.pred >= threshold, :]
                match_tbl = new_ptf.merge(prev_portfolio, how = 'left', on = 'ticker', suffixes=('', '_prev'))
                match_tbl['is_prev'] = match_tbl['is_prev'].fillna(0)
                match_tbl = match_tbl.sort_values(by = ['is_prev', 'pred'], ascending = False).reset_index(drop = True)

                portfolio = match_tbl.loc[0:top_cutoff, top_predict.columns]

            portfolio.loc[:, 'date'] = latest_month
            portfolio.loc[:, 'score'] = score 

            all_portfolio = pd.concat([all_portfolio, portfolio])

            # add latest month to train set
            train = pd.concat([train, latest_data])

        except Exception as e:
            print("error on date ", str(latest_month))
            print(e)

    return all_portfolio

def backtest_classifier_base(all_data, Y_field, incl, xgb_params, cutoff): 
    date_list = all_data.loc[:, 'date'].sort_values().unique()
    train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
    train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)

    all_portfolio = pd.DataFrame()
    start = cutoff
    for k in range(start,len(date_list)):
        try:
            latest_month = date_list[k]
            latest_data = all_data.loc[all_data.date == latest_month, :].reset_index(drop = True)
            
            X_train = train.loc[:, incl]
            Y_train = train.loc[:, Y_field]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, Y_train)

            # make predictions
            
            pred_class = pd.DataFrame(model.predict(latest_data.loc[:, incl]), columns = ['pred_label']) 
            score = f1_score(latest_data.loc[:, Y_field], pred_class)
            print(str(latest_month), ': ', str(score))

            pred = pd.DataFrame(model.predict_proba(latest_data.loc[:, incl])[:, 1], columns = ['pred']) 
            to_add = latest_data.loc[:, ['ticker', 'next_month_return']]
            pred = pd.concat([pred, to_add], axis = 1)

            to_add = latest_data.loc[:, Y_field]
            to_add.name = 'dependent_var'
            pred = pd.concat([pred, to_add], axis = 1)

            top_predict = pred.sort_values(by = 'pred', ascending = False).reset_index(drop=True)

            ## top 50, full replacement
            top_cutoff = 50
            portfolio = top_predict.iloc[0:top_cutoff, :]

            portfolio.loc[:, 'date'] = latest_month
            portfolio.loc[:, 'score'] = score 

            all_portfolio = pd.concat([all_portfolio, portfolio])

            #add latest to training set
            train = pd.concat([train, latest_data])

        except Exception as e:
            print("error on date ", str(latest_month))
            print(e)

    return all_portfolio

def backtest_classifier_base_short(all_data, Y_field, incl, xgb_params, cutoff): 
    date_list = all_data.loc[:, 'date'].sort_values().unique()
    train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
    train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)

    all_portfolio = pd.DataFrame()
    start = cutoff
    for k in range(start,len(date_list)):
        try:
            latest_month = date_list[k]
            latest_data = all_data.loc[all_data.date == latest_month, :].reset_index(drop = True)
            
            X_train = train.loc[:, incl]
            Y_train = train.loc[:, Y_field]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, Y_train)

            # make predictions
            
            #pred_class = pd.DataFrame(model.predict(latest_data.loc[:, incl]), columns = ['pred_label']) 
            #score = f1_score(latest_data.loc[:, Y_field], pred_class)
            #print(str(latest_month), ': ', str(score))

            pred = pd.DataFrame(model.predict_proba(latest_data.loc[:, incl])[:, 1], columns = ['pred']) 
            to_add = latest_data.loc[:, ['ticker', 'next_month_return']]
            pred = pd.concat([pred, to_add], axis = 1)

            #to_add = latest_data.loc[:, Y_field]
            #to_add.name = 'dependent_var'
            #pred = pd.concat([pred, to_add], axis = 1)

            top_predict = pred.sort_values(by = 'pred', ascending = False).reset_index(drop=True)

            ## top 50, full replacement
            top_cutoff = 50
            portfolio = top_predict.iloc[0:top_cutoff, :]

            portfolio.loc[:, 'date'] = latest_month
            #portfolio.loc[:, 'score'] = score 

            all_portfolio = pd.concat([all_portfolio, portfolio])

            #add latest to training set
            train = pd.concat([train, latest_data])

        except Exception as e:
            print("error on date ", str(latest_month))
            print(e)

    return all_portfolio

def run_backtest_replace100(all_data, Y_field, incl, xgb_params, cutoff):
    date_list = all_data.loc[:, 'date'].sort_values().unique()
    train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
    train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)

    all_portfolio = pd.DataFrame()
    start = cutoff
    for k in range(start,len(date_list)):
        prev_month = date_list[k-1]
        latest_month = date_list[k]

        latest_data = all_data.loc[all_data.date == latest_month, :].reset_index(drop = True)

        X_train = train.loc[:, incl]
        Y_train = train.loc[:, Y_field]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, Y_train)

        # make predictions
        
        pred_class = pd.DataFrame(model.predict(latest_data.loc[:, incl]), columns = ['pred_label']) 
        score = f1_score(latest_data.loc[:, Y_field], pred_class)
        print(str(latest_month), ': ', str(score))

        pred = pd.DataFrame(model.predict_proba(latest_data.loc[:, incl])[:, 1], columns = ['pred']) 
        to_add = latest_data.loc[:, ['ticker', 'next_month_return']]
        pred = pd.concat([pred, to_add], axis = 1)

        to_add = latest_data.loc[:, Y_field]
        to_add.name = 'dependent_var'
        pred = pd.concat([pred, to_add], axis = 1)

        top_predict = pred.sort_values(by = 'pred', ascending = False).reset_index(drop=True)

        ## top 50, full replacement
        top_cutoff = 50
        if k == start:
            portfolio = top_predict.iloc[0:top_cutoff, :]

        else:
        # top 50, replace according to rules
            prev_portfolio = all_portfolio.loc[all_portfolio.date == prev_month, ['ticker', 'pred']]
            prev_portfolio['is_prev'] = 1
            new_ptf = top_predict.loc[0:100, :]
            match_tbl = new_ptf.merge(prev_portfolio, how = 'left', on = 'ticker', suffixes=('', '_prev'))
            match_tbl['is_prev'] = match_tbl['is_prev'].fillna(0)
            match_tbl = match_tbl.sort_values(by = ['is_prev', 'pred'], ascending = False).reset_index(drop = True)

            portfolio = match_tbl.loc[0:top_cutoff, top_predict.columns]

        portfolio.loc[:, 'date'] = latest_month
        portfolio.loc[:, 'score'] = score 

        all_portfolio = pd.concat([all_portfolio, portfolio])

        # add latest month to train set
        train = pd.concat([train, latest_data])

    return all_portfolio

def run_backtest_pred_threshold(all_data, Y_field, incl, xgb_params, pred_threshold, cutoff):
    date_list = all_data.loc[:, 'date'].sort_values().unique() 
    train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
    train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)

    all_portfolio = pd.DataFrame()

    start = cutoff
    for k in range(start,len(date_list)):
        try:
            prev_month = date_list[k-1]
            latest_month = date_list[k]

            latest_data = all_data.loc[all_data.date == latest_month, :].reset_index(drop = True)

            X_train = train.loc[:, incl]
            Y_train = train.loc[:, Y_field]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, Y_train)

            # make predictions
            
            pred_class = pd.DataFrame(model.predict(latest_data.loc[:, incl]), columns = ['pred_label']) 
            score = f1_score(latest_data.loc[:, Y_field], pred_class)
            print(str(latest_month), ': ', str(score))

            pred = pd.DataFrame(model.predict_proba(latest_data.loc[:, incl])[:, 1], columns = ['pred']) 
            to_add = latest_data.loc[:, ['ticker', 'next_month_return']]
            pred = pd.concat([pred, to_add], axis = 1)

            to_add = latest_data.loc[:, Y_field]
            to_add.name = 'dependent_var'
            pred = pd.concat([pred, to_add], axis = 1)

            top_predict = pred.sort_values(by = 'pred', ascending = False).reset_index(drop=True)

            ## top 50, full replacement
            top_cutoff = 50
            if k == start:
                portfolio = top_predict.iloc[0:top_cutoff, :]

            else:
            # top 50, replace according to rules
                prev_portfolio = all_portfolio.loc[all_portfolio.date == prev_month, ['ticker', 'pred']]
                prev_portfolio['is_prev'] = 1
                threshold = min(pred_threshold, top_predict.pred[50])
                new_ptf = top_predict.loc[top_predict.pred >= threshold, :]
                match_tbl = new_ptf.merge(prev_portfolio, how = 'left', on = 'ticker', suffixes=('', '_prev'))
                match_tbl['is_prev'] = match_tbl['is_prev'].fillna(0)
                match_tbl = match_tbl.sort_values(by = ['is_prev', 'pred'], ascending = False).reset_index(drop = True)

                portfolio = match_tbl.loc[0:top_cutoff, top_predict.columns]

            portfolio.loc[:, 'date'] = latest_month
            portfolio.loc[:, 'score'] = score 

            all_portfolio = pd.concat([all_portfolio, portfolio])

            # add latest month to train set
            train = pd.concat([train, latest_data])

        except Exception as e:
            print("error on date ", str(latest_month))
            print(e)

    return all_portfolio

def backtest_classifier_top_mcap(all_data, Y_field, incl, xgb_params, cutoff, mcap_tbl): 
    date_list = all_data.loc[:, 'date'].sort_values().unique()
    train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
    train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)

    all_portfolio = pd.DataFrame()
    start = cutoff
    for k in range(start,len(date_list)):
        try:
            prev_month = date_list[k-1]
            latest_month = date_list[k]

            latest_data = all_data.loc[all_data.date == latest_month, :].reset_index(drop = True)
            
            X_train = train.loc[:, incl]
            Y_train = train.loc[:, Y_field]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, Y_train)

            # make predictions
            
            pred_class = pd.DataFrame(model.predict(latest_data.loc[:, incl]), columns = ['pred_label']) 
            score = f1_score(latest_data.loc[:, Y_field], pred_class)
            #print(str(latest_month), ': ', str(score))

            pred = pd.DataFrame(model.predict_proba(latest_data.loc[:, incl])[:, 1], columns = ['pred']) 
            to_add = latest_data.loc[:, ['ticker', 'next_month_return']]
            pred = pd.concat([pred, to_add], axis = 1)

            to_add = latest_data.loc[:, Y_field]
            to_add.name = 'dependent_var'
            pred = pd.concat([pred, to_add], axis = 1)
            
            pred.loc[:, 'date'] = latest_month
            pred = pred.merge(mcap_tbl, how = 'left', on = ['ticker', 'date'])
            pred = pred.loc[pred.mcap_pct_rank <= 0.5, :]

            top_predict = pred.sort_values(by = 'pred', ascending = False).reset_index(drop=True)
            top_predict = top_predict.drop(columns = ['marketcap', 'mcap_pct_rank'])
            
            ## top 50, full replacement
            top_cutoff = 50
            portfolio = top_predict.iloc[0:top_cutoff, :]
            
            portfolio.loc[:, 'score'] = score 

            all_portfolio = pd.concat([all_portfolio, portfolio])

            # add latest month to train set
            train = pd.concat([train, latest_data])

        except Exception as e:
            print("error on date ", str(latest_month))
            print(e)

    return all_portfolio

def run_backtest_pred_threshold_top_mcap(all_data, Y_field, incl, xgb_params, pred_threshold, cutoff, mcap_tbl):
    date_list = all_data.loc[:, 'date'].sort_values().unique() 
    train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
    train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)

    all_portfolio = pd.DataFrame()

    start = cutoff
    for k in range(start,len(date_list)):
        try:
            prev_month = date_list[k-1]
            latest_month = date_list[k]

            latest_data = all_data.loc[all_data.date == latest_month, :].reset_index(drop = True)

            X_train = train.loc[:, incl]
            Y_train = train.loc[:, Y_field]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, Y_train)

            # make predictions
            
            pred_class = pd.DataFrame(model.predict(latest_data.loc[:, incl]), columns = ['pred_label']) 
            score = f1_score(latest_data.loc[:, Y_field], pred_class)
            print(str(latest_month), ': ', str(score))

            pred = pd.DataFrame(model.predict_proba(latest_data.loc[:, incl])[:, 1], columns = ['pred']) 
            to_add = latest_data.loc[:, ['ticker', 'next_month_return']]
            pred = pd.concat([pred, to_add], axis = 1)

            to_add = latest_data.loc[:, Y_field]
            to_add.name = 'dependent_var'
            pred = pd.concat([pred, to_add], axis = 1)
            
            pred.loc[:, 'date'] = latest_month
            pred = pred.merge(mcap_tbl, how = 'left', on = ['ticker', 'date'])
            pred = pred.loc[pred.mcap_pct_rank <= 0.5, :]

            top_predict = pred.sort_values(by = 'pred', ascending = False).reset_index(drop=True)
            top_predict = top_predict.drop(columns = ['marketcap', 'mcap_pct_rank'])
            
            ## top 50, full replacement
            top_cutoff = 50
            if k == start:
                portfolio = top_predict.iloc[0:top_cutoff, :]

            else:
            # top 50, replace according to rules
                prev_portfolio = all_portfolio.loc[all_portfolio.date == prev_month, ['ticker', 'pred']]
                prev_portfolio['is_prev'] = 1
                threshold = min(pred_threshold, top_predict.pred[50])
                new_ptf = top_predict.loc[top_predict.pred >= threshold, :]
                match_tbl = new_ptf.merge(prev_portfolio, how = 'left', on = 'ticker', suffixes=('', '_prev'))
                match_tbl['is_prev'] = match_tbl['is_prev'].fillna(0)
                match_tbl = match_tbl.sort_values(by = ['is_prev', 'pred'], ascending = False).reset_index(drop = True)

                portfolio = match_tbl.loc[0:top_cutoff, top_predict.columns]

            portfolio.loc[:, 'score'] = score 

            all_portfolio = pd.concat([all_portfolio, portfolio])
            
            # add latest month to train set
            train = pd.concat([train, latest_data])

        except Exception as e:
            print("error on date ", str(latest_month))
            print(e)

    return all_portfolio

def backtest_classifier_base_ipo_delay(all_data, Y_field, incl, xgb_params, cutoff): 
    date_list = all_data.loc[:, 'date'].sort_values().unique()
    train_date = pd.DataFrame(date_list[0:cutoff], columns = ['date'])
    train = all_data.merge(train_date, how = 'inner', on ='date').reset_index(drop=True)

    all_portfolio = pd.DataFrame()
    start = cutoff
    for k in range(start,len(date_list)):
        try:
            prev_month = date_list[k-1]
            latest_month = date_list[k]

            latest_data = all_data.loc[all_data.date == latest_month, :].reset_index(drop = True)

            record_count = train.groupby('ticker')['date'].count().reset_index()
            record_keep = record_count.loc[record_count.date > 3, 'ticker']
            train = train.merge(record_keep, how = 'inner', on = 'ticker')

            X_train = train.loc[:, incl]
            Y_train = train.loc[:, Y_field]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, Y_train)

            # make predictions
            
            pred_class = pd.DataFrame(model.predict(latest_data.loc[:, incl]), columns = ['pred_label']) 
            score = f1_score(latest_data.loc[:, Y_field], pred_class)
            #print(str(latest_month), ': ', str(score))

            pred = pd.DataFrame(model.predict_proba(latest_data.loc[:, incl])[:, 1], columns = ['pred']) 
            to_add = latest_data.loc[:, ['ticker', 'next_month_return']]
            pred = pd.concat([pred, to_add], axis = 1)

            to_add = latest_data.loc[:, Y_field]
            to_add.name = 'dependent_var'
            pred = pd.concat([pred, to_add], axis = 1)

            top_predict = pred.sort_values(by = 'pred', ascending = False).reset_index(drop=True)

            ## top 50, full replacement
            top_cutoff = 50
            portfolio = top_predict.iloc[0:top_cutoff, :]

            portfolio.loc[:, 'date'] = latest_month
            portfolio.loc[:, 'score'] = score 

            all_portfolio = pd.concat([all_portfolio, portfolio])

            # add latest month to train set
            train = pd.concat([train, latest_data])

        except Exception as e:
            print("error on date ", str(latest_month))
            print(e)

    return all_portfolio


### Write functions

def write_model_to_db(Y_field, incl, xgb_params, score, importance, model_result_table_id, model_variable_table_id):
    client = bigquery.Client()
    query_job = client.query("""
    SELECT max(model_id) as latest
    FROM {}
    """.format(model_result_table_id))

    last_num = query_job.result().to_dataframe()
    model_id = last_num.iloc[0,0] + 1

    parameters_tbl = pd.DataFrame([xgb_params], columns = xgb_params.keys()).transpose().reset_index()
    parameters_tbl.columns = ['parameter', 'value']
    parameters_tbl['value'] = parameters_tbl['value'].astype(str)
    parameters_tbl.loc[:, 'model_id'] = model_id
    parameters_tbl.loc[:, 'model_type'] = 'XGBClassifier'
    parameters_tbl.loc[:, 'model_run_date'] = datetime.now()
    parameters_tbl.loc[:, 'score'] = score
    parameters_tbl.loc[:, 'score_type'] = 'f1'
    parameters_tbl.loc[:, 'dependent_var'] = Y_field
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(
                    parameters_tbl, model_result_table_id, job_config=job_config
                )
    
    gain_tbl = pd.DataFrame([importance], columns = importance.keys()).transpose()
    gain_tbl = gain_tbl.reset_index()
    gain_tbl.columns = ['variable', 'importance']
    gain_tbl = gain_tbl.merge(pd.DataFrame(incl, columns=['variable']), how = 'outer', on ='variable').fillna(0)
    gain_tbl.loc[:, 'model_id'] = model_id
    #gain_tbl.to_csv('variables.csv')

    job = client.load_table_from_dataframe(gain_tbl, model_variable_table_id, job_config=job_config)
    return model_id

def write_backtest_to_db(all_portfolio, model_id, dependent_var_name, description, train_dataset, write_table_id):
    client = bigquery.Client()
    query_job = client.query("""
    SELECT max(backtest_id) as latest
    FROM {}
    """.format(write_table_id))

    last_num = query_job.result().to_dataframe()
    backtest_id = last_num.iloc[0,0] + 1

    all_portfolio.loc[:, 'backtest_id'] = backtest_id
    all_portfolio.loc[:, 'model_id'] = model_id
    all_portfolio.loc[:, 'dependent_var_name'] = dependent_var_name
    all_portfolio.loc[:, 'description'] = description
    all_portfolio.loc[:, 'train_dataset'] = train_dataset
    all_portfolio.loc[:, 'prediction_type'] = 'probability'
    all_portfolio.loc[:, 'score_type'] = 'f1'
    all_portfolio['dependent_var'] = all_portfolio['dependent_var'].astype(float)
    all_portfolio['run_date'] = datetime.now()

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(
                all_portfolio, write_table_id, job_config=job_config
            )
    
    return backtest_id

def write_combo_backtest_to_db(all_portfolio, model_id, dependent_var_name, description, train_dataset, write_table_id):
    client = bigquery.Client()
    query_job = client.query("""
    SELECT max(backtest_id) as latest
    FROM {}
    """.format(write_table_id))

    last_num = query_job.result().to_dataframe()
    backtest_id = last_num.iloc[0,0] + 1

    all_portfolio.loc[:, 'backtest_id'] = backtest_id
    all_portfolio.loc[:, 'model_id'] = model_id
    all_portfolio.loc[:, 'description'] = description
    all_portfolio.loc[:, 'train_dataset'] = train_dataset
    all_portfolio['run_date'] = datetime.now()

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(
                all_portfolio, write_table_id, job_config=job_config
            )
    
    return backtest_id

def write_prod_strategy_to_db(all_portfolio, backtest_id, model_id, dependent_var_name, algorithm, train_dataset, write_table_id):
    client = bigquery.Client()

    all_portfolio.loc[:, 'backtest_id'] = backtest_id
    all_portfolio.loc[:, 'model_id'] = model_id
    all_portfolio.loc[:, 'dependent_var_name'] = dependent_var_name
    all_portfolio.loc[:, 'description'] = algorithm
    all_portfolio.loc[:, 'train_dataset'] = train_dataset
    all_portfolio.loc[:, 'prediction_type'] = 'probability'
    all_portfolio.loc[:, 'score_type'] = 'f1'
    all_portfolio['dependent_var'] = all_portfolio['dependent_var'].astype(float)
    all_portfolio['run_date'] = datetime.now()

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(
                all_portfolio, write_table_id, job_config=job_config
            )
    

## performance calculation
def create_monthly_return(this_ptf, cost_per_trade):
    this_ptf = this_ptf.sort_values(by = ['date'])

    monthly_return = this_ptf.groupby(['date'])['next_month_return'].mean().reset_index()

    na_list = monthly_return.loc[monthly_return.next_month_return.isnull(), 'date'].tolist()
    na_list = [str(x) for x in na_list]

    #if len(na_list) > 0 :
        #print("check NAs:", ', '.join(na_list))

    monthly_return['next_month_return'] = monthly_return.next_month_return.fillna(0)
    
    # calculate turnover
    date_list = this_ptf.loc[:, 'date'].sort_values().unique()

    turnover_tbl = list()
    for i in range(1, len(date_list)):
        first_month = date_list[i-1]
        next_month = date_list[i]
        first_ptf = this_ptf.loc[this_ptf.date == first_month, 'ticker'].values
        next_ptf = this_ptf.loc[this_ptf.date == next_month, 'ticker'].values

        idx = np.in1d(next_ptf, first_ptf)
        turnover = len(next_ptf[~idx])

        this_turnover = [first_month, turnover]
        turnover_tbl.append(this_turnover)

    turnover_tbl = pd.DataFrame(turnover_tbl, columns = ['date', 'turnover'])

    monthly_return = monthly_return.merge(turnover_tbl, how = 'left', on = 'date')

    monthly_return.loc[:, 'trade_costs'] = monthly_return.turnover * 1/50 * cost_per_trade
    monthly_return.loc[:, 'total_return'] = monthly_return.next_month_return - monthly_return.trade_costs 

    monthly_return.loc[:, 'total_return_idx'] = 100.0

    for i in range(1, len(monthly_return)):
        monthly_return.loc[i, 'total_return_idx'] =  monthly_return.loc[i-1, 'total_return_idx'] * (1+monthly_return.loc[i, 'total_return'])

    return monthly_return

def create_performance_stats(monthly_return, start, end):
    
    return_5y = monthly_return.loc[end, 'total_return_idx'] / monthly_return.loc[start, 'total_return_idx'] - 1
    std_5y = monthly_return.loc[start:end, 'total_return'].std() * np.sqrt(12)
    sharpe_5y = monthly_return.loc[start:end, 'total_return'].mean()*12 / (monthly_return.loc[start:end, 'total_return'].std() * np.sqrt(12))
    drawdown_5y = monthly_return.loc[start:end, 'total_return'].min()
    drawdown_5y_dt = monthly_return.loc[monthly_return.total_return == drawdown_5y, 'date'].values[0]

    std = monthly_return.loc[0:end, 'total_return'].std() * np.sqrt(12)
    sharpe = monthly_return.loc[0:end, 'total_return'].mean()*12 / (monthly_return.loc[0:end, 'total_return'].std() * np.sqrt(12))
    drawdown = monthly_return.loc[0:end, 'total_return'].min()
    drawdown_dt = monthly_return.loc[monthly_return.total_return == drawdown, 'date'].values[0]

    this_stat = [return_5y, std_5y, sharpe_5y, drawdown_5y, drawdown_5y_dt, std, sharpe, drawdown, drawdown_dt]
    this_col = ['return_5y_latest', 'std_5y', 'sharp_5y', 'max_5y_drawdown', '5y_drawdown_dt', 'std_all', 'sharpe_all', 'max_drawdown', 'drawdown_dt']
    
    all = zip(this_col, this_stat)
    stat_tbl = pd.DataFrame(all)

    return stat_tbl


def create_performance_stats_v2(monthly_return, lookback):
    
    end = len(monthly_return)-2
    start = end - lookback*12

    return_ = monthly_return.loc[end, 'total_return_idx'] / monthly_return.loc[start, 'total_return_idx'] - 1
    std = monthly_return.loc[start:end, 'total_return'].std() * np.sqrt(12)
    sharpe = monthly_return.loc[start:end, 'total_return'].mean()*12 / (monthly_return.loc[0:end, 'total_return'].std() * np.sqrt(12))
    drawdown = monthly_return.loc[start:end, 'total_return'].min()
    drawdown_dt = monthly_return.loc[monthly_return.total_return == drawdown, 'date'].values[0]

    this_stat = [return_, std, sharpe, drawdown, drawdown_dt]
    this_col = ['return', 'std', 'sharp', 'max_drawdown', 'drawdown_dt']
    
    all = zip(this_col, this_stat)
    stat_tbl = pd.DataFrame(all, columns = ['field', 'value'])
    stat_tbl.loc[:, 'period'] = str(lookback) + 'y'

    return stat_tbl



# combo ranks
def create_combo_factor_ptf(mfactor_rank, Top_cutoff):
        all_dates = mfactor_rank['date'].unique()
        
        multi_factor=pd.DataFrame()
        for date in all_dates:
            X = mfactor_rank.loc[mfactor_rank.date == date, :]
            Y = X.dropna(subset=["ComboRank"]).sort_values(["ComboRank"]).reset_index(drop=True)
            Z = Y.iloc[0:Top_cutoff, :]
            multi_factor = pd.concat([multi_factor, Z])
        
        return(multi_factor)
