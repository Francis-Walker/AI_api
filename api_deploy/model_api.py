import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from flask import Flask, request, jsonify
import json


app = Flask(__name__)

@app.route('/', methods=['GET'])
def getMill():

    raw_ts = pd.read_pickle('./time_series.pk1')
    raw_ts=raw_ts.reset_index(drop=True)
    raw_ts.drop(['var52(t-3)','var52(t-2)','var52(t-1)'],axis='columns', inplace=True)
    raw_ts = raw_ts.sort_values(by=['var2(t)','var3(t)'])
    raw_ts=raw_ts.reset_index(drop=True)
    raw_val = raw_ts.values 
    scaler = MinMaxScaler(feature_range=(0, 1))
    raw_scaled = scaler.fit_transform(raw_val)
    raw_eval = raw_scaled[57193:,:]
    raw_train_test = raw_scaled[:57193,:]
    raw_train_test_x = raw_train_test[:, :-1]
    raw_train_test_y = raw_train_test[:, -1]
    x_train= raw_train_test_x[:42588, :]
    x_test = raw_train_test_x[42588:, :]
    y_train=raw_train_test_y[:42588]
    y_test= raw_train_test_y[42588:]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    raw_eval_x = raw_eval[:, :-1]
    x_eval= raw_eval_x.reshape((raw_eval_x.shape[0], 1, raw_eval_x.shape[1]))
    raw_est = pd.read_csv("RVESTfull.csv") 
    extract_columns = [154,155,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204]
    col_names = ['SEASON','SUGARMONTH','Amatikulu','Darnall','Eston','Felixton','Gledhow','Komati','Maidstone','Malelane','Noodsberg','Pongola','Sezela','UCL','Umfolozi','Umzimkulu','ErrorRV']
    dummie_cols = ['Amatikulu','Darnall','Eston','Felixton','Gledhow','Komati','Maidstone','Malelane','Noodsberg','Pongola','Sezela','UCL','Umfolozi','Umzimkulu']


    y_pred_1 = model.predict(x_test)
    x_test_1 = x_test.reshape((x_test.shape[0], x_test.shape[2]))
    test_1 =  concatenate((x_test_1,y_pred_1), axis=1 )
    test_1_scaled =  scaler.inverse_transform(test_1)
    y_test = y_test.reshape(x_test.shape[0],1)

    test_1_actual =  concatenate((x_test_1,y_test), axis=1 )
    test_1_actual_scaled =  scaler.inverse_transform(test_1_actual)
    y_test_pred = test_1_scaled[:, -1]
    y_test_actual = test_1_actual_scaled[:, -1]
    df_test_actual = pd.DataFrame(test_1_actual_scaled)
    df_test_pred = pd.DataFrame(test_1_scaled)
    mill_season_month_error_actual_test = df_test_actual[df_test_actual[155]>5][extract_columns]
    mill_season_month_error_actual_test.columns = col_names

    mill_col = mill_season_month_error_actual_test[dummie_cols].idxmax(axis=1)
    mill_season_month_error_actual_test['mill'] = mill_col
    mill_season_month_error_actual_test.drop(dummie_cols,axis='columns', inplace=True)
    mill_season_month_error_actual_test
    mill_season_month_error_pred_test = df_test_pred[df_test_pred[155]>5][extract_columns]
    mill_season_month_error_pred_test.columns = col_names
    mill_season_month_error_actual_test['pred_ErrorRv']=mill_season_month_error_pred_test['ErrorRV']
    eval_1 = mill_season_month_error_actual_test
    eval_1['SUGARMONTH'] = eval_1['SUGARMONTH'].round()

    ev_1 = eval_1[eval_1['SUGARMONTH']<9.5].groupby(by=['mill','SUGARMONTH'])[['pred_ErrorRv','ErrorRV']].mean()
    ev_1 = ev_1.reset_index(drop=False)
    final_op_test = pd.merge(left= raw_est[(raw_est['SUGARMONTH']>6.5)&(raw_est['fa_SEASON']==2020)], right=ev_1[['mill','SUGARMONTH','pred_ErrorRv']], how='left', left_on=['cf_mill','SUGARMONTH'], right_on=['mill','SUGARMONTH'])
    final_op_test['pred_rv'] = final_op_test['FCFORECAST'] + final_op_test['pred_ErrorRv']
    final_op_test = final_op_test.dropna(how='any')
    final_op_test.columns= ['SUGARMONTH', 'FC_FORECAST', 'Actual_RV', 'ErrorRV', 'cf_mill', 'fa_SEASON','mill', 'pred_ErrorRv', 'Prediction']
    test_op = final_op_test[['fa_SEASON','SUGARMONTH','FC_FORECAST','Actual_RV','Prediction','mill']]

    extract_columns = [154,155,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204]
    col_names = ['SEASON','SUGARMONTH','Amatikulu','Darnall','Eston','Felixton','Gledhow','Komati','Maidstone','Malelane','Noodsberg','Pongola','Sezela','UCL','Umfolozi','Umzimkulu','ErrorRV']
    dummie_cols = ['Amatikulu','Darnall','Eston','Felixton','Gledhow','Komati','Maidstone','Malelane','Noodsberg','Pongola','Sezela','UCL','Umfolozi','Umzimkulu']
    y_pred_2 = model.predict(x_train)
    x_train_2 = x_train.reshape((x_train.shape[0], x_train.shape[2]))
    train_2 =  concatenate((x_train_2,y_pred_2), axis=1 )
    train_1_scaled =  scaler.inverse_transform(train_2)
    y_train = y_train.reshape(y_train.shape[0],1)
    train_1_actual =  concatenate((x_train_2,y_train), axis=1 )
    train_1_actual_scaled =  scaler.inverse_transform(train_1_actual)
    y_train_pred = train_1_scaled[:, -1]
    y_train_actual = train_1_actual_scaled[:, -1]
    df_train_actual = pd.DataFrame(train_1_actual_scaled)
    df_train_pred = pd.DataFrame(train_1_scaled)
    mill_season_month_error_actual_train = df_train_actual[extract_columns].copy()
    mill_season_month_error_actual_train.columns = col_names

    mill_col = mill_season_month_error_actual_train[dummie_cols].idxmax(axis=1)
    mill_season_month_error_actual_train['mill'] = mill_col
    mill_season_month_error_actual_train.drop(dummie_cols,axis='columns', inplace=True)
    mill_season_month_error_actual_train
    mill_season_month_error_pred_train = df_train_pred[extract_columns]
    mill_season_month_error_pred_train.columns = col_names
    mill_season_month_error_actual_train['pred_ErrorRv']=mill_season_month_error_pred_train['ErrorRV']
    eval_2 = mill_season_month_error_actual_train
    eval_2['SUGARMONTH'] = eval_2['SUGARMONTH'].round()
    ev_2 = eval_2[eval_2['SUGARMONTH']<9.5].groupby(by=["SEASON",'mill','SUGARMONTH'])[['pred_ErrorRv','ErrorRV']].mean()
    ev_2 = ev_2.reset_index(drop=False)
    ev_2
    final_op_train = pd.merge(left= raw_est, right=ev_2[['mill','SEASON','SUGARMONTH','pred_ErrorRv']], how='left', left_on=['cf_mill','fa_SEASON','SUGARMONTH'], right_on=['mill','SEASON','SUGARMONTH'])
    final_op_train = final_op_train.dropna(how='any')
    final_op_train['pred_rv'] = final_op_train['FCFORECAST'] + final_op_train['pred_ErrorRv']
    final_op_train.drop(['SEASON'],axis='columns', inplace=True)
    final_op_train.columns= ['SUGARMONTH', 'FC_FORECAST', 'Actual_RV', 'ErrorRV', 'cf_mill', 'fa_SEASON','mill', 'pred_ErrorRv', 'Prediction']       
    train_op = final_op_train[['fa_SEASON','SUGARMONTH','FC_FORECAST','Actual_RV','Prediction','mill']]



    extract_columns = [154,155,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204]
    col_names = ['SEASON','SUGARMONTH','Amatikulu','Darnall','Eston','Felixton','Gledhow','Komati','Maidstone','Malelane','Noodsberg','Pongola','Sezela','UCL','Umfolozi','Umzimkulu','pred_ErrorRv']
    dummie_cols = ['Amatikulu','Darnall','Eston','Felixton','Gledhow','Komati','Maidstone','Malelane','Noodsberg','Pongola','Sezela','UCL','Umfolozi','Umzimkulu']
    y_pred_eval = model.predict(x_eval)
    x_eval_1 = x_eval.reshape((x_eval.shape[0], x_eval.shape[2]))
    eval =  concatenate((x_eval_1,y_pred_eval), axis=1 )
    eval_scaled =  scaler.inverse_transform(eval)
    eval_pred = eval_scaled[:, -1]
    df_eval  = pd.DataFrame(eval_scaled)
    mill_season_month_error_actual_eval = df_eval[extract_columns].copy()
    mill_season_month_error_actual_eval.columns = col_names
    mill_col = mill_season_month_error_actual_eval[dummie_cols].idxmax(axis=1)
    mill_season_month_error_actual_eval['mill'] = mill_col
    mill_season_month_error_actual_eval.drop(dummie_cols,axis='columns', inplace=True)

    eval_3 = mill_season_month_error_actual_eval
    eval_3['SUGARMONTH'] = eval_3['SUGARMONTH'].round()
    ev_3 = eval_3[eval_3['SUGARMONTH']<9.5].groupby(by=["SEASON",'mill','SUGARMONTH'])[['pred_ErrorRv']].mean()
    ev_3 = ev_3.reset_index(drop=False)
    ev_3
    final_op_eval = pd.merge(left= raw_est[raw_est['fa_SEASON']==2021], right=ev_3[['mill','SEASON','SUGARMONTH','pred_ErrorRv']], how='left', left_on=['cf_mill','fa_SEASON','SUGARMONTH'], right_on=['mill','SEASON','SUGARMONTH'])
    final_op_eval.drop(['SEASON','SRV','ErrorRV'],axis='columns', inplace=True)
    final_op_eval = final_op_eval.dropna(how='any')
    final_op_eval['pred_rv'] = final_op_eval['FCFORECAST'] + final_op_eval['pred_ErrorRv']
    final_op_eval.columns= ['SUGARMONTH', 'FC_FORECAST', 'cf_mill', 'fa_SEASON','mill', 'pred_ErrorRv', 'Prediction']       
    eval_op = final_op_eval[['fa_SEASON','SUGARMONTH','FC_FORECAST','Prediction','mill']]

    print(test_op.shape)
    print(train_op.shape)
    print(eval_op.shape)
    mill = request.args.get('mill', type = str)
    print(mill)

    response =  jsonify({'train': json.loads(train_op[train_op['mill']==mill].to_json(orient='index')) , 'test': json.loads(test_op[test_op['mill']==mill].to_json(orient='index')) ,'eval': json.loads(eval_op[eval_op['mill']==mill].to_json(orient='index'))})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response




if __name__ == '__main__':

    model = load_model('./Model3')
    app.run(debug=True, host='0.0.0.0')

















