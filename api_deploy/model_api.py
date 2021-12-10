import pandas as pd
from flask import Flask, request, jsonify
import json
import pickle5 as p 

app = Flask(__name__)

@app.route('/', methods=['GET'])
def getMill():
    with open('./ai_ops/train.pk1','rb') as f:
        train_op = p.load(f)
    train_op.columns = ['fa_SEASON', 'SUGARMONTH', 'FC_FORECAST', 'Actual_RV', 'Prediction',
       'mill']
    with open('./ai_ops/test.pk1','rb') as f:
        test_op = p.load(f)
    test_op.columns = ['fa_SEASON', 'SUGARMONTH', 'FC_FORECAST', 'Actual_RV', 'Prediction',
       'mill']
    with open('./ai_ops/eval.pk1','rb') as f:
        eval_op = p.load(f)
    eval_op.columns = ['fa_SEASON', 'SUGARMONTH', 'FC_FORECAST', 'Prediction',
       'mill']
    mill = request.args.get('mill', type = str)
    print(mill)

    response =  jsonify({'train': json.loads(train_op[train_op['mill']==mill].to_json(orient='index')) , 'test': json.loads(test_op[test_op['mill']==mill].to_json(orient='index')) ,'eval': json.loads(eval_op[eval_op['mill']==mill].to_json(orient='index'))})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

















