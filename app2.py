import numpy as np
import pandas as pd
import datatable as dt
from tqdm import tqdm
import joblib
import xgboost as xgb
from sklearn.metrics import make_scorer, matthews_corrcoef
from datetime import datetime
from datatable import f, min, max

import os
from flask import Flask, flash,jsonify, request, redirect, render_template
from werkzeug.utils import secure_filename

app=Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #flash('File(s) successfully uploaded')
        def function1(input_file): #This function takes an input where features are separated by comma in the same order as training data given
            def cat_data(df_cat_nan):
                df_cat = df_cat_nan.fillna('not_passed')
                
                encoder = joblib.load('pkl_files/OrdinalEncoder.pkl')
                model_ftrl = joblib.load('pkl_files/model_ftrl.pkl')
                imp_cols = joblib.load('pkl_files/cat_data_imp_cols.pkl')
                min_val = joblib.load('pkl_files/min_val_each_col_in_cat_data.pkl')
                df_cat = df_cat[imp_cols[1:]]
        
                #df_cat = df_cat.drop('Id',axis=1)
                row = encoder.transform(np.array(df_cat.iloc[0]).reshape(-1,1))
                df_cat.iloc[0] = row.reshape(1,-1)[0]
                df_cat = df_cat.astype('int32')
                cat_dt = dt.Frame(df_cat)
                col_name = cat_dt.names
                
                for col in range(len(col_name)):
                    cols = col_name[col]
                    minVal = min_val[col]
                    cat_dt[:,f[col]] = cat_dt[:, (f[col] - minVal)/(93 - minVal)]
        
                prediction = model_ftrl.predict(cat_dt)
        
                df_cat_output = pd.DataFrame()
        
                df_cat_output = df_cat_nan[['Id']].copy()
                df_cat_output['no_of_category'] = df_cat_nan.count(axis=1)-1 #no of values except nan
        
                different_cat = len(df_cat_nan.iloc[0].value_counts())-1
                df_cat_output['different_category'] = different_cat #total no of different category used in a single product
        
                df_cat_output['probability_value'] = prediction[0,0]
        
                return df_cat_output
            
            def nume_data(df_nume):
                columns = df_nume.columns
                imp_numerical_col = joblib.load('pkl_files/imp_numerical_features_name.pkl')
                imp_numerical_col.append("Id")
                df = df_nume[imp_numerical_col]
                return df
            
            def date_data(df_date):
                imp_date_cols = joblib.load('pkl_files/imp_date_cols_name.pkl')
                imp_date_cols = list(imp_date_cols.values)
                imp_date_cols.append('Id')
                imp_date_single_cols = joblib.load('pkl_files/imp_date_single_cols_name.pkl')
                col_24_25 = joblib.load('pkl_files/col_24_25.pkl')
        
                diff_st_date = []
                for i in range(51):
                    diff_st_date.append('S'+str(i))
        
                ohe_station = pd.DataFrame(columns=diff_st_date)
                df = df_date[imp_date_single_cols]
        
                leave_col = ['S24','S25']
                for col in df.columns:
                    lst = [0]*len(df)
                    st = col.split("_")[1]
                    if st not in leave_col:
                        temp = df[col]
                        for i in range(len(temp)):
                            val = temp.iloc[i]
                            if str(val) != 'nan':
                                lst[i] = 1
                    ohe_station[st] = lst
        
                col_24, col_25 = col_24_25[0], col_24_25[1]
                df_24, df_25 = df_date[col_24], df_date[col_25]
        
                n_24 = [0]*len(df_24)
                for i in range(len(df_24)):
                    row = df_24.iloc[i]
                    if row.count() != 0:
                        n_24[i] = 1
                ohe_station['S24'] = n_24
        
                n_25 = [0]*len(df_25)
                for i in range(len(df_25)):
                    row = df_25.iloc[i]
                    if row.count() != 0:
                        n_25[i] = 1
                ohe_station['S25'] = n_25
        
                start_station = []
                end_station = []
                path_length = []
                for i in range(len(ohe_station)):
                    row = ohe_station.iloc[i]
                    start = 0
                    end = 0
        
                    for j in range(len(row)):
                        if row[j] == 1:
                            start = j
                            break
                    for k in range(len(row)-1,-1,-1):
                        if row[k] == 1:
                            end = k
                            break
                    path_length.append(np.count_nonzero(row == 1))
                    start_station.append(start)
                    end_station.append(end)
        
                ohe_station['start_station'] = start_station
                ohe_station['end_station'] = end_station
                ohe_station['path_length'] = path_length
        
                ohe_station['Id'] = df_date['Id']
        
                #credit:-https://www.kaggle.com/choithuthoi/my-solution
        
                #new_date_feature = pd.DataFrame()
                columns = df_date.columns[1:] 
                new_date_feature = df_date[['Id']].copy()
                new_date_feature['mintime'] = df_date[columns].min(axis=1).values
                new_date_feature['maxtime'] = df_date[columns].max(axis=1).values
                new_date_feature['time_difference'] =  new_date_feature['maxtime'] - new_date_feature['mintime']
                new_date_feature.fillna(0,inplace=True)
                week_duration = 1679
        
                new_date_feature['part_week'] = ((new_date_feature['mintime'].values * 100)  % week_duration).astype(np.int64)
                new_date_feature['min_time_station'] =  df_date[columns].idxmin(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)
                new_date_feature['max_time_station'] =  df_date[columns].idxmax(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)
                
                final_date_feature = pd.merge(new_date_feature,ohe_station,on = 'Id')
                imp_date_feature = df_date[imp_date_cols[1:]]
                final_date_feature = pd.merge(final_date_feature,imp_date_feature,on='Id')
                
                line = ['L0','L1','L2','L3']
                d = {} #key=station name and value=all the date feature related to that station
                for l in line:
                    lst = []
                    for i in df_date.columns[1:]:
                        t = i.split("_")
                        if l in t:
                            lst.append(i)
                    d[l] = lst
                    
                fraction_of_measurements = df_date[['Id']].copy()
                for l in line:
                    col = d[l]
                    temp = df_date[col]
                    fraction_of_measurements['fraction_of_measurements'+l] = temp.count(axis=1)/len(col)
                    
                final_date_feature['average_time_per_measurement'] = final_date_feature['time_difference']/final_date_feature['path_length']
                final_date_feature = pd.merge(final_date_feature,fraction_of_measurements,on="Id")
        
                imp_station = ['S32', 'S24', 'S38', 'S26', 'S28', 'S27', 'S36', 'S37', 'S30', 'S29']
                lst = []
                imp_date_st = final_date_feature[imp_station]
                for i in range(len(final_date_feature)):
                    lst.append((np.count_nonzero(imp_date_st.iloc[i]))/10)
                final_date_feature['error_rate_on_imp_stations'] = lst
            
                imp_st_col_date = []
                columns = df_date.columns
                for col in columns[1:]:
                    if col.split("_")[1] in imp_station:
                        imp_st_col_date.append(col)
                
                df_date_ = df_date[imp_st_col_date]
                new_date_feature_imp_st = df_date[['Id']].copy()
                new_date_feature_imp_st['mintime_imp_st'] = df_date_.min(axis=1).values
                new_date_feature_imp_st['maxtime_imp_st'] = df_date_.max(axis=1).values
                new_date_feature_imp_st['time_difference_imp_st'] =  new_date_feature_imp_st['maxtime_imp_st'] - new_date_feature_imp_st['mintime_imp_st']
                df_ = pd.merge(final_date_feature,new_date_feature_imp_st,on="Id")
                
                row = df_date.iloc[0][1:]
                total_no_of_different_date = [len(pd.unique(row))]
                df_['total_no_of_different_date'] = total_no_of_different_date
        
                return df_
                
            files = os.listdir(input_file)
            
            input_file_cat = open(input_file+'/'+files[0],"r")
            input_cat = list(input_file_cat)
            input_file_cat.close()
            temp_cat = input_cat[0].split(",")
            lst_cat = [np.nan if i == 'nan' else i for i in temp_cat[1:]]
            lst_cat.insert(0,float(temp_cat[0]))
            
            input_file_nume = open(input_file+'/'+files[2],"r")
            input_nume = list(input_file_nume)
            input_file_nume.close()
            temp_nume = input_nume[0].split(",")
            lst_nume = [np.nan if i == 'nan' else float(i) for i in temp_nume]
            
            input_file_date = open(input_file+'/'+files[1],"r")
            input_date = list(input_file_date)
            input_file_date.close()
            temp_date = input_date[0].split(",")
            lst_date = [np.nan if i == 'nan' else float(i) for i in temp_date]
            
            cat_cols = joblib.load('pkl_files/categorical_columns')
            nume_cols = joblib.load('pkl_files/numerical_columns')
            date_cols = joblib.load('pkl_files/date_columns')
                
            cat_df = pd.DataFrame(columns = cat_cols)
            nume_df = pd.DataFrame(columns = nume_cols)
            date_df = pd.DataFrame(columns = date_cols)
            
            cat_df.loc[0] = lst_cat
            nume_df.loc[0] = lst_nume
            date_df.loc[0] = lst_date
            
            cat_output = cat_data(cat_df)
            nume_output = nume_data(nume_df)
            date_output = date_data(date_df)
            
            final_df = pd.merge(cat_output,nume_output,on="Id")
            final_df = pd.merge(final_df,date_output,on='Id')
            
            model = joblib.load('pkl_files/final_model.pkl')
            prediction = model.predict(final_df.drop('Id',axis=1))
            if prediction[0] == 0:
                output = 'Not Defective'
            if prediction[0] == 1:
                output = 'Defective'
            return output
        
        output = function1('uploads')
        return jsonify({'prediction': output})


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=False,threaded=True)