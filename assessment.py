#%% importing libraries
import os
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from assessmentfunction import ExploratoryDataAnalysis
from assessmentfunction import ModelCreation
from assessmentfunction import ModelEvaluation

data_path = "cases_malaysia_covid.csv" 
df = pd.read_csv(data_path)
df.fillna(0, inplace=True)
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'Saved_path', 'model.h5')
LOG_PATH = os.path.join(os.getcwd(), 'Log')

# Constant
window_size = 30 # how many day of prediction

#%% EDA
# Step 1) Load data
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

# Step 2) Analyse the loaded data

train_df.info()
train_df.duplicated().sum()
train_df.isnull().mean()
# There are missing data on all the cluster columns, about 50% data are missing
# Date & cases_new column in object datatype
# No duplicated data found

test_df.describe()
test_df.info()
test_df.duplicated().sum()
# there is 1 missing data in 1 column
# Only date in object datatype
# No duplicated data found

# Discussion:
# Since we are going to analyse 'new_cases' column only,
# we can ignore all cluster columns.

# Step 3) Data cleaning
# Try changing type of cases_new column datatype to int type got error. 
# Double check on the column, its contain symbol '?' 
# and there are missing value but not detected as NaN
train_df['cases_new'] = train_df['cases_new'].str.replace(r'[^0-9a-zA-Z:,]+', '')
train_df['cases_new'] = train_df['cases_new'].replace(r'^\s*$', np.NaN, regex=True)
train_df['cases_new'] = train_df['cases_new'].replace(np.NaN, 0, regex=True)
train_df['cases_new'] = train_df['cases_new'].astype('float')
train_df.info()

# Similar to test_df
test_df['cases_new'] = test_df['cases_new'].replace(r'[^0-9a-zA-Z:,]+', '')
test_df['cases_new'] = test_df['cases_new'].replace(r'^\s*$', np.NaN, regex=True)
test_df['cases_new'] = test_df['cases_new'].replace(np.NaN, 0, regex=True)
test_df['cases_new'] = test_df['cases_new'].astype('float')

# Now column cases_new has 12 missing value, cluster columns with 50% NaN
# We will do KNN imputer on the missing data
eda = ExploratoryDataAnalysis()
train_df_clean = eda.knn_imputer(train_df,n_neighbors=24)
test_df_clean = eda.knn_imputer(test_df,n_neighbors=24)

# Step 5) Feature selection
# Step 6) Data preprocessing
# In this analysis, we only choose 'cases_new' with index 0
scaled_train_df = eda.mm_scaler(train_df_clean, index_column=0)
scaled_test_df = eda.mm_scaler(test_df_clean, index_column=0)

# Testing dataset with windows 30 day
window_size = 30

# train data
#eda.train_process_window(train_df, scaled_train_df)
X_train=[]
Y_train=[]

for i in range(window_size, len(train_df)):
    X_train.append(scaled_train_df[i-window_size:i,0])
    Y_train.append(scaled_train_df[i,0])
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# test data
temp = np.concatenate((scaled_train_df, scaled_test_df))
length_window = window_size+len(scaled_test_df)
temp = temp[-length_window:] 

#eda.train_process_window(temp, window_size=30)
X_test=[]
Y_test=[]

for i in range(window_size, len(temp)):
    X_test.append(temp[i-window_size:i,0])
    Y_test.append(temp[i,0])
    
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# expend dimension
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
  
#%% Model creation
mc = ModelCreation()
model = mc.lstm_layer(X_train, nodes=64, dropout=0.2, output=1)

model.compile(optimizer='adam', loss='mse', metrics='mse')

# callback
log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir)
early_stopping_callback = EarlyStopping(monitor='loss', patience=15)

hist = model.fit(X_train, Y_train, epochs=200, batch_size=200,
                 validation_data=(X_test,Y_test),
                 callbacks=[tensorboard_callback, early_stopping_callback])

print(hist.history.keys())

#%% model deployment
predicted = [] 

for test in X_test:
    predicted.append(model.predict(np.expand_dims(test,axis=0)))

predicted = np.array(predicted)

y_true = Y_test
y_pred = predicted.reshape(len(predicted),1)

#%% Model Analysis
me = ModelEvaluation()
me.model_report(y_true, y_pred)

#%% Save model
model.save(MODEL_SAVE_PATH)