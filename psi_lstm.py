
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

psi_df = pd.read_csv('./PSI/psi_df_2016_2019.csv', parse_dates=True)
print(psi_df.head(3))

#Convert 'datetime' column to date/time type
#Use temporary dataframe for making a datetime column
#Make datetime column

columns_names = psi_df.keys().values.tolist()
psi_df['year'] = psi_df['timestamp'].str[0:4].astype(int)
psi_df['month'] = psi_df['timestamp'].str[5:7].astype(int)
psi_df['day'] = psi_df['timestamp'].str[8:10].astype(int)
psi_df['hour'] = psi_df['timestamp'].str[11:13].astype(int)
psi_df['minute'] = psi_df['timestamp'].str[14:16].astype(int)
psi_df['second'] = psi_df['timestamp'].str[17:19].astype(int)


temp_datetime_df = psi_df.drop(columns=columns_names)  
psi_df['datetime'] = pd.to_datetime(temp_datetime_df) 

temp_columns=['timestamp','year','month','day', 'hour','minute','second']
psi_df = psi_df.drop(columns=temp_columns)
print('Min : {}, Max : {}'.format(min(psi_df.datetime), max(psi_df.datetime)))

print(psi_df.describe())

psi_df= psi_df.set_index("datetime") # set 'datetime' column for better management of time series 
print(psi_df.head(3))

psi_week=psi_df['2016':].resample('w').mean().national # resample by 'national' column week to week data
psi_week.plot(ylim=[0,140])

psi_day=psi_df['2016':].resample('D').mean().national # resample by 'national' column day to day data
psi_day.plot(ylim=[0,140])

psi_day_all=psi_df['2016':].resample('D').mean() # resample by all columns day to day data
psi_day_all.plot(ylim=[0,140])

psi_day_all= psi_day_all.dropna()
psi_day_all.isnull().sum()

# Making model based on 4 days prediction just for 'national' column
# You can use any other column base on your priority
train_data= psi_day_all
X_length = 4 
X_y = np.zeros((train_data.shape[0] - X_length, X_length + 1))
for i in range(X_y.shape[0]):
    X_y[i, :] = train_data['national'].to_numpy()[i:i + X_length + 1]

# Split train and test sets
test_ratio = 0.2
indices = np.arange(start=0, stop=X_y.shape[0])
test_set_size = int(X_y.shape[0] * test_ratio)
train_index = np.sort(indices[:-test_set_size])
test_index = np.sort(indices[-test_set_size:])

train_data = X_y[train_index, :]
test_data = X_y[test_index, :]

print(train_data.shape[0], "train +", test_data.shape[0], "test")

# np.save('./PSI/train_data', train_data)
# np.save('./PSI/test_data', test_data)
y_train = train_data[:, -1]
X_train = train_data[:, :-1]

# All training data is used to set Min and Max Parameters

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_data.reshape(-1, 1))  
X_train_normalized = scaler.transform(X_train)
y_train_normalized = scaler.transform(y_train.reshape(-1, 1))

# Split train and validation sets
np.random.seed(123)
val_ratio = 0.1
indices = np.random.permutation(X_train_normalized.shape[0])
val_set_size = int(X_train_normalized.shape[0] * val_ratio)
train_indices = np.sort(indices[:-val_set_size])
val_indices = np.sort(indices[-val_set_size:])

X_val = X_train_normalized[val_indices, :]
y_val = y_train_normalized[val_indices, :]
X_train = X_train_normalized[train_indices, :]
y_train = y_train_normalized[train_indices, :]

n_steps = 4
# reshape from [samples, timesteps] into [samples, timesteps, features] to make 3 dimentions 
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

# define model
model = Sequential()
model.add(Bidirectional(LSTM(30, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()

# use tensorboard to view metrics
tensorboard = TensorBoard(log_dir="logs")
# define early stopping to stop training after 5 epochs of not improving
early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)

batch_size = 32
epochs = 100
# train the model using the training set and validating using validation set
history= model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),callbacks=[tensorboard, early_stopping])

#Plot Training & Validation loss
history_dict=history.history
loss_values= history_dict['loss']
val_loss_values= history_dict['val_loss']
epochss= range(1, len(loss_values)+1)

plt.plot(epochss , loss_values, 'b' , label= 'Training loss')
plt.plot(epochss , val_loss_values, 'grey' , label= 'Validation loss')
plt.title('Training & Validation loss')

plt.xlabel('Epoch number')
plt.ylabel('MSE Loss')

# preparing Data and Label for test set
# Normalize test set with Scaler fitted with train data
# demonstrate prediction

y_test = test_data[:, -1]
X_test = test_data[:, :-1]

X_test_normalized = scaler.transform(X_test)
y_test_normalized = scaler.transform(y_test.reshape(-1, 1))

y_pred = np.zeros_like(y_test_normalized)

x_input = X_test_normalized.reshape((X_test_normalized.shape[0], n_steps, n_features))
y_pred = model.predict(x_input, verbose=0)

err = y_test_normalized - y_pred
# Root Mean Square Error
RMSE = np.sqrt(np.mean(np.power(err, 2)))
print(RMSE)

#Function to prepare and test a new samle
def predict_psi_level(X, model, scaler):
    # Normalize test set with Scaler fitted with train data
    X_normalized = scaler.transform(X.reshape(-1, 1))
    # Reshape matrix to 3 dimentions data
    x_input = X_normalized.reshape(1, n_steps, n_features)
    # Prediction
    prediction_normalized = model.predict(x_input, verbose=0) 
    # Inverse of Normalization
    prediction = np.squeeze(scaler.inverse_transform(prediction_normalized.reshape(-1, 1)))
    psi_bands = np.array([0., 56., 150., 250])
    psi_levels = ['Normal', 'Elevated', 'High', 'Very high']
    return psi_levels[np.argwhere(prediction >= psi_bands)[-1][0]]

i_test = 5
X = test_data[i_test, :-1]

PSI_level = predict_psi_level(X, model, scaler)
print(PSI_level)