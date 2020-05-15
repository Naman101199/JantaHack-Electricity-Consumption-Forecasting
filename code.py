import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization


train = pd.read_csv("train_6BJx641.csv")

test = pd.read_csv('test_pavJagI.csv')


#For Creating Lagged variable

combined = [train,test]
combined = pd.concat(combined)

combined = combined.sort_values(by = ['datetime'])

df = combined.copy()

df['datetime'] = pd.to_datetime(df['datetime'])

df['day'] = df['datetime'].apply(lambda x: x.day)
df['month'] = df['datetime'].apply(lambda x: x.month)
df['year'] = df['datetime'].apply(lambda x: x.year)
df['hour'] = df['datetime'].apply(lambda x: x.hour)


df = df.drop('datetime',axis = 1)

var2 = pd.get_dummies(df['var2'])

df = pd.merge(df,var2,on = df.ID)

df = df.drop(['var2','key_0','ID'],axis = 1)


new_df = df.copy()


for day in range(1,8):
 
 new_df['var1_day_' + str(day)] = new_df.var1.shift(24*day)
 
 new_df['windspeed_day_' + str(day)] = new_df.windspeed.shift(24*day)
 
 new_df['temp_day_' + str(day)] = new_df.temperature.shift(24*day)
 
 new_df['pressure_day_' + str(day)] = new_df.pressure.shift(24*day)   


for hour in range(1,25):
 new_df['var1_' + str(hour)] = new_df.var1.shift(hour)
 
 new_df['windspeed_' + str(hour)] = new_df.windspeed.shift(hour)
 
 new_df['temp_' + str(hour)] = new_df.temperature.shift(hour)
 
 new_df['pressure_' + str(hour)] = new_df.pressure.shift(hour)   


#Train-test split

new_df_train = new_df.dropna()
new_df_test  = new_df[new_df['electricity_consumption'].isnull() == True].drop('electricity_consumption',axis = 1)

x_train = new_df_train.drop('electricity_consumption',axis = 1).values
y_train = new_df_train['electricity_consumption'].values

model = Sequential()
model.add(Dense(512, input_dim=135, kernel_initializer='normal', activation='relu'))
model.add(Dense(256, kernel_initializer='normal', activation='relu'))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(x_train, y_train, batch_size = 96, epochs = 100, verbose = 1)

y_pred = model.predict(new_df_test.values).round()

submission = pd.read_csv('sample_submission_bYgKb77.csv')
submission['electricity_consumption'] = y_pred
submission.to_csv('submission.csv',index_label = False)
