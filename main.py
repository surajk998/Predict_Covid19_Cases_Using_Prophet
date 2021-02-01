import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot importplot_cross_validation_metric

'''Reading CSV file from covid19 clean complete file location'''
df=pd.read_csv(r':/file_location/covid_19_clean_complete.csv')

# print(df.head())
# print(df.shape)
# print(df.dtypes)

'''Converting df['Date'] type from object to date time'''
df['Date']=pd.to_datetime(df['Date'])

# print(df.dtypes)
# print(df.isnull().sum())
# print(df['Date'].nunique())
# print(df.shape)

'''Grouping the covid19 cases(confirmed,active,deaths,recovered)'''
total=df.groupby(['Date'])['Confirmed','Deaths','Active','Recovered'].sum().reset_index()

# print(total.head())

'''Renaming the columns as 'ds' and 'y' since the documentation says default columns names'''
df_prophet=total.rename(columns={'Date':'ds','Confirmed':'y'})

'''initializing prophet model'''
m=Prophet()

'''fitting the data into the model'''
model=m.fit(df_prophet)
print(model.seasonalities)

'''taking a 30 days future prediction with current model'''
future_global=model.make_future_dataframe(periods=30,freq='D')
# print(future_global.head())
# print(future_global.shape)

'''predicting the future 30days data'''
prediction = model.predict(future_global)
# print(prediction)
print(prediction[['ds','yhat','yhat_lower','yhat_upper']].tail())

'''plotting the predicted model graph(make as exponential)'''
model.plot(prediction)
model.plot_components(prediction)

fig=model.plot(prediction)
a=add_changepoints_to_plot(fig.gca(),model,prediction)

'''checking the cross validation of the predicted model'''
df_cv=cross_validation(model,horizon='30 days',period='15 days',initial='90 days')
# print(df_cv.head())
# print(df_cv.shape)

'''checking the performance metrics of the cross validated predicted model'''
df_performance = performance_metrics(df_cv)
# print(df_performance.head())

df_performance = plot_cross_validation_metric(df_cv,metric='rmse')
df_performance = plot_cross_validation_metric(df_cv,metric='mse')
df_performance = plot_cross_validation_metric(df_cv,metric='mae')
df_performance = plot_cross_validation_metric(df_cv,metric='mape')
df_performance = plot_cross_validation_metric(df_cv,metric='mdape')


















