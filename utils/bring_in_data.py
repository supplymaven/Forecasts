import quandl
import MySQLdb
import MySQLdb.cursors

quandl.ApiConfig.api_key='d-LyarSr8s9xzyQ7qoqG'

# data source: https://www.quandl.com/search?filters=%5B%22Indexes%22%5D

data_categories=['ZILLOW/M1300_MPPRSF', 'NASDAQOMX/XQC', 'YALE/RBCI', 'MULTPL/SP500_DIV_YIELD_MONTH', 'ML/EMHYY', 'RICI/RICI']
tables=['machinelearning_zillow', 'machinelearning_nasdaq', 'machinelearning_yale', 'machinelearning_sp_ratios', 'machinelearning_corporate_bond_yield_rates', 'machinelearning_commodity_indices']

db=MySQLdb.connect(host="localhost", port=3306, user="root", passwd="a6!modern", db="timeseries")
cursor=db.cursor()
for j in range(len(data_categories)):
    data=quandl.get(data_categories[j])
    for i in range(len(data)):
        sql="INSERT IGNORE INTO " + tables[j] + " (observation_date, inx) VALUES ('" + str(data.index[i]).replace("00:00:00","") + "', " + str(data.iloc[i,0]) + ");"
        try:
            if data.iloc[i,0]!=0:
                cursor.execute(sql)
        except:
            pass
            
db.commit()
cursor.close()
db.close()        





