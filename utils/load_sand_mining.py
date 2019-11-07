import MySQLdb
import MySQLdb.cursors
import pandas as pd
import numpy as np

df=pd.read_excel('C:/Users/light/Desktop/Django_Projects/Forecasts/utils/PCU21232221232291.xlsx')

db=MySQLdb.connect(host="localhost", port=3306, user="root", passwd="a6!modern", db="timeseries")
cursor=db.cursor()
for index,row in df.iterrows():
    sql1="INSERT INTO machinelearning_sand_mining (observation_date, inx) VALUES ('" + str(row['observation_date']).replace(" 00:00:00","") + "', " + str(row['ppi']) + ");"
    sql1="INSERT INTO machinelearning_sand_mining (observation_date, inx) VALUES ('" + str(row['observation_date']).replace(" 00:00:00","") + "', " + str(row['ppi']) + ");"
    cursor.execute(sql1)
        
db.commit()
cursor.close()
db.close()        