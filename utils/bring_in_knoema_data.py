import knoema
import MySQLdb
import MySQLdb.cursors

df=knoema.get('WHTDRCOP2019Nov', timerange='2009M1-2019M1', frequency='Q', Indicator='IND.05;IND.04;IND.42;IND.53;IND.06;IND.29')

db=MySQLdb.connect(host="localhost", port=3306, user="root", passwd="a6!modern", db="timeseries")
cursor=db.cursor()

for i in range(len(df)):
    sql="INSERT INTO machinelearning_crude (frequency, wti_real_price, world_liquid_fuels_production_capacity_change, avg_num_outstanding_oil_futures_contract, assets_under_management, world_gdp_growth, world_liquid_fuels_consumption_change) VALUES ('" + str(df.index[i]).replace("00:00:00","") + "'," + str(df.iloc[i,1]) + "," + str(df.iloc[i,0]) + "," + str(df.iloc[i,2]) + "," + str(df.iloc[i,5]) + "," + str(df.iloc[i,4]) + "," + str(df.iloc[i,3]) + ");"
    cursor.execute(sql)
    
db.commit()
cursor.close()
db.close()    
    