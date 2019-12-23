import knoema
import MySQLdb
import MySQLdb.cursors

apicfg=knoema.ApiConfig()
apicfg.host='knoema.com'
apicfg.app_id='BlH9ZAI'
apicfg.app_secret='s2r0J8s9xiOYeQ'

df=knoema.get('WHTDRCOP2019Nov', timerange='2009M1-2019M1', frequency='Q', Indicator='IND.04;IND.03;IND.19;IND.21;IND.27;IND.28;IND.40;IND.24;IND.23;IND.42;IND.53;IND.06;IND.29;IND.54')

db=MySQLdb.connect(host="localhost", port=3306, user="root", passwd="a6!modern", db="timeseries")
cursor=db.cursor()

for i in range(len(df)):
    sql="INSERT INTO machinelearning_crude (frequency, wti_real_price,non_opec_liquid_fuels_production,saudi_arabia_crude_oil_production_change,opec_spare_production_capacity,non_oecd_liquid_fuels_consumption_change,non_oecd_gdp_growth,oecd_liquid_fuels_consumption_change,wti_crude_12_1_futures_price_spread_change,oecd_liquid_fuels_inventories_change,avg_num_outstanding_oil_futures_contract,assets_under_management,world_gdp_growth,world_liquid_fuels_consumption_change,dow_jones_ubs_commodity_index) VALUES ('" + str(df.index[i]).replace("00:00:00","") + "'," + str(df.iloc[i,0]) + "," + str(df.iloc[i,1]) + "," + str(df.iloc[i,2]) + "," + str(df.iloc[i,3]) + "," + str(df.iloc[i,4]) + "," + str(df.iloc[i,5]) + "," + str(df.iloc[i,6]) + "," + str(df.iloc[i,7]) + "," + str(df.iloc[i,8]) + "," + str(df.iloc[i,9]) + "," + str(df.iloc[i,10]) + "," + str(df.iloc[i,11]) + "," + str(df.iloc[i,12]) + "," + str(df.iloc[i,13]) + ");"
    cursor.execute(sql)
    
#try:    
    
db.commit()
cursor.close()
db.close()    
#except:
#    print(sql)