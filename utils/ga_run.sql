-- MySQL dump 10.13  Distrib 5.7.30, for Linux (x86_64)
--
-- Host: localhost    Database: timeseries
-- ------------------------------------------------------
-- Server version	5.7.30-0ubuntu0.18.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `MachineLearning_econometric_predictions`
--

DROP TABLE IF EXISTS `MachineLearning_econometric_predictions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `MachineLearning_econometric_predictions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `future_date` date DEFAULT NULL,
  `inx` decimal(10,4) NOT NULL,
  `forecasted_series_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `MachineLearning_econ_forecasted_series_id_0dfbd327_fk_MachineLe` (`forecasted_series_id`),
  CONSTRAINT `MachineLearning_econ_forecasted_series_id_0dfbd327_fk_MachineLe` FOREIGN KEY (`forecasted_series_id`) REFERENCES `MachineLearning_econometric_forecast` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=251305 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `MachineLearning_econometric_predictions`
--

LOCK TABLES `MachineLearning_econometric_predictions` WRITE;
/*!40000 ALTER TABLE `MachineLearning_econometric_predictions` DISABLE KEYS */;
INSERT INTO `MachineLearning_econometric_predictions` VALUES (251125,'2020-01-01',137.1180,20980),(251126,'2020-02-01',137.1133,20980),(251127,'2020-03-01',137.1766,20980),(251128,'2020-04-01',137.2066,20980),(251129,'2020-05-01',136.8333,20980),(251130,'2020-06-01',136.1500,20980),(251131,'2020-07-01',135.9547,20980),(251132,'2020-08-01',136.0264,20980),(251133,'2020-09-01',136.2670,20980),(251134,'2020-10-01',136.6836,20980),(251135,'2020-11-01',136.8899,20980),(251136,'2020-12-01',136.9530,20980),(251137,'2020-01-01',118.3638,20981),(251138,'2020-02-01',118.4565,20981),(251139,'2020-03-01',118.5693,20981),(251140,'2020-04-01',118.7235,20981),(251141,'2020-05-01',118.8146,20981),(251142,'2020-06-01',118.9081,20981),(251143,'2020-07-01',119.0059,20981),(251144,'2020-08-01',119.0899,20981),(251145,'2020-09-01',119.1846,20981),(251146,'2020-10-01',119.2796,20981),(251147,'2020-11-01',119.3735,20981),(251148,'2020-12-01',119.4677,20981),(251149,'2020-01-01',186.5228,20982),(251150,'2020-02-01',187.0731,20982),(251151,'2020-03-01',187.5136,20982),(251152,'2020-04-01',187.9017,20982),(251153,'2020-05-01',188.1033,20982),(251154,'2020-06-01',188.2979,20982),(251155,'2020-07-01',188.5733,20982),(251156,'2020-08-01',188.7220,20982),(251157,'2020-09-01',188.8777,20982),(251158,'2020-10-01',189.1726,20982),(251159,'2020-11-01',189.4157,20982),(251160,'2020-12-01',189.7243,20982),(251161,'2020-01-01',92.8537,20983),(251162,'2020-02-01',93.1356,20983),(251163,'2020-03-01',93.8132,20983),(251164,'2020-04-01',94.3372,20983),(251165,'2020-05-01',94.8577,20983),(251166,'2020-06-01',95.5978,20983),(251167,'2020-07-01',96.1350,20983),(251168,'2020-08-01',96.8384,20983),(251169,'2020-09-01',97.3499,20983),(251170,'2020-10-01',98.0437,20983),(251171,'2020-11-01',98.7115,20983),(251172,'2020-12-01',99.2005,20983),(251173,'2020-01-01',230.3363,20984),(251174,'2020-02-01',230.0211,20984),(251175,'2020-03-01',229.6795,20984),(251176,'2020-04-01',229.3179,20984),(251177,'2020-05-01',229.0340,20984),(251178,'2020-06-01',228.7917,20984),(251179,'2020-07-01',228.4808,20984),(251180,'2020-08-01',228.2513,20984),(251181,'2020-09-01',227.9636,20984),(251182,'2020-10-01',227.7199,20984),(251183,'2020-11-01',227.4612,20984),(251184,'2020-12-01',227.2172,20984),(251185,'2020-01-01',4.7603,20985),(251186,'2020-02-01',4.7612,20985),(251187,'2020-03-01',4.7638,20985),(251188,'2020-04-01',4.7713,20985),(251189,'2020-05-01',4.7721,20985),(251190,'2020-06-01',4.7691,20985),(251191,'2020-07-01',4.7691,20985),(251192,'2020-08-01',4.7680,20985),(251193,'2020-09-01',4.7713,20985),(251194,'2020-10-01',4.7768,20985),(251195,'2020-11-01',4.7753,20985),(251196,'2020-12-01',4.7718,20985),(251197,'2020-01-01',106.7360,20986),(251198,'2020-02-01',106.8937,20986),(251199,'2020-03-01',107.0184,20986),(251200,'2020-04-01',107.1695,20986),(251201,'2020-05-01',107.3288,20986),(251202,'2020-06-01',107.4958,20986),(251203,'2020-07-01',107.6768,20986),(251204,'2020-08-01',107.8434,20986),(251205,'2020-09-01',108.0158,20986),(251206,'2020-10-01',108.1423,20986),(251207,'2020-11-01',108.3106,20986),(251208,'2020-12-01',108.4735,20986),(251209,'2020-01-01',123.8244,20987),(251210,'2020-02-01',124.0231,20987),(251211,'2020-03-01',124.2184,20987),(251212,'2020-04-01',124.4136,20987),(251213,'2020-05-01',124.6080,20987),(251214,'2020-06-01',124.8027,20987),(251215,'2020-07-01',124.9971,20987),(251216,'2020-08-01',125.1913,20987),(251217,'2020-09-01',125.3857,20987),(251218,'2020-10-01',125.5795,20987),(251219,'2020-11-01',125.7742,20987),(251220,'2020-12-01',125.9684,20987),(251221,'2020-01-01',215.1751,20988),(251222,'2020-02-01',215.4951,20988),(251223,'2020-03-01',215.7809,20988),(251224,'2020-04-01',216.0709,20988),(251225,'2020-05-01',216.3566,20988),(251226,'2020-06-01',216.6529,20988),(251227,'2020-07-01',216.9481,20988),(251228,'2020-08-01',217.2569,20988),(251229,'2020-09-01',217.5568,20988),(251230,'2020-10-01',217.8750,20988),(251231,'2020-11-01',218.1800,20988),(251232,'2020-12-01',218.4835,20988),(251233,'2020-01-01',96.8654,20989),(251234,'2020-02-01',96.8949,20989),(251235,'2020-03-01',96.8107,20989),(251236,'2020-04-01',96.7434,20989),(251237,'2020-05-01',96.6641,20989),(251238,'2020-06-01',96.6494,20989),(251239,'2020-07-01',96.6374,20989),(251240,'2020-08-01',96.6124,20989),(251241,'2020-09-01',96.5388,20989),(251242,'2020-10-01',96.4854,20989),(251243,'2020-11-01',96.4701,20989),(251244,'2020-12-01',96.4612,20989),(251245,'2020-01-01',115.8369,20990),(251246,'2020-02-01',115.8660,20990),(251247,'2020-03-01',115.8976,20990),(251248,'2020-04-01',115.9295,20990),(251249,'2020-05-01',115.9596,20990),(251250,'2020-06-01',115.9921,20990),(251251,'2020-07-01',116.0246,20990),(251252,'2020-08-01',116.0573,20990),(251253,'2020-09-01',116.0900,20990),(251254,'2020-10-01',116.1229,20990),(251255,'2020-11-01',116.1518,20990),(251256,'2020-12-01',116.1847,20990),(251257,'2020-01-01',160.2276,20991),(251258,'2020-02-01',159.3479,20991),(251259,'2020-03-01',158.7710,20991),(251260,'2020-04-01',158.0685,20991),(251261,'2020-05-01',157.6673,20991),(251262,'2020-06-01',157.1522,20991),(251263,'2020-07-01',156.7112,20991),(251264,'2020-08-01',155.9252,20991),(251265,'2020-09-01',155.5033,20991),(251266,'2020-10-01',154.8398,20991),(251267,'2020-11-01',154.1294,20991),(251268,'2020-12-01',153.3369,20991),(251269,'2020-01-01',85.1199,20992),(251270,'2020-02-01',85.0717,20992),(251271,'2020-03-01',85.1928,20992),(251272,'2020-04-01',85.5578,20992),(251273,'2020-05-01',85.3617,20992),(251274,'2020-06-01',85.4497,20992),(251275,'2020-07-01',85.5864,20992),(251276,'2020-08-01',85.7780,20992),(251277,'2020-09-01',85.6666,20992),(251278,'2020-10-01',85.7513,20992),(251279,'2020-11-01',85.8937,20992),(251280,'2020-12-01',85.8273,20992),(251281,'2020-01-01',118.2973,20993),(251282,'2020-02-01',118.0602,20993),(251283,'2020-03-01',118.3167,20993),(251284,'2020-04-01',118.3986,20993),(251285,'2020-05-01',118.5503,20993),(251286,'2020-06-01',118.5367,20993),(251287,'2020-07-01',118.7953,20993),(251288,'2020-08-01',118.7618,20993),(251289,'2020-09-01',118.6007,20993),(251290,'2020-10-01',118.6411,20993),(251291,'2020-11-01',119.0389,20993),(251292,'2020-12-01',118.8700,20993),(251293,'2020-01-01',546.0477,20994),(251294,'2020-02-01',548.4601,20994),(251295,'2020-03-01',548.2305,20994),(251296,'2020-04-01',550.0618,20994),(251297,'2020-05-01',553.5509,20994),(251298,'2020-06-01',551.8413,20994),(251299,'2020-07-01',551.5209,20994),(251300,'2020-08-01',551.7541,20994),(251301,'2020-09-01',561.1337,20994),(251302,'2020-10-01',563.0938,20994),(251303,'2020-11-01',565.9474,20994),(251304,'2020-12-01',566.8840,20994);
/*!40000 ALTER TABLE `MachineLearning_econometric_predictions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `MachineLearning_econometric_coefficients`
--

DROP TABLE IF EXISTS `MachineLearning_econometric_coefficients`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `MachineLearning_econometric_coefficients` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `independent_variables` varchar(100) DEFAULT NULL,
  `coefficients` decimal(10,4) NOT NULL,
  `pvalues` decimal(10,4) NOT NULL,
  `forecasted_series_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `MachineLearning_econ_forecasted_series_id_146fc4e1_fk_MachineLe` (`forecasted_series_id`),
  KEY `MachineLearning_econometric_independent_variables_5c0df31e` (`independent_variables`),
  CONSTRAINT `MachineLearning_econ_forecasted_series_id_146fc4e1_fk_MachineLe` FOREIGN KEY (`forecasted_series_id`) REFERENCES `MachineLearning_econometric_forecast` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=226792 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `MachineLearning_econometric_coefficients`
--

LOCK TABLES `MachineLearning_econometric_coefficients` WRITE;
/*!40000 ALTER TABLE `MachineLearning_econometric_coefficients` DISABLE KEYS */;
INSERT INTO `MachineLearning_econometric_coefficients` VALUES (226727,'const',191.1835,0.0000,20980),(226728,'Electricity in Size Class B/C all urban consumers not seasonally adjusted',-0.1982,0.0550,20980),(226729,'Utility (piped) gas per therm in Size Class A average price not seasonally adjusted',-20.6101,0.0650,20980),(226730,'const',12.9769,0.1750,20981),(226731,'Gasoline unleaded midgrade in South urban all urban consumers not seasonally adjusted',-0.0068,0.0940,20981),(226732,'PPI Commodity data for Machinery and equipment-Off-highway equipment ex. parts not seasonally adjust',-0.0244,0.2700,20981),(226733,'PPI industry data for Ethyl alcohol manufacturing-Secondary products not seasonally adjusted',0.0046,0.5540,20981),(226734,'PPI industry data for Fan blower air purification equipment mfg-Other centrifugal fans and blowers e',0.0725,0.0060,20981),(226735,'PPI industry data for Other measuring and controlling device mfg-Physical properties testing & inspe',0.6246,0.0000,20981),(226736,'const',30.9287,0.0160,20982),(226737,'Gasoline unleaded midgrade per gallon/3.785 liters in Miami-Fort Lauderdale-West Palm Beach FL avera',39.9516,0.0000,20982),(226738,'Monthly import price index for NAICS 1113 Fruit and tree nut farming not seasonally adjusted',0.1094,0.0000,20982),(226739,'PPI Commodity data for Intermediate demand by commodity type-Unprocessed materials less energy not s',0.0982,0.0430,20982),(226740,'const',-106.0180,0.0000,20983),(226741,'Inpatient hospital services in U.S. city average all urban consumers not seasonally adjusted',0.1288,0.2100,20983),(226742,'Monthly import price index for NAICS 325 Chemical manufacturing not seasonally adjusted',1.8494,0.0000,20983),(226743,'PPI Commodity data for Telecommunication cable and internet user services not seasonally adjusted',-0.9782,0.0090,20983),(226744,'const',232.6330,0.0050,20984),(226745,'Gasoline all types per gallon/3.785 liters in Atlanta-Sandy Springs-Roswell GA average price not sea',12.5362,0.0190,20984),(226746,'PPI Commodity data for Processed foods and feeds-Dairy cattle feed supplements concentrates and prem',1.5313,0.0120,20984),(226747,'PPI Commodity data for Processed foods and feeds-Pork fresh/frozen unprocessed all cuts except sausa',0.2499,0.0870,20984),(226748,'PPI industry data for Other nonferrous metal foundries except die-casting-Copper foundries (excludin',-0.6670,0.0310,20984),(226749,'const',4.2708,0.0000,20985),(226750,'Motor fuel in Los Angeles-Long Beach-Anaheim CA all urban consumers not seasonally adjusted',0.0018,0.0480,20985),(226751,'const',1.0663,0.0980,20986),(226752,'Commodities in Los Angeles-Long Beach-Anaheim CA all urban consumers not seasonally adjusted',-0.0047,0.3470,20986),(226753,'PPI industry data for Paper bag and coated and treated paper mfg not seasonally adjusted',0.7600,0.0000,20986),(226754,'Utility (piped) gas service in Baltimore-Columbia-Towson MD all urban consumers not seasonally adjus',-0.0011,0.1690,20986),(226755,'const',-2.0747,0.0090,20987),(226756,'PPI Commodity data for Chemicals and allied products-Other chemical preparations n.e.c. not seasonal',0.0012,0.7110,20987),(226757,'PPI Commodity data for Final demand-Finished distributive services seasonally adjusted',0.0237,0.0000,20987),(226758,'PPI industry data for Other millwork including flooring-Hardwood flooring other than oak and maple n',0.8610,0.0000,20987),(226759,'const',-86.6695,0.0160,20988),(226760,'PPI Commodity data for Metals and metal products-Pressure and soil pipe & fittings cast iron not sea',0.0214,0.8370,20988),(226761,'PPI Commodity data for Nonmetallic mineral products-Rotary oil seals not seasonally adjusted',1.7490,0.0020,20988),(226762,'PPI industry data for Metal tank heavy gauge mfg-Primary products not seasonally adjusted',0.3321,0.0250,20988),(226763,'const',68.2307,0.0040,20989),(226764,'Distilled spirits excluding whiskey at home in U.S. city average all urban consumers seasonally adju',0.1872,0.0420,20989),(226765,'PPI industry data for Asphalt paving mixture & block manufacturing-Emulsified asphalt including liqu',0.0444,0.0030,20989),(226766,'Salad dressing in U.S. city average all urban consumers seasonally adjusted',-0.1205,0.2900,20989),(226767,'const',98.3211,0.0000,20990),(226768,'PPI industry data for Mining machinery and equipment mfg-Parts and attachments for mining machinery',-0.0538,0.0510,20990),(226769,'PPI industry data for Pharmaceutical preparation manufacturing-Other cardiovascular preparations not',0.0158,0.0000,20990),(226770,'PPI industry data for U.S. Postal Service-Standard class mail not seasonally adjusted',0.0911,0.0380,20990),(226771,'const',-2.5851,0.0210,20991),(226772,'PPI Commodity data for Transportation equipment-All other miscellaneous complete engine electrical e',0.0153,0.0960,20991),(226773,'PPI industry data for Beet sugar manufacturing-Primary products not seasonally adjusted',1.0155,0.0000,20991),(226774,'PPI industry data for Other heavy machinery rental and leasing-Construction equipment rental and lea',0.0110,0.0050,20991),(226775,'const',214.0709,0.0020,20992),(226776,'Cheese and related products in U.S. city average all urban consumers seasonally adjusted',-0.3284,0.0360,20992),(226777,'Distilled spirits at home in U.S. city average all urban consumers seasonally adjusted',0.5124,0.0140,20992),(226778,'Monthly export price index for BEA End Use 12750 Industrial rubber products not seasonally adjusted',-0.9484,0.0000,20992),(226779,'PPI Commodity data for Intermediate demand by production flow-Inputs to stage 2 goods producers seas',0.1594,0.0020,20992),(226780,'PPI industry data for Cheese manufacturing-Primary products not seasonally adjusted',-0.1154,0.0240,20992),(226781,'Salad dressing in U.S. city average all urban consumers not seasonally adjusted',-0.3834,0.0010,20992),(226782,'const',-2.8274,0.0150,20993),(226783,'Fresh fruits and vegetables in U.S. city average all urban consumers seasonally adjusted',0.0064,0.0100,20993),(226784,'PPI industry data for Commercial bakeries-Rolls hamburger and wiener not seasonally adjusted',0.9949,0.0000,20993),(226785,'PPI industry data for Copper nickel lead and zinc mining not seasonally adjusted',0.0002,0.3420,20993),(226786,'Wireless telephone services in U.S. city average all urban consumers not seasonally adjusted',0.0172,0.0290,20993),(226787,'const',291.4981,0.4620,20994),(226788,'Monthly import price index for Harmonized 0901 Coffee whether or not roasted or decaffeinated; coffe',1.3956,0.2240,20994),(226789,'PPI Commodity data for Metals and metal products-No. 1 copper scrap including wire not seasonally ad',1.6267,0.0000,20994),(226790,'PPI industry data for All other plastics product manufacturing-All other reinforced and fiberglass p',1.5581,0.3260,20994),(226791,'Transportation commodities less motor fuel in U.S. city average all urban consumers not seasonally a',-7.2789,0.0220,20994);
/*!40000 ALTER TABLE `MachineLearning_econometric_coefficients` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `MachineLearning_econometric_forecast`
--

DROP TABLE IF EXISTS `MachineLearning_econometric_forecast`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `MachineLearning_econometric_forecast` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `series_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `MachineLearning_econ_series_id_194ab247_fk_MachineLe` (`series_id`),
  CONSTRAINT `MachineLearning_econ_series_id_194ab247_fk_MachineLe` FOREIGN KEY (`series_id`) REFERENCES `MachineLearning_series_names` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=20995 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `MachineLearning_econometric_forecast`
--

LOCK TABLES `MachineLearning_econometric_forecast` WRITE;
/*!40000 ALTER TABLE `MachineLearning_econometric_forecast` DISABLE KEYS */;
INSERT INTO `MachineLearning_econometric_forecast` VALUES (20980,473),(20984,522),(20982,592),(20985,1183),(20989,1697),(20983,1782),(20987,3177),(20994,3535),(20993,3952),(20991,4564),(20992,5359),(20981,5361),(20986,6197),(20990,6866),(20988,6996);
/*!40000 ALTER TABLE `MachineLearning_econometric_forecast` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-10 23:36:13
