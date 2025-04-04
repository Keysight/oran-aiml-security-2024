# Isolated-ad-model

This folder contains isolated ADS model for O-RAN available under "https://gerrit.o-ran-sc.org/r/ric-app/ad"

The main difference is that isolated from xApp framework for Near-RT RIC. This way it does not require the Near-RT RIC deployment to work with.

# Installation

There are two ways of using it. With InfluxDB and without (Not tested). The latter option imports the database 'ue.csv' directly via Pandas library in database.py file. 

## InfluxDB option
To use it with InfluxDB one must first deploy a docker container with InfluxDB defined in compose.yaml file. One may also use Graphana to simplyfly accessing the InfluxDB from the browser, although this is optional.

After the container with InfluxDB is deployed, the config.ini file contains all the credentials and configuration. Some of them are hardcoded in database.py file as default values. This code is not intended to use in an unsecure enviroment. 

The ADS by default hardcodes path to files in src/ directory. You may need to create it. 

To populate the InfluxDB with data use insert.py file. Make sure to have 'ue.csv' in the same directory or modify the hardcoded path accordingly. Make sure the InfluxDB is running and reachable.

To run ADS use main.py, while insert.py is running in the background. You may need to wait a few minutes for insert.py to populate sufficient amount of data for training.

The detection anomalies will be dumped in ADS folder. The communication with Traffic Steering (TS) has been removed as it requires Near-RT RIC deployment (precisely RMR procotol)

 ## Remarks
- Note that this version of ADS does not work with InfluxDB 2.0
- Note that standard graphana port was switched from 8888 to 8889 to avoid port collision with JupyterLab.
- Note that to work with Graphana user may need to enter credentials to the InfluxDB database in Graphana's GUI in browser.