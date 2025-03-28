import json
import os
import time
import pandas as pd
import schedule
from mdclogpy import Logger, Level
from ad_model import modelling, CAUSE
from ad_train import ModelTraining
from database import DATABASE, DUMMY
from datetime import datetime

db = None
cp = None
threshold = None

logger = Logger(name=__name__, level=Level.DEBUG)

def load_model():
    global md
    global cp
    global threshold
    md = modelling()
    cp = CAUSE()
    threshold = 70
    logger.info("throughput threshold parameter is set as {}% (default)".format(threshold))


def train_model():
    if not os.path.isfile('src/model'):
        mt = ModelTraining(db)
        mt.train()


def predict():
    """Read the latest ue sample from influxDB and detects if that is anomalous or normal..
      Send the UEID, DUID, Degradation type and timestamp for the anomalous samples to Traffic Steering (rmr with the message type as 30003)
      Get the acknowledgement of sent message from the traffic steering.
    """
    db.read_data()
    val = None
    if db.data is not None:
        if set(md.num).issubset(db.data.columns):
            db.data = db.data.dropna(axis=0)
            if len(db.data) > 0:
                val = predict_anomaly(db.data)
        else:
            logger.warning("Parameters does not match with of training data")
    else:
        logger.warning("No data in last 1 second")
        time.sleep(1)
    if (val is not None) and (len(val) > 2):
        print("Anomaly detected!")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"src/anomaly_{timestamp}"

        with open(filename + ".json", "w") as file:
            json.dump(json.loads(val.decode('utf-8')), file, indent=4)

        # msg_to_ts(self, val)


def predict_anomaly(df):
    """ calls ad_predict to detect if given sample is normal or anomalous
    find out the degradation type if sample is anomalous
    write given sample along with predicted label to AD measurement

    Parameter
    ........
    ue: array or dataframe

    Return
    ......
    val: anomalus sample info(UEID, DUID, TimeStamp, Degradation type)
    """
    df['Anomaly'] = md.predict(df)
    df.loc[:, 'Degradation'] = ''
    val = None
    if 1 in df.Anomaly.unique():
        df.loc[:, ['Anomaly', 'Degradation']] = cp.cause(df, db, threshold)
        df_a = df.loc[df['Anomaly'] == 1].copy()
        if len(df_a) > 0:
            df_a['time'] = df_a.index
            cols = [db.ue, 'time', 'Degradation']
            # rmr send 30003(TS_ANOMALY_UPDATE), should trigger registered callback
            result = json.loads(df_a.loc[:, cols].to_json(orient='records'))
            val = json.dumps(result).encode()
    df.loc[:, 'RRU.PrbUsedDl'] = df['RRU.PrbUsedDl'].astype('float')
    df.index = pd.date_range(start=df.index[0], periods=len(df), freq='1ms')
    db.write_anomaly(df)
    return val


def connectdb(thread=False):
    # Create a connection to InfluxDB if thread=True, otherwise it will create a dummy data instance
    global db
    if thread:
        db = DUMMY()
    else:
        db = DATABASE()
    success = False
    while not success:
        success = db.connect()

def main():
    connectdb()
    train_model()
    load_model()
    schedule.every(0.5).seconds.do(predict)
    while True:
        schedule.run_pending()

if __name__ == "__main__":
    logger.info("Starting application")
    main()