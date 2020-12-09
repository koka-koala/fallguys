import json
import os
import joblib
import numpy as np
import tensorflow as tf

from google.cloud import storage
from google.oauth2 import service_account
from fallguys.params import BUCKET_NAME, PROJECT_ID,\
    MODEL_NAME, MODEL_VERSION, BUCKET_X_DATA_PATH, BUCKET_y_DATA_PATH

# PROJECT_ID = "wagon509-jin"
# BUCKET_NAME = 'fall-guys-project'
# MODEL_NAME = 'model'
# MODEL_VERSION = 'model_1208_recall100_83'
# BUCKET_X_DATA_PATH = 'data/X_test.npy'
# BUCKET_y_DATA_PATH = 'data/y_test.npy'

def get_credentials():
    credentials_raw = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if '.json' in credentials_raw:
        credentials_raw = open(credentials_raw).read()
    creds_json = json.loads(credentials_raw)
    creds_gcp = service_account.Credentials.from_service_account_info(creds_json)
    return creds_gcp


def download_model(model_version=MODEL_VERSION, bucket=BUCKET_NAME, rm=True):
    creds = get_credentials()
    client = storage.Client(credentials=creds, project=PROJECT_ID).bucket(bucket)

    storage_location = '{}/{}/{}'.format(
        MODEL_NAME,
        model_version,
        "saved_model.pb")
    blob = client.blob(storage_location)
    blob.download_to_filename('model')
    print(f"=> model downloaded from storage")
    model = tf.keras.models.load_model('model')
    if rm:
        os.remove('model')
    return model


def download_Xdata(data=BUCKET_X_DATA_PATH, bucket=BUCKET_NAME, rm=True):
    creds = get_credentials()
    client = storage.Client(credentials=creds, project=PROJECT_ID).bucket(bucket)

    storage_location = data
    blob = client.blob(storage_location)
    blob.download_to_filename('X_test.npy')
    print(f"=> X_data downloaded from storage")
    X_data = np.load('X_test.npy')
    if rm:
        os.remove('X_test.npy')
    return X_data


def download_ydata(data=BUCKET_y_DATA_PATH, bucket=BUCKET_NAME, rm=True):
    creds = get_credentials()
    client = storage.Client(credentials=creds, project=PROJECT_ID).bucket(bucket)

    storage_location = data
    blob = client.blob(storage_location)
    blob.download_to_filename('y_test.npy')
    print(f"=> y_data downloaded from storage")
    y_data = np.load('y_test.npy')
    if rm:
        os.remove('y_test.npy')
    return y_data

print("Start loading data X")
X = download_Xdata()
print(X.shape)
print("Start loading data y")
y = download_ydata()
print(y.shape)
