import json
import os
import joblib
import numpy as np

from google.cloud import storage
from google.oauth2 import service_account
from termcolor import colored
from TaxiFareModel.params import BUCKET_NAME, PROJECT_ID, MODEL_NAME, MODEL_VERSION


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

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_version,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print(f"=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
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
