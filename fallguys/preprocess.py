""" Preprocess code for fallguys Project
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os

def an_filename(name):
    '''Get the real file name not the path from csv file'''
    filename = "a_" + name.split("a_")[1]
    return filename


def sen_filename(name):
    '''Get the real file name not the path from csv file'''
    filename = "s_" + name.split("s_")[1]
    return filename


def process_fall(path_anno, path_sensor):
    """ create npy model
    """
    def check_between(time_stamp):
    for i in range(fall_anclean1.shape[0]):
        if fall_anclean1.loc[i,"start"] < time_stamp and time_stamp < fall_anclean1.loc[i,"end"]:
            return fall_anclean1.loc[i,"activitiy"] + " " + str(fall_anclean1.loc[i,"batch"])
    return None

    def key(X):
        key, value = list(X.items())[0]
        return key.strip("()")


    def value(X):
        key, value = list(X.items())[0]
        key2, value2 = list(value[0].items())[0]
        return value2


    def process(X):
        key, value = list(X.items())[0]
        key2, value2 = list(value[0].items())[0]
        return key2

    #annotation
    fall_an_1 = pd.read_json(path_anno)
    fall_an_1["activities"] = fall_an_1["dt"].apply(key)
    fall_an_1["process"] = fall_an_1["dt"].apply(process)
    fall_an_1["time"] = fall_an_1["dt"].apply(value)
    fall_an_1 = fall_an_1[fall_an_1.columns[-3:]]
    # create df
    row = fall_an_1.shape[0]
    n = 0
    list_begin = []
    list_end = []
    list_act = []
    for row in range(int(row/2)):
        list_begin.append(fall_an_1["time"][n])
        list_act.append(fall_an_1["activities"][n])
        list_end.append(fall_an_1["time"][n+1])
        n +=2
    fall_an_dict1 = {"activitiy" : list_act, "start" : list_begin, "end" : list_end}
    fall_an_1 = pd.DataFrame(fall_an_dict1)
    batch = []
    batch_no = 0
    start = []
    end = []
    activitiy = []
    for row in range(fall_an_1.shape[0]):
        start_an = fall_an_1.loc[row, "start"]
        end_an = start_an + 2000
        while end_an < fall_an_1.loc[row, "end"]:
            start.append(start_an)
            end.append(end_an)
            activitiy.append(fall_an_1.loc[row, "activitiy"])
            batch.append(batch_no)
            batch_no += 1
            start_an += 2000
            end_an += 2000
        if fall_an_1.loc[row, "activitiy"] == "fall":
            start.append(start_an)
            end.append(fall_an_1.loc[row, "end"])
            activitiy.append(fall_an_1.loc[row, "activitiy"])
            batch.append(batch_no)
            batch_no += 1
    fall_an_dict1 = {"activitiy" : activitiy, "start" : start, "end" : end, "batch" : batch}
    fall_anclean1 = pd.DataFrame(fall_an_dict1)
    #json
    list_json = os.listdir(path_sensor)
    list_json = [ s for s in list_json if "s_" in s]

    all_jason_df = pd.DataFrame(columns=["time_stamp","sensor_time_stamp","X","Y","Z"])
    for sen_jsonpath in list_json:
        sensor_path = jsonpath + sen_jsonpath
        sen_json = pd.read_json(sensor_path)
        if sen_json.dt["a"] == []:
            continue
        sen_acc = np.array(sen_json.dt["a"])
        acc_df = pd.DataFrame(sen_acc, columns=["time_stamp", "sensor_time_stamp","X","Y","Z"])
        all_jason_df = pd.concat([all_jason_df,acc_df])
    all_jason_df["activity"] = all_jason_df["time_stamp"].apply(check_between)

    all_jason_df.dropna(inplace=True)
    all_jason_df["model"] = "ASUS_X013DB" #depends on the phone
    all_jason_df.reset_index(inplace=True)
    all_jason_df.drop("index", axis = 1, inplace=True)

    n = 0
    npypath = "../raw_data/fall_clean/"
    for row in range(all_jason_df.shape[0]):
        curr_act = all_jason_df.loc[row,"activity"]
        curr_act = curr_act.split()[0]
        curr_batch = all_jason_df.loc[row,"activity"]
        curr_batch = curr_batch.split()[1]
        model = all_jason_df.loc[row,"model"]
        curr_time = int(all_jason_df.loc[row,"time_stamp"])
        acc_value = all_jason_df.loc[row,"time_stamp":"Z"]
        acc_value = [acc_value.to_numpy(dtype = 'float64')]
        if row == 0:
            acc = acc_value
            pre_batch = curr_batch
        else:
            if pre_batch == curr_batch:
                acc = np.append(acc,acc_value, axis = 0)
                pre_batch = curr_batch
                n += 1
                save_path = f"{npypath}/{curr_act}/{curr_act}_{model}_{curr_time}.npy"
            else:
                if n == 0:
                    save_path = f"{npypath}/{curr_act}/{curr_act}_{model}_{curr_time}.npy"
                zero = np.zeros( acc.shape, dtype=float)
                stack_gb = np.stack((acc,zero,zero))
                np.save(save_path, stack_gb)
                pre_batch = curr_batch
                acc = acc_value
