import pandas as pd
import numpy as np


def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df.values


def get_features(data):
    # return the features of yasu data
    return data[:, 11:32]


def get_ids(data):
    # return the labels of yasu data
    return data[:, 1:2].flatten().tolist()


def get_label(data):
    data = data[:, 3:4].flatten().tolist()
    data = [1 if int(d) > 0 else 0 for d in data]
    return data


def load_df_yasu_data(path_data):
    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    indexes, new_ids, new_labels, new_features = list(), list(), list(), list()
    cnt_noexits = 0
    for i in range(0, len(ids)):
        try:
            indexes.append(i)
        except FileNotFoundError:
            print('File commit id no exits', ids[i], cnt_noexits)
            cnt_noexits += 1
    ids = [ids[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    features = features[indexes]
    return (ids, np.array(labels), features)


def load_yasu_data(project):
    if project == 'openstack':
        ids, labels, features = list(), list(), list()
        path_data = '../data/jit_defect/yasu_replication_data/' + project + '.STRATA_PER_YEAR.4.all.8.csv'
        data = load_df_yasu_data(path_data=path_data)
        return data


if __name__ == '__main__':
    project = 'openstack'
    path_file = '../output/' + project
    data = load_yasu_data(project=project)
