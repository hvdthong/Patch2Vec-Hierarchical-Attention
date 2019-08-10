import pickle


if __name__ == '__main__':
    path_data = './data/jit_openstack.pkl'
    # path_data = './data/jit_qt.pkl'
    with open(path_data, 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code, _ = data