import pickle


if __name__ == '__main__':
    path_data = './data/jit_openstack.pkl'
    # path_data = './data/jit_qt.pkl'
    with open(path_data, 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code, _ = data
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Shape of the label of bug fixing patches:', labels.shape)
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))

