import pickle
from parameters import read_args

if __name__ == '__main__':
    # loading data
    ##########################################################################################################
    # data = pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code
    # pad_msg: padding commit message
    # pad_added_code: padding added code
    # pad_removed_code: padding removed code
    # labels: label of our data, stable or non-stable patches
    # dict_msg: dictionary of commit message
    # dict_code: dictionary of commit code

    with open('./data/linux_bfp.pickle', 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    ##########################################################################################################
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print(len(dict_msg), len(dict_code))

    input_option = read_args().parse_args()
    input_help = read_args().print_help()



