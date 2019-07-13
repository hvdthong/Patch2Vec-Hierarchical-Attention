from parameters import read_args
import pickle

if __name__ == '__main__':
    with open('./data/linux_bfp.pickle', 'rb') as input:
        data = pickle.load(input)
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    ##########################################################################################################
    print(pad_msg.shape, pad_added_code.shape, pad_removed_code.shape, labels.shape)
    print('Shape of the commit message:', pad_msg.shape)
    print('Shape of the added/removed code:', (pad_added_code.shape, pad_removed_code.shape))
    print('Shape of the label of bug fixing patches:', labels.shape)
    print('Total words in the message dictionary: ', len(dict_msg))
    print('Total words in the code dictionary: ', len(dict_code))

    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    input_option.datetime = '2019-07-08_23-13-28'
    input_option.embed_size = 128
    input_option.hidden_size = 64
    input_option.start_epoch = 1
    input_option.end_epoch = 50
