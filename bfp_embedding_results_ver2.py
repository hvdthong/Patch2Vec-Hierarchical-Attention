import pickle
from parameters import read_args

if __name__ == '__main__':
    with open('./data/bfp_linux.pickle', 'rb') as input:
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

    input_option.datetime = '2019-07-21_20-56-53'
    input_option.start_epoch = 5
    input_option.end_epoch = 5

    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_model = './embedding/' + input_option.datetime + '/epoch_' + str(epoch) + '.txt'