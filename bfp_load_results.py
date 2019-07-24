from parameters import read_args_cnn

if __name__ == '__main__':
    input_option = read_args_cnn().parse_args()

    input_option.path_model = '2019-07-22_16-57-31'
    input_option.start_model = 1
    input_option.end_model = 100