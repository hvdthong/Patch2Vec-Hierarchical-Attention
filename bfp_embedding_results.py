from parameters import read_args
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def bfp_clf_results(path, labels=None, algorithm=None, kfold=5):
    embedding = np.loadtxt(path)  # be careful with the shape since we don't include the last batch
    nrows = embedding.shape[0]
    labels = labels[:nrows]
    kf = KFold(n_splits=kfold)
    clf = None
    if algorithm == 'lr':
        clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=embedding, y=labels)
    elif algorithm == 'svm':
        clf = SVC(gamma='auto').fit(X=embedding, y=labels)
    elif algorithm == 'nb':
        clf = GaussianNB().fit(X=embedding, y=labels)

    y_pred = clf.predict(embedding)
    print(precision_score(y_true=labels, y_pred=y_pred))


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

    # input_option.datetime = '2019-07-08_23-13-28'
    # input_option.start_epoch = 1
    # input_option.end_epoch = 50
    #
    # input_option.datetime = '2019-07-12_21-00-10'
    # input_option.start_epoch = 1
    # input_option.end_epoch = 20
    #
    # input_option.datetime = '2019-07-08_23-11-15'
    # input_option.start_epoch = 1
    # input_option.end_epoch = 22

    # input_option.datetime = '2019-07-08_23-11-15'
    # input_option.start_epoch = 1
    # input_option.end_epoch = 100

    input_option.datetime = '2019-07-21_20-56-53'
    input_option.start_epoch = 1
    input_option.end_epoch = 5

    algorithm, kfold = 'lr', 5
    # algorithm, kfold = 'svm', 5
    # algorithm, kfold = 'nb', 5

    for epoch in range(input_option.start_epoch, input_option.end_epoch + 1):
        path_model = './embedding/' + input_option.datetime + '/epoch_' + str(epoch) + '.txt'
        bfp_clf_results(path=path_model, labels=labels, algorithm=algorithm, kfold=kfold)
        # exit()
