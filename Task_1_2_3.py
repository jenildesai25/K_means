from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np


def make_data_of_100_rows(file_path):
    try:
        if file_path == 'ATNTFaceImages400.txt':
            # here the data is not transpose.columns contains images. ex. column 1st contains the image.
            data_frame = pd.read_csv(file_path, header=None, sep=",")
            data_frame_with_data = data_frame.iloc[:, :100]
            data_frame_without_label = data_frame_with_data.iloc[1:, :]
            # labels are in columns which has size (100,0)
            labels = data_frame_with_data.transpose()[0].values
            return data_frame_with_data, labels, data_frame_without_label
        elif file_path == 'HandWrittenLetters.txt':
            data_frame = pd.read_csv(file_path, header=None, sep=",")
            # data_frame_with_data = data_frame.iloc[:, :100]
            data_frame_without_label = data_frame.iloc[1:, :]
            # labels are in columns which has size (100,0)
            labels = data_frame.transpose()[0].values
            return data_frame, labels, data_frame_without_label
    except Exception as e:
        print(e)


def make_data_of_400_rows(file_path):
    try:
        if file_path == 'ATNTFaceImages400.txt':
            # here the data is not transpose.columns contains images. ex. column 1st contains the image.
            data_frame = pd.read_csv(file_path, header=None, sep=",")
            data_frame_with_data = data_frame.iloc[:, :400]
            data_frame_without_label = data_frame_with_data.iloc[1:, :]
            # labels are in columns which has size (100,0)
            labels = data_frame_with_data.transpose()[0].values
            return data_frame_with_data, labels, data_frame_without_label
    except Exception as e:
        print(e)


def k_means_algo(data_no_label, label_values, k):
    try:
        data_no_label = data_no_label.transpose()
        k_means = KMeans(n_clusters=k)
        k_means.fit(data_no_label)
        labels_from_kmeans = k_means.labels_
        print('K means labels', labels_from_kmeans)
        C = confusion_matrix(y_true=label_values, y_pred=labels_from_kmeans)
        print('Confusion matrix is: ', C)
        C = C.T
        ind = linear_assignment(-C)
        C_opt = C[:, ind[:, 1]]
        print('re ordered matrix', C_opt)
        acc_opt = np.trace(C_opt) / np.sum(C_opt)
        accuracy = cluster_acc(label_values, labels_from_kmeans)
        print('accuracy of k means is:', accuracy * 100)
    except Exception as e:
        print(e)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    try:
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    except Exception as e:
        print(e)


if __name__ == '__main__':
    user_input = input('please enter the file name if file does not exist in the directory program exists please give full path: ')
    k = int(input('enter value of k: '))
    if 'ATNT' in user_input and k == 40:
        data, label, data_without_label = make_data_of_400_rows('ATNTFaceImages400.txt')
        k_means_algo(data_without_label, label, k)
    elif 'Hand' in user_input:
        data, label, data_without_label = make_data_of_100_rows('HandWrittenLetters.txt')
        k_means_algo(data_without_label, label, k)
    elif 'ATNT' in user_input:
        data, label, data_without_label = make_data_of_100_rows('ATNTFaceImages400.txt')
        k_means_algo(data_without_label, label, k)