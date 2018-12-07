from pandas._libs.parsers import k
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np


def make_data_of_100_rows(file_path):
    try:
        # here the data is not transpose.columns contains images. ex. column 1st contains the image.
        data_frame = pd.read_csv(file_path, header=None, sep=",")
        data_frame_with_data = data_frame.iloc[:, :100]
        labels = data_frame_with_data[0:1]
        data_frame_without_label = data_frame_with_data.iloc[1:, :]
        return data_frame_with_data, labels, data_frame_without_label
    except Exception as e:
        print(e)


def k_means_algo(data_no_label, label_values, k):
    try:
        data_no_label = data_no_label.transpose()
        k_means = KMeans(n_clusters=k)
        k_means.fit(data_no_label)
        labels_from_kmeans = k_means.labels_
        print('length of labels by kmenas', len(labels_from_kmeans))
        print('length of labels',len(label_values.values))
        # print('label from data', label_values)
        # print('labels from kmeans algo is', labels_from_kmeans)
        C = confusion_matrix(y_true=label_values, y_pred=labels_from_kmeans)
        C = C.T
        ind = linear_assignment(-C)
        C_opt = C[:, ind[:, 1]]
        acc_opt = np.trace(C_opt) / np.sum(C_opt)
        accuracy = cluster_acc(label_values, labels_from_kmeans)
        print('accuracy of k means is:', accuracy)
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
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


if __name__ == '__main__':
    data, label, data_without_label = make_data_of_100_rows('ATNTFaceImages400.txt')
    # data = data.transpose()
    k_means_algo(data_without_label, label, 10)
