from sklearn.cluster import KMeans
import pandas as pd


def make_data_of_100_rows(file_path):
    data_frame = pd.read_csv(file_path)
    data_frame = data_frame[:100]
    return data_frame


if __name__ == '__main__':
    make_data_of_100_rows('ATNTFaceImages400.txt')
