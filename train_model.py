import pandas as pd
from sklearn.model_selection import train_test_split
import os

from deep_classifier import DeepClassifier


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    df = pd.read_pickle('../data/df2.pcl')
    x = df.drop('target', axis=1)
    y = df['target']

    dae = DeepClassifier(features=x.shape[1], restart=True, batch_size=512)

    dae.train_dae(x)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    dae.train_clf(x_train, y_train, x_val, y_val, restart=False)


if __name__=='__main__':
    main()