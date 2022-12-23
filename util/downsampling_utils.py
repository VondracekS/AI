import pandas as pd
from sklearn.utils import resample


def train_downsample(target_var: str, data_train: pd.DataFrame, downsampling_factor=1,
                     positive_case=1) -> pd.DataFrame:
    """
    Perform downsampling on the training set in order to prevent overfitting on the prevalent class
    :param target_var: Label binary variable
    :param data_train: Train set
    :param downsampling_factor: Denotes the rate of positive/negative class in the label variable, default is 1:1
    :param positive_case: Denotes the positive case in the sample, default is 1
    :return: Downsampled df
    """

    target_positive = data_train.loc[data_train[target_var] == positive_case].shape[0]
    train_downsampled = resample(data_train.loc[data_train[target_var] != positive_case], replace=True,
                                 n_samples=int(target_positive / downsampling_factor),
                                 random_state=42)
    train_downsampled = pd.concat([train_downsampled, data_train.loc[data_train[target_var]==positive_case]])
    train_downsampled = resample(train_downsampled, replace=False)

    return train_downsampled
