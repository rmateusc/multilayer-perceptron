# import required packages
import bisect
import numpy as np
from pandas import DataFrame

def shuffle_data(df: DataFrame) -> DataFrame:
    """Shuffles rows of the DataFrame according to its index

    Args:
        df (DataFrame): DataFrame to be shuffled

    Returns:
        df (DataFrame): DataFrame shuffled
    """
    df = df.sample(frac=1)
    return df

def random_number(distribution: str) -> list:
    """Creates a list of random numbers (between [0,1]) according to a distribution.

    Args:
        size (int): size of the list
        distribution (str): uniform, triangular, beta

    Returns:
        list (list): list of random numbers between [0,1]
    """
    if distribution == 'uniform':
        return np.random.uniform(0, 1)
    elif distribution == 'triangular':
        return np.random.triangular(0, 0.5, 1)
    elif distribution == 'beta':
        return np.random.beta(1, 1)
    else:
        raise Exception('Distribution not recognized')

def binary_search_df(range: list, num: float) -> int:
    """Binary search algorithm for a number in an interval.

    Args:
        range (list): list of numbers
        num (float): number to be found

    Returns:
        index (int): index of the number in the list
    """
    new_range = np.delete(range, 0)
    index = bisect.bisect_left(new_range, num)
    return index

def random_pick(df: DataFrame, train_size: float, val_size: float, distribution: str):
    """Picks rows of the DataFrame randomly according to a distribution.

    Args:
        df (DataFrame): DataFrame
        train_size (float): percentage of the DataFrame to be used for training
        val_size (float): percentage of the DataFrame to be used for validation
        distribution (str): uniform, triangular, beta

    Returns:
        train (DataFrame): DataFrame with the rows picked randomly
        test (DataFrame): DataFrame with the rows not picked
    """

    df = shuffle_data(df)

    train_size = int(train_size * len(df))
    val_size = int(val_size * len(df))
    df_range = np.arange(0, 1, 1/len(df))

    train_indices = []
    while len(train_indices) < train_size:
        rand_num = random_number(distribution)
        index = binary_search_df(df_range, rand_num)
        if index not in train_indices:
            train_indices.append(index)

    val_indices = []
    while len(val_indices) < val_size:
        rand_num = random_number(distribution)
        index = binary_search_df(df_range, rand_num)
        if (index not in train_indices) and (index not in val_indices):
            val_indices.append(index)

    test_indices = []
    for i in range(len(df)):
        if (i not in train_indices) and (i not in val_indices):
            test_indices.append(i)

    train_df = df.loc[train_indices, :]
    val_df = df.loc[val_indices, :]
    test_df = shuffle_data(df.loc[test_indices, :])

    return train_df, val_df, test_df
