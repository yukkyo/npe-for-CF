import tensorflow as tf
import numpy as np
import pandas as pd
from npe import NeuralPersonalizedEmbedding
import gc
from sklearn.model_selection import train_test_split
from numba import jit


def main():
    options = {
        "num_users": 10,
        "num_items": 5,
        "dim_emb": 15,
        "seed": 10,
        "learning_rate": 0.001,
        "dropout_rate": 0.5  # == 1 - keep_prob. It is not tuned value
    }
    # args = parse_args()
    epoch = 100
    num_users = 10
    num_items = 5
    input_R = np.random.randint(0, 2, num_users * num_items).reshape((num_users, num_items))
    input_userids = np.array([i for i in range(num_users) for j in range(num_items)])
    input_itemids = np.array([j for i in range(num_users) for j in range(num_items)])
    input_labels = np.array([input_R[i][j] for i in range(num_users) for j in range(num_items)])

    with tf.Graph().as_default(), tf.Session() as session:
        model = NeuralPersonalizedEmbedding(options, session)
        for i in range(epoch):
            model.train(input_R, input_userids, input_itemids, input_labels)
            if i % 5 == 0:
                model.print_loss()
        print(model.predict(input_R, [3], [3]))
        print(input_R[3][3])
        #    model.eval()  # loss の表示
        # model.save(session, "./mymodel_20180601.ckpt")


def preparation(df):
    df["userid"] = df["userid"] - 1
    df["itemid"] = df["itemid"] - 1
    df["rating"] = df["rating"] // 4  # 0 if less than 4 else 1
    print("Preparation end")
    return df


@jit
def count_user_item_mtx(user_item_mtx, userids, itemids, labels):
    for i in range(len(userids)):
        user_item_mtx[userids[i]][itemids[i]] = labels[i]
    return user_item_mtx


def split_userid_itemid(df):
    userids = df["userid"].values
    itemids = df["itemid"].values
    labels = df["rating"].values
    return userids, itemids, labels


def split_train_valid_test(df, ratios, seed):
    """
    Splitting train, valid, test data
    :param df:
    :param ratios: [train, valid, test] ex. [0.7, 0.1, 0.2]
    :param seed: seed for random split
    :return:
    """
    test_size = ratios[2] / (sum(ratios))
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    valid_size = ratios[1] / (ratios[0] + ratios[1])
    df_train, df_valid = train_test_split(df_train, test_size=valid_size, random_state=seed)
    print("train_size:", len(df_train), "valid_size:", len(df_valid), "test_size", len(df_test))
    return df_train, df_valid, df_test


def test():
    epoch = 3
    batch_size = 10000
    # number of negative samples per positive example
    negative_rate = 4
    
    # TODO: use pathlib
    df = pd.read_csv(
        "../input/ml-100k/u.data",
        sep="\t",
        names=["userid", "itemid", "rating", "timestamp"],
        usecols=["userid", "itemid", "rating"]
    )
    print("load end!")
    count_users = df["userid"].max()
    count_items = df["itemid"].max()
    # Options
    options = {
        "num_users": count_users,
        "num_items": count_items,
        "dim_emb": 30,
        "seed": 10,
        "learning_rate": 0.001,
        "dropout_rate": 0.3,  # == 1 - keep_prob,
        "ratios_train_valid_test": [0.7, 0.1, 0.2],
    }
    # Data preparation
    df = preparation(df)

    # TODO: calc loss by valid
    # TODO: early stopping

    # Split train, valid, test
    df_train, df_valid, df_test = split_train_valid_test(df,
                                                         options["ratios_train_valid_test"],
                                                         options["seed"])
    del df
    gc.collect()
    # for negative down sampling
    df_positive = df_train[df_train["rating"] == 1].copy()
    df_negative = df_train[df_train["rating"] == 0].copy()
    sample_size_negative = len(df_negative)
    if len(df_positive) * negative_rate < len(df_negative):
        sample_size_negative = len(df_positive) * negative_rate
    gc.collect()
    print("train(positive): ", len(df_positive), "train(negative)", len(df_negative))

    # user item matrix by train data. It is not changed by negative samples
    userids, itemids, labels = split_userid_itemid(df_positive)
    user_item_mtx_train = np.zeros((count_users, count_items))
    user_item_mtx_train = count_user_item_mtx(user_item_mtx_train, userids,
                                              itemids, labels)

    # user item matrix by valid data
    userids_valid, itemids_valid, labels_valid = split_userid_itemid(df_valid)
    user_item_mtx_valid = np.zeros((count_users, count_items))
    user_item_mtx_valid = count_user_item_mtx(user_item_mtx_valid,
                                              userids_valid,
                                              itemids_valid,
                                              labels_valid)

    with tf.Graph().as_default(), tf.Session() as sess:
        # model init
        model = NeuralPersonalizedEmbedding(options, sess)
        for i in range(epoch):
            # Negative down sampling
            df_tmp = pd.concat([df_positive,
                                df_negative.sample(n=sample_size_negative)],
                               axis=0)
            userids, itemids, labels = split_userid_itemid(df_tmp)
            del df_tmp
            gc.collect()
            # Batch processing
            rnd_idx = np.random.permutation(len(userids))
            for idxs in np.array_split(rnd_idx, len(userids) // batch_size):
                # user-item matrix is not changed by negative down sampling
                model.train(user_item_mtx_train,
                            userids[idxs],
                            itemids[idxs],
                            labels[idxs])
        # For check early stopping
            print("epoch: ", epoch)
            model.print_loss()
        print("learning end")


if __name__ == "__main__":
    # main()
    test()