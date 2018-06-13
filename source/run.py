import tensorflow as tf
import numpy as np
import pandas as pd
from npe import NeuralPersonalizedEmbedding
import gc
from sklearn.model_selection import train_test_split
from numba import jit

# Options
options = {
    # Parameters  for npe
    "num_users": 0,  # redefine when read data
    "num_items": 0,  # redefine when read data
    "dim_emb": 50,
    "seed": 10,
    "learning_rate": 0.001,
    "dropout_rate": 0.3,  # == 1 - keep_prob,
    # These parameters is not necessary for npe
    "ratios_train_valid_test": [0.7, 0.1, 0.2],
    "count_for_early_stopping": 5,
    "path_save_model": "../saved_model/",
    "epoch": 3,
    "batch_size": 10000,
    "negative_rate": 4  # number of negative samples per positive example
}


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


def train(df_train, df_valid):
    """
    Train npe model with these techniques
      * negative down sampling
      * mini-batch learning
      * early stopping
    :param df_train:
    :param df_valid:
    :return:
    """
    # for negative down sampling
    df_positive = df_train[df_train["rating"] == 1].copy()
    df_negative = df_train[df_train["rating"] == 0].copy()
    sample_size_negative = len(df_negative)
    if len(df_positive) * options["negative_rate"] < len(df_negative):
        sample_size_negative = len(df_positive) * options["negative_rate"]
    gc.collect()
    print("train(positive): ", len(df_positive), "train(negative)", len(df_negative))

    # shape of user-item matrix
    shape = (options["num_users"], options["num_items"])

    # user item matrix by train data. It is not changed by negative samples
    userids, itemids, labels = split_userid_itemid(df_positive)
    user_item_mtx_train = count_user_item_mtx(np.zeros(shape), userids, itemids, labels)

    # user item matrix by valid data
    userids_valid, itemids_valid, labels_valid = split_userid_itemid(df_valid)
    user_item_mtx_valid = count_user_item_mtx(np.zeros(shape), userids_valid,
                                              itemids_valid, labels_valid)

    with tf.Graph().as_default(), tf.Session() as sess:
        # model init
        model = NeuralPersonalizedEmbedding(options, sess)
        # loss for early stopping
        loss_valid_best = np.inf
        cnt_not_decrease_loss = 0

        for i in range(options["epoch"]):
            print("epoch: ", i, end=" ")
            # Negative down sampling
            df_tmp = pd.concat([df_positive,
                                df_negative.sample(n=sample_size_negative)],
                               axis=0)
            userids, itemids, labels = split_userid_itemid(df_tmp)
            del df_tmp
            gc.collect()
            # Mini batch learning
            rnd_idx = np.random.permutation(len(userids))
            for idxs in np.array_split(rnd_idx, len(userids) // options["batch_size"]):
                # user-item matrix is not changed by negative down sampling
                model.train(user_item_mtx_train, userids[idxs],
                            itemids[idxs], labels[idxs])
            print("end")
            # Print train, valid loss
            loss_train = model.get_loss()
            loss_valid = model.calc_loss(user_item_mtx_valid, userids_valid,
                                         itemids_valid, labels_valid)
            print("loss(train): ", loss_train, "loss(valid): ", loss_valid)
            # Early stopping
            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
            else:
                cnt_not_decrease_loss += 1
                if cnt_not_decrease_loss >= options["count_for_early_stopping"]:
                    print("Early stopping !!")
                    break
        print("learning end")
        model.save(sess, options["path_save_model"])


def test_train():
    # TODO: use pathlib
    # Read data
    df = pd.read_csv(
        "../input/ml-100k/u.data",
        sep="\t",
        names=["userid", "itemid", "rating", "timestamp"],
        usecols=["userid", "itemid", "rating"]
    )
    print("load end!")
    # Modify options
    # ops = Option()
    options["num_users"] = df["userid"].max()
    options["num_items"] = df["itemid"].max()

    # Data preparation
    df = preparation(df)

    # Split train, valid, test
    df_train, df_valid, df_test = split_train_valid_test(df,
                                                         options["ratios_train_valid_test"],
                                                         options["seed"])
    del df
    gc.collect()

    # training model
    train(df_train, df_valid)

    # TODO: Predict & eval


if __name__ == "__main__":
    # main()
    test_train()