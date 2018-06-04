import tensorflow as tf
import numpy as np
import pandas as pd
from npe import NeuralPersonalizedEmbedding
import gc


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


def split_userid_itemid(df):
    userids = df["userid"].values
    itemids = df["itemid"].values
    labels = df["rating"].values
    return userids, itemids, labels


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
    # Data preparation
    df = preparation(df)
    # user item matrix
    user_item_mtx = df.pivot(index='userid', columns='itemid', values='rating').fillna(0).values
    # Options
    options = {
        "num_users": user_item_mtx.shape[0],
        "num_items": user_item_mtx.shape[1],
        "dim_emb": 30,
        "seed": 10,
        "learning_rate": 0.001,
        "dropout_rate": 0.3  # == 1 - keep_prob
    }
    print(df.describe())
    print(options)
    print(user_item_mtx.shape)

    # TODO: dropout
    # TODO: split train, valid, test
    # TODO: calc loss by valid
    # TODO: early stopping

    # for negative down sampling
    df_positive = df[df["rating"] == 1].copy()
    df_negative = df[df["rating"] == 0].copy()
    sample_size_negative = len(df_negative)
    if len(df_positive) * negative_rate < len(df_negative):
        sample_size_negative = len(df_positive) * negative_rate
    del df
    gc.collect()

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
                model.train(user_item_mtx,
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