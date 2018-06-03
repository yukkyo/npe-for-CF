import tensorflow as tf


class NeuralPersonalizedEmbedding(object):
    def __init__(self, options, session):
        self._session = session
        self._num_users = options["num_users"]
        self._num_items = options["num_items"]
        self._dim_emb = options["dim_emb"]
        self._learning_rate = options["learning_rate"]
        self.build_graph()

    def build_graph(self):
        R = tf.placeholder(tf.float32, shape=[None, None])
        userids = tf.placeholder(tf.int32)
        itemids = tf.placeholder(tf.int32)
        labels = tf.placeholder(tf.float32)
        self._R = R
        self._userids = userids
        self._itemids = itemids
        self._labels = labels

        # Parameters to be learned
        emb_user = tf.Variable(
            tf.random_normal(shape=[self._num_users, self._dim_emb]))
        emb_item = tf.Variable(
            tf.random_normal(shape=[self._num_items, self._dim_emb]))
        emb_context_item = tf.Variable(
            tf.random_normal(shape=[self._num_items, self._dim_emb]))

        # Convert to onehot
        userids_onehot = tf.one_hot(userids, depth=self._num_users, dtype=tf.float32)
        userids_onehot = tf.reshape(userids_onehot, [-1, self._num_users])
        itemids_onehot = tf.one_hot(itemids, depth=self._num_items, dtype=tf.float32)
        itemids_onehot = tf.reshape(itemids_onehot, [-1, self._num_items])

        # Converting userids in training data to embedding
        h = tf.nn.relu(tf.nn.embedding_lookup(emb_user, userids))

        # Converting itemids in training data to embedding
        w = tf.nn.relu(tf.nn.embedding_lookup(emb_item, itemids))

        # Convert training data to contet
        # contet = history user u clicked without item i
        context = tf.subtract(tf.matmul(userids_onehot, R), itemids_onehot)

        # Convert context to embedding and through relu
        v = tf.nn.relu(tf.matmul(context, emb_context_item))

        # inner product
        r_1 = tf.reduce_sum(tf.multiply(h, w), axis=1)
        r_2 = tf.reduce_sum(tf.multiply(w, v), axis=1)
        r = tf.add(r_1, r_2)

        # Predict
        predicts = tf.nn.sigmoid(r)
        self._predictop = predicts

        # loss: binary cross entropy
        bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=r
        )
        loss = tf.reduce_sum(bce)
        self._lossop = loss

        # Optimizer
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train = optimizer.minimize(loss)

        # Trainer
        self._trainop = train
        tf.global_variables_initializer().run()

        # Saver
        self.saver = tf.train.Saver()
        print("graph build!")

    def train(self, user_item_mtx, userids, itemids, labels):
        """Do train one epoch"""
        self._session.run(
            self._trainop,
            feed_dict={
                self._R: user_item_mtx, self._userids: userids,
                self._itemids: itemids, self._labels: labels
            }
        )
        self._loss = self._session.run(
            self._lossop,
            feed_dict={
                self._R: user_item_mtx, self._userids: userids,
                self._itemids: itemids, self._labels: labels
            }
        )

    def predict(self, user_item_mtx, userids, itemids):
        predicts = self._session.run(
            self._predictop,
            feed_dict={
                self._R: user_item_mtx, self._userids: userids,
                self._itemids: itemids
            }
        )
        return predicts

    def print_loss(self):
        print("loss: ", self._loss)

    def get_loss(self):
        return self._loss

    def save(self, session, path):
        self.saver.save(session, path)
        print("model saved!")