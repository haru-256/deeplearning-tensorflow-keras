import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(1234)


def inference(x, n_in, n_hiddens, n_out, train_fg):
    def weight_variable(shape):
        initial = np.sqrt(2.0 / shape[0]) * tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def batch_normalization(shape, x):
        eps = 1e-8
        beta = tf.Variable(tf.zeros(shape))  # betaとgammaは今の層のニューロン数
        gamma = tf.Variable(tf.ones(shape))  # の縦ベクトルとなる．
        # tf.nn.moments(x, axes=[0]) はaxes方向の平均と分散を求める．
        # axes=[0] で行方向，axes=[0,1]で全体の平均と分散を求める．
        mean, var = tf.nn.moments(x, axes=[0])
        return gamma * (x - mean) / tf.sqrt(var + eps) + beta

    # 入力層 - 隠れ層、隠れ層 - 隠れ層
    for i, n_hidden in enumerate(n_hiddens):
        if i == 0:
            input = x
            input_dim = n_in
        else:
            input = output
            input_dim = n_hiddens[i - 1]

        # W = weight_variable([input_dim, n_hidden])

        # tf.contrib.layers.variance_scaling_initializer によりHeの初期値を利用
        # することができるTensorを作成 詳細はoDocument参照
        he_init = tf.contrib.layers.variance_scaling_initializer(
            factor=2,
            mode='FAN_IN',
            uniform=False,
            seed=None,
            dtype=tf.float32
        )
        """これでもOKだが tf.name_scopeと使わない方が良い
        # W = tf.get_variable("W_{}".format(i), shape=[input_dim, n_hidden],
                            initializer=he_init)
        """

        W = tf.Variable(he_init([input_dim, n_hidden]))
        u = tf.matmul(input, W)
        # h = batch_normalization([n_hidden], u)
        """
        eps = 1e-8
        beta = tf.Variable(tf.zeros([n_hidden]))  # betaとgammaは今の層のニューロン数の縦ベクトルとなる．
        # gamma = tf.Variable(tf.ones([n_hidden]))  # gammma を適用すると発散してしまう．
        # tf.nn.moments(x, axes=[0]) はaxes方向の平均と分散を求める．
        # axes=[0] で行方向，axes=[0,1]で全体の平均と分散を求める．
        mean, var = tf.nn.moments(u, axes=[0])
        h = tf.nn.batch_normalization(u,
                                      mean=mean,
                                      variance=var,
                                      variance_epsilon=eps,
                                      offset=beta,
                                      scale=None)
        """
        # BNはtf.layers.batch_normalizationを使うと楽
        h = tf.layers.batch_normalization(u,
                                          training=train_fg,
                                          momentum=0.99,
                                          epsilon=0.001
                                          )
        output = tf.nn.relu(h)

    # 隠れ層 - 出力層
    W_out = weight_variable([n_hiddens[-1], n_out])
    b_out = bias_variable([n_out])
    y = tf.nn.softmax(tf.matmul(output, W_out) + b_out)
    return y


def loss(y, t):
    cross_entropy = \
        tf.reduce_mean(-tf.reduce_sum(
                       t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                       reduction_indices=[1]))
    return cross_entropy


def training(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1,  # BNを用いているので学習率
                                       beta1=0.9,          # を大きくしても収束する．
                                       beta2=0.999)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # BNのために必要
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss)
    return train_step


def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


if __name__ == '__main__':
    '''
    データの生成
    '''
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    n = len(mnist.data)
    N = 30000  # MNISTの一部を使う
    N_train = 20000
    N_validation = 4000
    indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

    X = mnist.data[indices]
    # X = X / 255.0
    # X = X - X.mean(axis=1).reshape(len(X), 1)
    y = mnist.target[indices]
    Y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, train_size=N_train)

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X_train, Y_train, test_size=N_validation)

    '''
    モデル設定
    '''
    n_in = len(X[0])
    n_hiddens = [200, 200, 200]  # 各隠れ層の次元数
    n_out = len(Y[0])

    x = tf.placeholder(tf.float32, shape=[None, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    # BNの推論か訓練かを切り替えるためのplacefolder
    train_fg = tf.placeholder(dtype=tf.bool)

    y = inference(x, n_in=n_in,
                  n_hiddens=n_hiddens,
                  n_out=n_out,
                  train_fg=train_fg)
    loss = loss(y, t)
    train_step = training(loss)

    accuracy = accuracy(y, t)
    early_stopping = EarlyStopping(patience=10, verbose=1)

    history = {
        'val_loss': [],
        'val_acc': []
    }

    '''
    モデル学習
    '''
    epochs = 50
    batch_size = 200

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    n_batches = N_train // batch_size

    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step, feed_dict={
                x: X_[start:end],
                t: Y_[start:end],
                train_fg: True
            })

        # 検証データを用いた評価
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            train_fg: False
        })
        val_acc = accuracy.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            train_fg: False
        })

        # 検証データに対する学習の進み具合を記録
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print('epoch:', epoch,
              ' validation loss:', val_loss,
              ' validation accuracy:', val_acc)

        # Early Stopping チェック
        if early_stopping.validate(val_loss):
            break

    '''
    学習の進み具合を可視化
    '''
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(history['val_loss'])), history['val_loss'],
             label='loss', color='black')
    plt.xlabel('epochs')
    plt.show()

    '''
    予測精度の評価
    '''
    accuracy_rate = accuracy.eval(session=sess, feed_dict={
        x: X_test,
        t: Y_test
    })
    print('accuracy: ', accuracy_rate)
