import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(123)


def inference(x, n_batch, maxlen=None, n_hidden=None, n_out=None):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)

    # cell = tf.contrib.rnn.BasicRNNCell(n_hidden)

    # tf.contrib.rnn is aliases of tf.nn.rnn_cell
    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    # To set initial state to zeros
    initial_state = cell.zero_state(n_batch, tf.float32)  # return zero-filled state tensors.

    state = initial_state
    outputs = []  # 過去の隠れ層の出力を保存
    with tf.variable_scope('RNN'):  # Use argument of `reuse=True` instead of tf.get_variable_scope().reuse_variables()
        for t in range(maxlen):
            if t > 0:
                tf.get_variable_scope().reuse_variables()  # reuse variables
            (cell_output, state) \
                = cell(x[:, t, :], state)  # Run this RNN cell on inputs, starting from the given state
            outputs.append(cell_output)

    V = weight_variable([n_hidden, n_out])
    c = bias_variable([n_out])
    ys = [tf.matmul(output, V) + c for output in outputs]  # 線形活性

    return ys  # Loss Function is composed of all output


def loss(ys, t):
    mse = tf.reduce_mean(tf.square(ys - t))
    # mse = tf.reduce_mean(tf.square(y - t))
    # mse = tf.losses.mean_squared_error(t, y)
    return mse


def training(loss):
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

    train_step = optimizer.minimize(loss)
    return train_step


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
    def sin(x, T=100):
        return np.sin(2.0 * np.pi * x / T)

    def toy_problem(T=100, ampl=0.05):
        x = np.arange(0, 2 * T + 1)
        noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
        return sin(x) + noise

    '''
    データの生成
    '''
    T = 100
    f = toy_problem(T)

    length_of_sequences = 2 * T  # 全時系列の長さ
    maxlen = 25  # ひとつの時系列データの長さ

    data = []
    target = []

    for i in range(0, length_of_sequences - maxlen + 1):
        data.append(f[i: i + maxlen])
        target.append(f[i: i + maxlen])

    X = np.array(data).reshape(len(data), maxlen, 1)
    Y = np.array(target).reshape(len(data), maxlen, 1)

    # データ設定
    N_train = int(len(data) * 0.9)
    N_validation = len(data) - N_train

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=N_validation, train_size=N_train)

    """
    モデル設定
    """
    n_in = len(X[0][0])  # 1(scalar)
    n_hidden = 20
    n_out = 1  # 1(scalar)

    x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
    t = tf.placeholder(tf.float32, shape=[None, maxlen, n_out])
    n_batch = tf.placeholder(tf.int32, shape=[])  # In the case of scalar, shape=[]

    y = inference(x, n_batch, maxlen=maxlen, n_hidden=n_hidden, n_out=n_out)
    loss = loss(y, t)
    train_step = training(loss)

    early_stopping = EarlyStopping(patience=15, verbose=1)
    history = {
        'val_loss': []
    }

    '''
    モデル学習
    '''
    epochs = 500
    batch_size = 10

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
                n_batch: batch_size
            })

        # 検証データを用いた評価
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            n_batch: N_validation
        })

        history['val_loss'].append(val_loss)
        print('epoch:', epoch,
              ' validation loss:', val_loss)

        # Early Stopping チェック
        if early_stopping.validate(val_loss):
            break

    '''
    出力を用いて予測
    '''
    truncate = maxlen
    Z = X[:1]  # 元データの最初の一部だけ切り出し, ndim=3とするためにX[0]ではなくX[:1]を使う

    original = [f[i] for i in range(maxlen)]
    predicted = [None for i in range(maxlen)]

    for i in range(length_of_sequences - maxlen + 1):
        # 最後の時系列データから未来を予測
        z_ = Z[-1:]  # Z[-1]とするとndim=2となるので ValueError: Cannot feed value of shape (25, 1) for Tensor 'Placeholder:0', which has shape '(?, 25, 1)' が起こる
        y_ = y.eval(session=sess, feed_dict={
            x: Z[-1:],
            n_batch: 1
        })
        # 予測結果を用いて新しい時系列データを生成
        sequence_ = np.concatenate(
            (z_.reshape(maxlen, n_in)[1:], y_), axis=0) \
            .reshape(1, maxlen, n_in)
        Z = np.append(Z, sequence_, axis=0)
        predicted.append(y_.reshape(-1))

    '''
    グラフで可視化
    '''
    plt.rc('font', family='serif')
    plt.figure()
    plt.ylim([-1.5, 1.5])
    plt.plot(toy_problem(T, ampl=0), linestyle='dotted', color='#aaaaaa')
    plt.plot(original, linestyle='dashed', color='black')
    plt.plot(predicted, color='black')
    plt.show()
