import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(1234)

'''
データ生成
'''
N = 300  # 全データ数
X, y = datasets.make_moons(N, noise=0.3)
Y = y.reshape(N, 1)  # Tensorに合うようにデータの形状を変更

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size=0.8,
                                                    test_size=0.2)


'''
モデル生成
'''
num_hidden = 3  # 隠れ層の次元数
# num_hidden = 2

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])

# 入力層 - 隠れ層
W = tf.Variable(tf.truncated_normal([2, num_hidden]))
b = tf.Variable(tf.zeros([num_hidden]))
h = tf.nn.sigmoid(tf.matmul(x, W) + b)

# 隠れ層 - 出力層
V = tf.Variable(tf.truncated_normal([num_hidden, 1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

'''
誤差関数の設定
'''
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
"""
最適化手法に勾配法を選択
"""
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# tf..cast(correct_prediction, dtype=tf.float32)は
# tf.to_float(correct_prediction)と同じ動き．
# しかし，tf.to_float()はtf.float32にしか変換されない．
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

'''
モデル学習
'''
batch_size = 20
n_batches = N // batch_size

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(500):
    X_, Y_ = shuffle(X_train, Y_train)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })
    print('epoch:{0:3d} accuracy: {1:.2f}'
          .format(epoch, accuracy.eval(session=sess,
                                       feed_dict={
                                           x: X_train,
                                           t: Y_train
                                       })))

'''
予測精度の評価
'''
accuracy_rate = accuracy.eval(session=sess, feed_dict={
    x: X_test,
    t: Y_test
})
print('accuracy: ', accuracy_rate)
