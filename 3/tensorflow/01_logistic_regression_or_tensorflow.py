"""
01_logistic_regression_or_tensorflow.py
"""
import numpy as np
import tensorflow as tf
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
if os.path.exists(LOG_DIR) is False:
    os.mkdir(LOG_DIR)

# 以下の乱数設定は何のために使うのか?
tf.set_random_seed(0)  # 乱数シード

# 何かの変数(パラメータ)はVariable
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

# 何らかの変数だが，データなど決まったものが入るのはtf.placeholder
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w) + b)

"""
誤差関数の定義
"""
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

"""
最適化手法の定義
"""
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)


# ORゲートの入力値
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

"""
セッションの初期化
"""
# 初期化

#  tfでtf.Variableを使う場合は初期化を行わなければならない
init = tf.global_variables_initializer()
sess = tf.Session()
tf.summary.FileWriter(LOG_DIR, sess.graph)  # TensorBoardに対応
sess.run(init)

# 学習
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

"""
学習結果の確認
"""

classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
# 上と同じことを以下は行う
"""
classified = sess.run(correct_prediction, feed_dict={
    x: X,
    t: Y
})
"""

prob = y.eval(session=sess, feed_dict={
    x: X
})

# 上と同じことを以下は行う
"""
prob = sess.run(y, feed_dict={
    x: X
})
"""
print('classified:')
print(classified)
print()
print('output probability:')
print(prob)
