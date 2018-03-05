import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(0)

M = 2      # 入力データの次元
K = 3      # クラス数
n = 100    # クラスごとのデータ数
N = n * K  # 全データ数

'''
データの生成
'''
X1 = np.random.randn(n, M) + np.array([0, 10])  # 中心(0, 10)のデータを生成
X2 = np.random.randn(n, M) + np.array([5, 5])  # 中心(5, 5)のデータを生成
X3 = np.random.randn(n, M) + np.array([10, 0])  # 中心(5, 5)のデータを生成
Y1 = np.array([[1, 0, 0] for _ in range(n)])
Y2 = np.array([[0, 1, 0] for _ in range(n)])
Y3 = np.array([[0, 0, 1] for _ in range(n)])

# クラスごとのデータを連結して１つのデータにする
X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

"""
モデル設定
"""
# W行列の形に注意
W = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
# tf.matmul(x, W) + bの+bはブロードキャストによって実現
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y),
#                               reduction_indices=[1]))
# 今はreduction_indicesではなくaxisを使うよう
cross_entropy_n = - tf.reduce_sum(t * tf.log(y), axis=[1])
cross_entropy = tf.reduce_mean(cross_entropy_n)

# 最適化手法にSGDを採用するので勾配降下法を適用
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 予測結果と等しいかどうかを見る
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(t, axis=1))

'''
モデル学習
'''
# 初期化
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

batch_size = 50  # ミニバッチサイズ
n_batches = N // batch_size

# ミニバッチ学習
for epoch in range(20):
    # sklearn.utils.shuffleは引数に複数のリストをとルことができ，かつ
    # 引数にとったリストを各要素でランダムに並べ替えてくれる．
    X_, Y_ = shuffle(X, Y)
    # バッチサイズごとに1エポック分学習
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })

'''
学習結果の確認
'''
X_, Y_ = shuffle(X, Y)

# 10個のデータが正しく分類されているかを確認
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X_[0:10],
    t: Y_[0:10]
})
prob = y.eval(session=sess, feed_dict={
    x: X_[0:10]
})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)
