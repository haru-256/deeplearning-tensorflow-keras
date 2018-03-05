import numpy as np
import tensorflow as tf

tf.set_random_seed(0)

'''
データの生成
'''
# XORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

'''
モデル設定
'''
# 入力データを設定
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])

# 入力層 - 隠れ層
# tf.truncated_normal()
# Tensorを正規分布で標準偏差の２倍までの値まででランダムに初期化する
W = tf.Variable(tf.truncated_normal(shape=[2, 2]))
b = tf.Variable(tf.zeros([2]))
# tf.nn.sigmoid() は要素ごとにシグモイド関数を適用する．
h = tf.nn.sigmoid(tf.matmul(x, W) + b)

# 隠れ層 - 出力層
V = tf.Variable(tf.truncated_normal([2, 1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

'''
モデル学習
'''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(4000):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })
    if epoch % 100 == 0:
        print('epoch: {0:3d} cross_entropy: {1:.4f}'
              .format(epoch,
                      cross_entropy.eval(session=sess, feed_dict={
                          x: X,
                          t: Y
                      })))

'''
学習結果の確認
'''
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
prob = y.eval(session=sess, feed_dict={
    x: X
})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)
