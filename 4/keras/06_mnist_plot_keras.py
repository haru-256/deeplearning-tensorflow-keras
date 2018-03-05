import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras import backend as K
from keras.initializers import TruncatedNormal
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(123)

# generate data
mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n = len(mnist.data)
N = 30000  # Using part of MNIST
N_train = 20000
N_validation = 4000
indices = np.random.permutation(n)[:N]  # random sampling

X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, train_size=N_train)
X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X_train, Y_train, test_size=N_validation)

# Model setting
n_in = len(X[0])  # 784
n_hiddens = [200, 200, 200]
n_out = len(Y[0])  # 10
p_drop = 0.5
activation = 'relu'


# 正則化に適用する関数は自由に定義できるが，引数にshapeをとる．
def weight_variable(shape):
    # バックエンドKはtensorflowとなる．
    return K.truncated_normal(shape, stddev=0.01)
    # return np.random.normal(scale=0.01, size=shape)


model = Sequential()
for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
    # kernel_regularizerは重み行列に適用する正則化関数を指定できる．
    model.add(Dense(n_hiddens[i], input_dim=input_dim,
                    kernel_initializer=weight_variable))
    # 上のコードは以下の二行のコードと同意
    # model.add(Dense(n_hiddens[i], input_dim=input_dim,
    #                kernel_initializer=TruncatedNormal(stddev=0.01)))
    model.add(Activation(activation))
    model.add(Dropout(p_drop))

model.add(Dense(n_out, input_dim=n_hiddens[-1],
                kernel_initializer=weight_variable))
model.add(Activation('softmax'))

# metrics引数を与えることで，訓練時にlossに加えてaccuracyを出してくれる．
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

'''
モデル学習
'''
epochs = 50
batch_size = 200

# kerasではバリデーションデータを与えることで，
# 学習の途中結果にvalidation_dataに対する損失とaccuracyを出してくれる．
hist = model.fit(X_train, Y_train, epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_validation, Y_validation))

# 学習の進み具合を可視化
# model.fitの戻り値に学習の途中結果が入っている．
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(range(epochs), val_acc, label='acc', color='black')
plt.xlabel('epochs')
plt.show()
# plt.savefig('mnist_keras.eps')

'''
予測精度の評価
'''
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
