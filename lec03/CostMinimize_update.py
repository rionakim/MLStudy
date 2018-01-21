import tensorflow as tf
#import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#hypothesis
hypothesis = X * W

#cost/loss func.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#minimize1: 앞에서 배운 미분...식ㄱ으로 사용할 경우의 수식
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y)*X)
descent = W - learning_rate * gradient
update = W.assign(descent)

#session
sess = tf.Session()

sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []


for step in range (21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

#show graph to me!
#plt.plot(W_val, cost_val)
#plt.show()
