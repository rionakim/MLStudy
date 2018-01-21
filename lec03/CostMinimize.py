import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

#hypothesis
hypothesis = X * W

#cost/loss func.

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#session
sess = tf.Session()

sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []


for i in range (-30, 50):
        feed_W = i * 0.08
        curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
        W_val.append(curr_W)
        cost_val.append(curr_cost)

#show graph to me!
plt.plot(W_val, cost_val)
plt.show()

#graph as  cost(W) : Y, W : X
