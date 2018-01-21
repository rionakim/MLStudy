#multi variables linear regression lab.
#hypothesis using matrix
import tensorflow as tf

#x1, x2, x3를 따로 만들 필요가 없어진다.

x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.],[73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]


#set placehoder for a tensor
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

#set variables(변수)
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis = x1 * w1 + x2 * w2 + x3 + w3 + b
hypothesis = tf.matmul(X, W) + b

#set cost function : cost func.이 최소화 되는 값.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#gradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={X:x_data, Y:y_data})

    if step % 10 == 0:
        print(step, "cost:", cost_val, "\nprediction:", hy_val)
