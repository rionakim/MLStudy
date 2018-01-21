import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

#최초 시작점을 설정.
W = tf.Variable(-3.0)

#hypothesis
hypothesis = X * W

#cost/loss func.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

##minimize1: 앞에서 배운 미분...식ㄱ으로 사용할 경우의 수식
#learning_rate = 0.1
#gradient = tf.reduce_mean((W * X - Y)*X)
#descent = W - learning_rate * gradient
#update = W.assign(descent)

#minimize2: 함수로 퉁치기 1번을 함수로 제공하는것.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

#launch the graph in a session
sess = tf.Session()

sess.run(tf.global_variables_initializer())


for step in range (100):
    print(step, sess.run(W))
    sess.run(train)
