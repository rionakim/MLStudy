#compute_gradient and apply_gradient
import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

#최초 시작점을 설정.
W = tf.Variable(5.)

#hypothesis
hypothesis = X * W

#manual grad.
gradient = tf.reduce_mean((W * X - Y) * X) *2

#cost/loss func.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#minimize2: 함수로 퉁치기 1번을 함수로 제공하는것.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

#get grad.
gvs = optimizer.compute_gradients(cost)

#apply grad.
apply_gradients = optimizer.apply_gradients(gvs)


#launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range (100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

    #gradient : 손으로 계산한값, W, 컴퓨터가 계산한값과 결과.(W)
