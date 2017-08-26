import numpy as np
import tensorflow as tf
#数据准备
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
X_train = mnist.train.images
Y_train = mnist.train.labels
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
#搭建网络
#第一卷积层
W_conv1 = tf.Variable(tf.random_normal([5,5,1,32],stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
conv_1 = tf.nn.relu(tf.nn.conv2d(tf.reshape(xs,[-1,28,28,1]),W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
pooling_1 = tf.nn.max_pool(conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #13*13
#第二卷积层
W_conv2 = tf.Variable(tf.random_normal([5,5,32,64],stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
conv_2 = tf.nn.relu(tf.nn.conv2d(pooling_1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
pooling_2= tf.nn.max_pool(conv_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #7*7
#全连接层1
W_3 = tf.Variable(tf.random_normal([7*7*64,1024],stddev=0.1))
b_3 = tf.Variable(tf.constant(0.1,shape=[1024]))
l_3_0 = tf.reshape(pooling_2,[-1,7*7*64])
l_3_1 = tf.matmul(l_3_0,W_3)+b_3
L_3 = tf.nn.relu(l_3_1)
#全连接层2
W_3 = tf.Variable(tf.random_normal([1024,1024],stddev=0.1))
b_3 = tf.Variable(tf.constant(0.1,shape=[1024]))
L_4 = tf.nn.sigmoid(tf.matmul(L_3,W_3)+b_3)
# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(L_4,keep_prob)
#output
W_out = tf.Variable(tf.random_normal([1024,10],stddev=0.1))
b_out = tf.Variable(tf.constant(0.1,shape=[10]))
y_out = tf.nn.softmax(tf.matmul(h_fc1_drop,W_out)+b_out)
#训练过程
loss = -tf.reduce_sum(ys*tf.log(y_out))
train_step = tf.train.GradientDescentOptimizer(0.00005).minimize(loss)
correct_p = tf.equal(tf.argmax(y_out,1),(tf.argmax(ys,1)))
accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("开始训练:")
for i in range(20000):
   batch = mnist.train.next_batch(50)
   sess.run(train_step,feed_dict={xs:batch[0],ys:batch[1],keep_prob:0.5})
   if i%200 == 0:
       print(i/200)
       print(sess.run(accuracy,feed_dict={xs:batch[0],ys:batch[1],keep_prob:1.0}))
print("进行测试集测试:")
testbatch = mnist.test.next_batch(1000)
print(sess.run(accuracy,feed_dict={xs:testbatch[0],ys:testbatch[1],keep_prob:1.0}))