import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
from getCategories import Category

def next_batch(num_split, feature, labels):
    ind = np.arange(0, len(feature))
    np.random.shuffle(ind)
    ind = ind[:num_split]
    feature_shuffle = [feature[i] for i in ind]
    labels_shuffle = [labels[i] for i in ind]

    return np.asarray(feature_shuffle), np.asarray(labels_shuffle)

mnist = np.load("../Dataset/DataSet.npy", allow_pickle=True)
random.shuffle(mnist)

x_train = np.array([mnist[i].features for i in range(0, int(len(mnist) * 0.8))])
x_test = np.array([mnist[i].features for i in range(int(len(mnist) * 0.8), len(mnist))])
y_train = np.array([mnist[i].label for i in range(0, int(len(mnist) * 0.8))])
y_test = np.array([mnist[i].label for i in range(int(len(mnist) * 0.8), len(mnist))])

# Parameters
learning_rate = 0.0005
training_epochs = 2500
batch_size = 128
display_step = 10

x = tf.placeholder(tf.float32, [None, 20])
y = tf.placeholder(tf.float32, [None, 2])

W1 = tf.Variable(tf.random_normal([20, 10], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([10]), name='b1')
W2 = tf.Variable(tf.random_normal([10, 2], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([2]), name='b2')

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

log = tf.matmul(hidden_out, W2) + b2

pred = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=log))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

acc_ = tf.summary.scalar('acc', accuracy)
loss_summary = tf.summary.scalar('loss', cost)


with tf.Session() as sess:
    writer = tf.summary.FileWriter('../logs/NeuralNetwork', sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(x_train)/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            _, c, acc, acc_summary, loss_ = sess.run([optimizer, cost, accuracy, acc_, loss_summary], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch
        if (epoch+1) % display_step == 0:
            writer.add_summary(acc_summary, epoch)
            writer.add_summary(loss_, epoch)
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "acc=", "{:.5}".format(acc))
    writer.close()
    print("Optimization Finished!")

    saver = tf.train.Saver()
    saver.save(sess, "../model/NeuralNetwork/model")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))