import tensorflow as tf
import numpy as np
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = "{}/run-{}".format(root_logdir, now)

#Mode
train = False

#network hyperparameters
n_outputs = 10

height = 28
width = 28
channels = 1

X = tf.placeholder(tf.float32, shape=(None, height, width, channels), name="X")
y = tf.placeholder(tf.int64, shape=(None), name = 'y')
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope("convnet"):
    conv1 = tf.layers.conv2d(X, filters=6, kernel_size=5, strides=[1,1], padding="VALID", name="conv1")
    conv1_activated = tf.nn.elu(conv1)

    conv2 = tf.layers.conv2d(conv1_activated, filters=16, kernel_size=1,
                             strides=[1,1], padding="SAME", name="conv2")
    conv2_activated = tf.nn.elu(conv2)

    conv3 = tf.layers.conv2d(conv2_activated, filters=16, kernel_size=3,
                             strides=[1,1], padding="SAME", name="conv3")

    pool4 = tf.nn.avg_pool(conv3, ksize=[1,2,2,1],
                                   strides=[1,2,2,1], padding="VALID", name="pool4")
    pool4_activated = tf.nn.elu(pool4)

    conv5 = tf.layers.conv2d(pool4_activated, filters=80, kernel_size=3,
                             strides=[1,1], padding="SAME", name="conv5")

    pool5 = tf.nn.avg_pool(conv5, ksize=[1,2,2,1],
                                   strides=[1,2,2,1], padding="VALID", name="pool5")
    pool5_activated = tf.nn.elu(pool5)

    conv6 = tf.layers.conv2d(pool5_activated, filters=80, kernel_size=3,
                             strides=[1,1], padding="SAME", name="conv6")
    conv6_activated = tf.nn.elu(conv6)

    conv6_flat = tf.reshape(conv6_activated, [-1, 6 * 6 * 80])

    fc7 = tf.layers.dense(conv6_flat, 256, name="fc7")
    fc7_act = tf.nn.elu(fc7)
    fc7_drop = tf.layers.dropout(fc7_act, 0.4, training=training)

    fc8 = tf.layers.dense(fc7_drop, 64, name="fc8")
    fc8_act = tf.nn.elu(fc8)
    fc8_drop = tf.layers.dropout(fc8_act, 0.2, training=training)

    logits = tf.layers.dense(fc8_drop, n_outputs, name='outputs')

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')


learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

acc_val = 0.0

#acc_summmary = tf.summary.scalar('Val Accuracy', acc_val)
loss_summary = tf.summary.scalar('LOSS', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#Execution
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/temp/data/")

n_epochs = 20
batch_size = 20

if train:
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.Session() as sess:
        init.run()
        n_batches = mnist.train.num_examples // batch_size
        X_val = np.array(mnist.validation.images).reshape(5000, 28, 28, 1)
        for epoch in range(n_epochs):
            for iteration in range(n_batches):
                X_batch_flat, y_batch = mnist.train.next_batch(batch_size)
                X_batch = np.array(X_batch_flat).reshape(batch_size, 28, 28, 1)
                if iteration % 20 == 0:
                    summary_str = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + iteration
                    file_writer.add_summary(summary_str, step)
                sess.run([training_op, extra_update_ops],
                         feed_dict={training: True, X: X_batch, y: y_batch})

            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_val,
                                               y: mnist.validation.labels})
            print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
        save_path = saver.save(sess, "./my_model_final.ckpt")
else:
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")
        X_test = np.array(mnist.test.images).reshape(10000, 28, 28, 1)
        y_test = mnist.test.labels
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Test Accuracy:", acc_test)

file_writer.close()
