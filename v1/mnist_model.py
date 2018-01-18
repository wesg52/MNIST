import tensorflow as tf
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = "{}/run-{}".format(root_logdir, now)

#Mode
train = True

#hyperparameters
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#Model description

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name = 'y')

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

loss_summary = tf.summary.scalar('LOSS', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


#Execution
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/temp/data/")

n_epochs = 40
batch_size = 50

if train:
    with tf.Session() as sess:
        init.run()
        n_batches = mnist.train.num_examples // batch_size
        for epoch in range(n_epochs):
            for iteration in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                if iteration % 10 == 0:
                    summary_str = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + iteration
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: mnist.validation.images,
                                               y: mnist.validation.labels})

            print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
        save_path = saver.save(sess, "./my_model_final.ckpt")
else:
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")
        X_test = mnist.test.images
        y_test = mnist.test.labels
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Test Accuracy:", acc_test)

file_writer.close()
