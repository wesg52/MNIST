import tensorflow as tf
from datetime import datetime
from functools import partial

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = "{}/run-{}".format(root_logdir, now)

#Mode
train = False

#network hyperparametes
n_inputs = 28*28

n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 128
n_hidden4 = 64
n_hidden5 = 32
n_outputs = 10

#Model description

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name = 'y')
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope("dnn"):
    batch_norm_wrapper = partial(tf.layers.batch_normalization,
                                 training=training, momentum=0.9)
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
    bn1 = batch_norm_wrapper(hidden1)
    bn1_activated = tf.nn.elu(bn1)
    hidden1_drop = tf.layers.dropout(bn1_activated, 0.5, training=training)

    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2")
    bn2 = batch_norm_wrapper(hidden2)
    bn2_activated = tf.nn.elu(bn2)
    hidden2_drop = tf.layers.dropout(bn2_activated, 0.5, training=training)

    hidden3 = tf.layers.dense(hidden2_drop, n_hidden3, name="hidden3")
    bn3 = batch_norm_wrapper(hidden3)
    bn3_activated = tf.nn.elu(bn3)
    hidden3_drop = tf.layers.dropout(bn3_activated, 0.4, training=training)

    hidden4 = tf.layers.dense(hidden3_drop, n_hidden4, name="hidden4")
    bn4 = batch_norm_wrapper(hidden4)
    bn4_activated = tf.nn.elu(bn4)
    hidden4_drop = tf.layers.dropout(bn4_activated, 0.3, training=training)

    hidden5 = tf.layers.dense(hidden4_drop, n_hidden4, name="hidden5")
    bn5 = batch_norm_wrapper(hidden5)
    bn5_activated = tf.nn.elu(bn5)
    hidden5_drop = tf.layers.dropout(bn5_activated, 0.2, training=training)

    logits_before_bn = tf.layers.dense(hidden5_drop, n_outputs, name='outputs')
    logits = batch_norm_wrapper(logits_before_bn)

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

loss_summary = tf.summary.scalar('LOSS', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


#Execution
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/temp/data/")

n_epochs = 80
batch_size = 50

if train:
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
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
                sess.run([training_op, extra_update_ops],
                         feed_dict={training: True, X: X_batch, y: y_batch})

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
