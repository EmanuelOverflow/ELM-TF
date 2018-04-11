import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
from models import elm

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 5000, "Batch size")
flags.DEFINE_integer("height", 28, "Frame height")
flags.DEFINE_integer("width", 28, "Frame width")
flags.DEFINE_integer("in_channels", 1, "Frame channels")
flags.DEFINE_integer("num_classes", 10, "Number of classes")
flags.DEFINE_integer("num_hidden", 512, "Number of hidden units")

# Input settings
flags.DEFINE_string("dataset_path", "MNIST_data",
                    "Path to mnist dataset file")

FLAGS = flags.FLAGS
FLAGS._parse_flags()
params_str = ""
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    params_str += "{} = {}\n".format(attr.upper(), value)
    print("{} = {}".format(attr.upper(), value))
print("")


def compute_loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(
        cross_entropy
    )

    tf.summary.scalar("loss", cross_entropy_mean)

    return cross_entropy_mean


def compute_accuracy(logits, labels):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy


def create_variable(name, shape, dtype=tf.float32, initializer=tf.random_normal_initializer()):
    with tf.device("/cpu:0"):
        var = tf.get_variable(name, dtype=dtype, shape=shape,
                              initializer=initializer)
    return var


def prepare_dataset():
    mnist = input_data.read_data_sets(FLAGS.dataset_path, one_hot=True)
    return mnist.train, mnist.test


def elm_mnist_example():
    train_data, test_data = prepare_dataset()
    input_size = FLAGS.height * FLAGS.width * FLAGS.in_channels

    x = tf.placeholder(tf.float32, shape=[None, input_size], name="input_placeholder")
    y = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name="labels_placeholder")
    beta = tf.placeholder(tf.float32, shape=[FLAGS.num_hidden, FLAGS.num_classes], name="beta_placeholder")

    weights = create_variable("wh1", shape=[input_size, FLAGS.num_hidden])
    bias = create_variable("bh1", shape=[FLAGS.num_hidden])

    train_op = elm.train(x, y, weights, bias, name="train_elm")

    logits = elm.inference(x, weights, bias, beta, name="inference_elm")
    loss = compute_loss(logits=logits, labels=y)
    acc = compute_accuracy(logits=logits, labels=y)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        print("Training...")
        train_samples, train_labels = train_data.next_batch(FLAGS.batch_size)
        start = time.time()
        beta_out = sess.run(train_op, feed_dict={
            x: train_samples,
            y: train_labels
        })

        beta_out = np.array(beta_out)
        total_time = time.time() - start

        out_loss, out_acc = sess.run([loss, acc], feed_dict={
            x: train_samples,
            y: train_labels,
            beta: beta_out
        })

        print("Loss: {} - Acc: {} - Time elapsed: {:.4f}s".format(out_loss, out_acc, total_time))

        print("Testing")
        test_samples, test_labels = test_data.next_batch(FLAGS.batch_size)
        out_loss, out_acc = sess.run([loss, acc], feed_dict={
            x: test_samples,
            y: test_labels,
            beta: beta_out
        })
        print("Loss: {} - Acc: {}".format(out_loss, out_acc))


def main(_):
    elm_mnist_example()


if __name__ == '__main__':
    tf.app.run()
