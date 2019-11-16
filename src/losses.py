import tensorflow as tf
import sys

def debug_loss(y_true, y_pred):
    tf.print("y_true:", y_true, output_stream=sys.stdout)
    tf.print("y_pred:", y_pred, output_stream=sys.stdout)
    diff = y_true - y_pred
    tf.print("diff:", diff, output_stream=sys.stdout)
    square = tf.square(diff)
    tf.print("square:", square, output_stream=sys.stdout)
    loss = tf.reduce_mean(square, axis=1)
    tf.print("loss:", loss, output_stream=sys.stdout)
    return loss

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred), axis=1)
