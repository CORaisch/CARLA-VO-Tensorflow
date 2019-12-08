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

# NOTE implementation of the weigted MSE loss used at DeepVO: https://www.cs.ox.ac.uk/files/9026/DeepVO.pdf
def weighted_mse(k=100):
    def call(y_true, y_pred):
        # ## beg DEBUG
        # tf.print("shape y_true:", tf.shape(y_true), output_stream=sys.stdout)
        # tf.print("shape y_pred:", tf.shape(y_pred), output_stream=sys.stdout)
        # ## end DEBUG

        # extract translation and rotation sub-tensors
        t_true = y_true[:,:,:3]; t_pred = y_pred[:,:,:3];
        r_true = y_true[:,:,3:]; r_pred = y_pred[:,:,3:];

        # ## beg DEBUG
        # tf.print("shape t_true:", tf.shape(t_true), output_stream=sys.stdout)
        # tf.print("shape t_pred:", tf.shape(t_pred), output_stream=sys.stdout)
        # tf.print("shape r_true:", tf.shape(r_true), output_stream=sys.stdout)
        # tf.print("shape r_pred:", tf.shape(r_pred), output_stream=sys.stdout)
        # ## end DEBUG

        # compute squared differences (L2 metric)
        t_L2 = tf.square(t_pred - t_true)
        r_L2 = tf.square(r_pred - r_true)

        # reduce sum over batches and timesteps
        loss_sum = tf.reduce_sum(t_L2 + k*r_L2)

        # return mean over all batches and timesteps
        loss = loss_sum / tf.cast(tf.shape(y_true)[0], loss_sum.dtype)
        tf.print("loss:", loss, output_stream=sys.stdout)
        return loss
    return call
