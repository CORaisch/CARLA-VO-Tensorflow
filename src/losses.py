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
    # NOTE expected shapes of y_true and y_pred: [B,T,6]
    def call(y_true, y_pred):
        # extract translation and rotation sub-tensors, output shapes=[B,T,3]
        t_true = y_true[:,:,:3]; t_pred = y_pred[:,:,:3];
        r_true = y_true[:,:,3:]; r_pred = y_pred[:,:,3:];

        # compute squared differences (L2 metric), output shapes=[B,T,3]
        t_L2 = tf.square(t_pred - t_true)
        r_L2 = tf.square(r_pred - r_true)

        # reduce sum over batches and timesteps, output shape=[]
        loss_sum = tf.reduce_sum(t_L2 + k*r_L2)

        # return mean over all batches and timesteps, output shape=[]
        loss = loss_sum / tf.cast(tf.shape(y_true)[0], loss_sum.dtype)
        return loss
    return call
