import tensorflow as tf
import numpy as np
import os
import src.config as config

# enable eager execution
tf.compat.v1.enable_eager_execution()

########################### SAMPLE CONFIG ###########################
# TRAINED_MODEL = "models/deepvo__in2_seqlen2_imw256_imh256_imc1_out6.h5"
TRAINED_MODEL = "models/deepvo_trained-20ep.h5"
LEFT_INPUT    = "/home/claudio/Datasets/CARLA/sequence_00/rgb/left/images"
# RIGHT_INPUT   = "/home/claudio/Datasets/CARLA/sequence_01/rgb/right/images"
PRED_FILE     = "predictions/pred_deepvo.txt"
SHAPE         = [256, 256, 1]
STEREO        = False
SEQ_LEN       = 2


########################### HELPERS ###########################
def load_and_preprocess_image(path, shape):
    image = tf.io.read_file(path)
    return preprocess_image(image, shape)

def preprocess_image(image, shape):
    # decod raw image
    image = tf.image.decode_image(image, channels=shape[2])
    # TODO filter images here
    # resize images
    image = tf.image.resize(image, [shape[0], shape[1]])
    # normalize final image to range [0,1]
    image /= 255.0
    # return final image
    return image

def write_pred_to_file(f, pred):
    fstr = str(pred[0]) + ' ' + str(pred[1]) + ' ' + str(pred[2]) + ' ' + str(pred[3]) + ' ' + str(pred[4]) + ' ' + str(pred[5]) + '\n'
    f.write(fstr)

# TODO visualize input images in grid
# left_i   | right_i
# left_i+1 | right_i+1
# ...
def visualize_inputs(images):
    print("[TODO] visualize input images")
    print(images.keys())
    # TODO rearrange images
    # TODO visualize in grid

########################### CODE ###########################
# force tensorflow to throw its inital messages on the very beginning of that script
tf.config.experimental_list_devices()

## load model
print("[INFO] loading model from \'{}\'...".format(TRAINED_MODEL), end='', flush=True)
model = tf.keras.models.load_model(TRAINED_MODEL)
print(" done")
print("[INFO] information about the DNN model thats going to be evaluated:")
model.summary()

## get number of images of current observation
nimages = len([name for name in os.listdir(LEFT_INPUT) if os.path.isfile(os.path.join(LEFT_INPUT,name))])

## load input-images
with open(PRED_FILE, 'w') as predf:
    for time in range(0,nimages-(SEQ_LEN-1)):
        # load input images for current timestep
        # TODO layernames must be computed from config file to be consistent with rest of data
        inputs = {}
        for i in range(0,SEQ_LEN):
            # load left image
            lname = 'rgb_left_'+str(i)
            lpath = os.path.join(LEFT_INPUT, '%010d' % (time+i) + '.png')
            inputs[lname] = tf.stack([load_and_preprocess_image(lpath, SHAPE)])
            # load right image
            if STEREO:
                rname = 'rgb_right_'+str(i)
                rpath = os.path.join(RIGHT_INPUT, '%010d' % (time+i) + '.png')
                inputs[rname] = tf.stack([load_and_preprocess_image(rpath, SHAPE)])

        # TODO visualize images
        # visualize_inputs(inputs)

        ## get model predictions for the input-images
        pred = model.predict(inputs, batch_size=1, verbose=0)
        assert(pred.shape==(1,6)) # prediction must always be 1x6 vector

        ## print prediction (serves as status report)
        print("[INFO] Prediction t={}: {}".format(time, pred[0]))

        ## write prediction to file
        write_pred_to_file(predf, pred[0])

#### LATER TODOs:
# 1) loop above code over complete sequence
# 2) convert predictions from euler to 3x4 matrices
# 3) write predictions to file
