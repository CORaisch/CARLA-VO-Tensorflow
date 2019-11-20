## sample call: python predict_sequence.py ~/Datasets/CARLA/sequence_00/rgb/left/images=rgb_left models/deepvo_trained.h5 configs/local.conf --out deepvo_pred.txt

import tensorflow as tf
import numpy as np
import os, argparse, zipfile
import src.config as config

# enable eager execution
tf.compat.v1.enable_eager_execution()

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

def parse_args():
    # create argparse instance
    argparser = argparse.ArgumentParser(description="takes a sequence of images and computes the predictions on the passed, trained DNN.")
    # add positional arguments
    argparser.add_argument('sequence', type=str, nargs='*', metavar="\'PATH/TO/IMAGES=LAYERNAME\'", help="sequence on which to compute the predictions")
    argparser.add_argument('model', type=str, help="file of trained model in .h5 format")
    argparser.add_argument('config', type=str, help="Config file needs to be passed in order to specify the training setup. See 'configs/sample.conf' for an template.")
    # add optional arguments
    argparser.add_argument('--out', '-o', type=str, default=None, help="file where the predictions are stored in (as .txt file). By default it will be saved in the predicitons subdir with the same name as the model file with added \'_pred\' suffix.")
    argparser.add_argument('--verbose', '-v', type=bool, default=False, help="set to show more information.")
    argparser.add_argument('--kitti', '-kitti', action='store_true', help="set to True if predicting on KITTI sequence")
    # parse args
    args = argparser.parse_args()
    # preprocess --out argument
    if args.out == None:
        base = args.model.split('/')[-1]
        extension = base.split('.')[-1]
        args.out = 'predictions/' + base.split(extension)[0] + 'txt'
    # postprocess sequence argument
    args.sequence = { pair.split('=')[0] : pair.split('=')[1] for pair in args.sequence }
    # return parsed arguments
    return args

########################### CODE ###########################
## force tensorflow to throw its inital messages on the very beginning of that script
tf.config.experimental_list_devices()

## parse arguments
args = parse_args()
conf = config.Config(args.config)

## load model
print("[INFO] loading model from \'{}\'...".format(args.model), end='', flush=True)
model = tf.keras.models.load_model(args.model)
print(" done")
print("[INFO] information about the DNN model thats going to be evaluated:")
model.summary()

## get number of images of current observation
seqpath = list(args.sequence.keys())[0]
nimages = len([name for name in os.listdir(seqpath) if os.path.isfile(os.path.join(seqpath, name))])

## load input-images
with open(args.out, 'w') as predf:
    for time in range(0, nimages-(conf.seq_len-1)):
        # load input images for current timestep
        inputs = {}
        for t in range(0, conf.seq_len): # for each timestep
            for i in range(0,len(args.sequence)): # load the appropriate ammount of images (mono, stereo, etc.)
                # load image
                seq = list(args.sequence.keys())[i]
                lname = args.sequence[seq] + '_' + str(t)
                if args.kitti:
                    pstr = '%06d'
                else:
                    pstr = '%010d'
                lpath = os.path.join(seq, pstr % (time+t) + '.png')
                inputs[lname] = tf.stack([load_and_preprocess_image(lpath, conf.image_shape)])

        # TODO visualize images
        # visualize_inputs(inputs)

        ## get model predictions for the input-images
        pred = model.predict(inputs, batch_size=1, verbose=0)
        assert(pred.shape==(1,6)) # prediction must always be 1x6 vector

        ## print prediction (serves as status report)
        if args.verbose:
            print("[INFO] Prediction t={}: {}".format(time, pred[0]))
        else:
            progress = int(100.0*(time/nimages))
            print("[INFO] Progress {}%".format(progress), end='\r', flush=True)

        ## write prediction to file
        write_pred_to_file(predf, pred[0])

#### LATER TODOs:
# 1) loop above code over complete sequence
# 2) convert predictions from euler to 3x4 matrices
# 3) write predictions to file
