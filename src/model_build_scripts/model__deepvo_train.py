## Script creates the DNN from DeepVO that takes Images as Input, stacks them along the Channel Dim and applies Convolutions and LSTMs. It has 6 output Neurons.

import tensorflow as tf
import os, zipfile, argparse
import src.config as config

def parse_args():
    argparser = argparse.ArgumentParser(description="This script generates the RCNN from the DeepVO paper.")
    argparser.add_argument('config', help="same config file as used for training with 'train_sequence.py'. Config file is required to set the input layers names as 'train_tfrec.py' expects it. Names will be generated from 'DATASET_FILES' option, so ensure to use this option consistently.")
    argparser.add_argument('--model_out', '-out', default='', help="path to where the model should be saved.")
    return argparser.parse_args()

# NOTE 'arch_file' holds the path to a sequence zip archive
# NOTE 't_inputs' defines the number of timesteps presented the DNN as input
#       -> recording N images at a single point in time then setting 't_inputs' to M will result in N*M input images to the DNN
def make_layernames(arch_file, t_inputs):
    layernames = []
    ## get filenames from archive
    # open archive
    fz = zipfile.ZipFile(arch_file, 'r')
    # extract names of image files
    for f in fz.namelist():
        # skip labels file
        if f == 'labels.npz':
            continue
        # cut away the file extension
        name = f.split('.')[0]
        # create enumerate input layernames
        for t in range(t_inputs):
            layernames.append(name + '_' + str(t))
    # close archive
    fz.close()
    ## return list of input layernames
    return layernames

# NOTE 'layernames' holds the names of the input layers => size of 'layernames' == size of input layers
# NOTE 'shape' is tuple that holds resolution of input images, assuming all images to have the same size
def create_model(layernames, shape):
    ## INFO architecture from DeepVO (https://www.cs.ox.ac.uk/files/9026/DeepVO.pdf)

    ## substitutions
    # activations
    relu = tf.keras.activations.relu
    linear = tf.keras.activations.linear
    # optimizer
    adam = tf.keras.optimizers.Adam()
    # layers
    Conv2D = tf.keras.layers.Conv2D
    TimeDistributed = tf.keras.layers.TimeDistributed
    Flatten = tf.keras.layers.Flatten
    LSTM = tf.keras.layers.LSTM
    Dense = tf.keras.layers.Dense

    ## build input layers
    # collect all input images in list
    input_layers   = {}
    for layername in layernames:
        # input layers expect subsequenced data: (BATCH, SUBSEQ, imH, imW, imC) [BATCH=None will be added by keras automatically]
        input_layers[layername] = tf.keras.layers.Input(shape=(None, shape[0], shape[1], shape[2]), name=layername)
    # stack input layers s.t. all input images are stacked together at their channel-dimension
    # TODO verify that stacking batches (of sequences) of images results in the same tensor as stacking the two input images beforehand
    input_stacked = tf.keras.layers.concatenate(inputs=input_layers.values(), axis=-1, name='stack_inputs')

    ## add FlowNet convolution layers: https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/flownet.pdf
    # Conv1: kernel=7x7, padding=zeropadding(3,3), stride=(2,2), channels=64, activation=ReLu
    conv1   = TimeDistributed(Conv2D(64, 7, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv1')(input_stacked)
    # Conv2: kernel=5x5, padding=zeropadding(2,2), stride=(2,2), channels=128, activation=ReLu
    conv2   = TimeDistributed(Conv2D(128, 5, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv2')(conv1)
    # Conv3: kernel=5x5, padding=zeropadding(2,2), stride=(2,2), channels=256, activation=ReLu
    conv3   = TimeDistributed(Conv2D(256, 5, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv3')(conv2)
    # Conv3_1: kernel=3x3, padding=zeropadding(1,1), stride=(1,1), channels=256, activation=ReLu
    conv3_1 = TimeDistributed(Conv2D(256, 3, padding='same', strides=1, activation=relu, data_format='channels_last'), name='conv3_1')(conv3)
    # Conv4: kernel=3x3, padding=zeropadding(1,1), stride=(2,2), channels=512, activation=ReLu
    conv4   = TimeDistributed(Conv2D(512, 3, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv4')(conv3_1)
    # Conv4_1: kernel=3x3, padding=zeropadding(1,1), stride=(1,1), channels=512, activation=ReLu
    conv4_1 = TimeDistributed(Conv2D(512, 3, padding='same', strides=1, activation=relu, data_format='channels_last'), name='conv4_1')(conv4)
    # Conv5: kernel=3x3, padding=zeropadding(1,1), stride=(2,2), channels=512, activation=ReLu
    conv5   = TimeDistributed(Conv2D(512, 3, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv5')(conv4_1)
    # Conv5_1: kernel=3x3, padding=zeropadding(1,1), stride=(1,1), channels=512, activation=ReLu
    conv5_1 = TimeDistributed(Conv2D(512, 3, padding='same', strides=1, activation=relu, data_format='channels_last'), name='conv5_1')(conv5)
    # Conv6: kernel=3x3, padding=zeropadding(1,1), stride=(2,2), channels=1024, activation=ReLu
    conv6 = TimeDistributed(Conv2D(1024, 3, padding='same', strides=2, activation=linear, data_format='channels_last'), name='conv6')(conv5_1)

    ## add LSTMs to learn a mapping from the extracted features to 6 dof pose
    # flatten convolution output:
    # conv_6: (BATCH, SUBSEQ, shape[0]/64, shape[1]/64, 1024) -> (BATCH, SUBSEQ, shape[0]/64 * shape[1]/64 * 1024)
    flatten = TimeDistributed(Flatten(data_format='channels_last'), name='flatten_features')(conv6)
    # add first LSTM layer with 1000 units
    # TODO figure out: should LSTM1 return its state  to initialize LSTM2?
    lstm1, state_h, state_c = LSTM(1000, time_major=False, return_state=True, stateful=False, return_sequences=True, name='LSTM1')(flatten)
    # lstm1 = LSTM(1000, time_major=False, stateful=False, return_sequences=True, name='LSTM1')(flatten)
    # add second LSTM layer with 1000 units
    lstm2 = LSTM(1000, time_major=False, stateful=False, return_sequences=True, name='LSTM2')(lstm1, initial_state=[state_h, state_c])
    # lstm2 = LSTM(1000, time_major=False, stateful=False, return_sequences=True, name='LSTM2')(lstm1)

    ## add final dense layer that maps LSTM outputs to 6 output neurons
    out = TimeDistributed(Dense(6, activation=linear), name='output')(lstm2)
    # NOTE output shape: (BATCH, SUBSEQ, 6)

    ## return untrained and uncompiled model
    model = tf.keras.models.Model(inputs=input_layers, outputs=out)
    return model

def main():
    # parse input arguemts
    args = parse_args()
    conf = config.Config(args.config)
    # make header for the DNN
    layernames = make_layernames(conf.training_files[0], conf.input_timesteps)
    # create and compile model
    model = create_model(layernames, conf.image_shape)
    # write model to disk
    name = 'deepvo__train__'\
             + 'in'       + str(len(layernames)) \
             + '_tInputs' + str(conf.input_timesteps) \
             + '_imw'     + str(conf.image_shape[0]) \
             + '_imh'     + str(conf.image_shape[1]) \
             + '_imc'     + str(conf.image_shape[2]) \
             + '_out6.h5'
    model.save(os.path.join(args.model_out, name))
    # print final model
    model.summary(line_length=150)

if __name__=="__main__":
    main()
