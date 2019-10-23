# script creates simple flat dummy model that takes arbitrary input arguments and has 6 output neurons

import tensorflow as tf
import os, zipfile, argparse
import src.config as config

def parse_args():
    argparser = argparse.ArgumentParser(description="This scripts generates a 7-layer CNN with final MPC.")
    argparser.add_argument('config', help="same config file as used for training with 'train_sequence.py'. Config file is required to set the input layers names as 'train_tfrec.py' expects it. Names will be generated from 'DATASET_FILES' option, so ensure to use this option consistently.")
    argparser.add_argument('--model_out', '-out', default='', help="path to where the model should be saved.")
    return argparser.parse_args()

# NOTE 'arch_file' holds the path to a sequence zip archive
# NOTE 'seq_len' defines the length of a single observation sequence
def make_layernames(arch_file, seq_len):
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
        for t in range(seq_len):
            layernames.append(name + '_' + str(t))
    # close archive
    fz.close()
    ## return list of input layernames
    return layernames

# NOTE 'layernames' holds the names of the input layers => size of 'layernames' == size of input layers
# NOTE 'image_shape' is tuple that holds resolution of input images, assuming all images to have the same size
def create_model(layernames, image_shape):
    relu = tf.keras.activations.relu
    # generate input layers
    input_layers   = []
    cnn_layers = []
    for layername in layernames:
        input_layer = tf.keras.layers.Input(shape=image_shape, name=layername)
        input_layers.append(input_layer)
        ## add conv layers
        ## architecture simplified from FlowNetCorr (http://openaccess.thecvf.com/content_iccv_2015/papers/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.pdf)
        # Conv1: kernel=7x7, padding=zeropadding(3,3), stride=(2,2), channels=64, activation=ReLu
        conv1 = tf.keras.layers.Conv2D(64, 7, padding='same', strides=2, activation=relu, data_format='channels_last')(input_layer)
        # Conv2: kernel=5x5, padding=zeropadding(2,2), stride=(2,2), channels=128, activation=ReLu
        conv2 = tf.keras.layers.Conv2D(128, 5, padding='same', strides=2, activation=relu, data_format='channels_last')(conv1)
        # Conv3: kernel=5x5, padding=zeropadding(2,2), stride=(2,2), channels=256, activation=ReLu
        conv3 = tf.keras.layers.Conv2D(256, 5, padding='same', strides=2, activation=relu, data_format='channels_last')(conv2)
        ## add layers to list
        cnn_layers.append(conv3)
    ## instead of applying correlation layer like in FlowNetCorr, simply concatenate the tensors
    # concatenation axis: last dimension (channels) = -1 (default)
    intermediate = tf.keras.layers.concatenate(inputs=cnn_layers)
    ## output from combined input convolutions is then passed to final convultion layers form FlowNetCorr
    # Conv3_1: kernel=3x3, padding=zeropadding(1,1), stride=(1,1), channels=256, activation=ReLu
    conv3_1 = tf.keras.layers.Conv2D(256, 3, padding='same', strides=1, activation=relu, data_format='channels_last')(intermediate)
    # Conv4: kernel=3x3, padding=zeropadding(1,1), stride=(2,2), channels=512, activation=ReLu
    conv4 = tf.keras.layers.Conv2D(512, 3, padding='same', strides=2, activation=relu, data_format='channels_last')(conv3_1)
    # Conv4_1: kernel=3x3, padding=zeropadding(1,1), stride=(1,1), channels=512, activation=ReLu
    conv4_1 = tf.keras.layers.Conv2D(512, 3, padding='same', strides=1, activation=relu, data_format='channels_last')(conv4)
    # Conv5: kernel=3x3, padding=zeropadding(1,1), stride=(2,2), channels=512, activation=ReLu
    conv5 = tf.keras.layers.Conv2D(512, 3, padding='same', strides=2, activation=relu, data_format='channels_last')(conv4_1)
    # Conv5_1: kernel=3x3, padding=zeropadding(1,1), stride=(1,1), channels=512, activation=ReLu
    conv5_1 = tf.keras.layers.Conv2D(512, 3, padding='same', strides=1, activation=relu, data_format='channels_last')(conv5)
    # Conv6: kernel=3x3, padding=zeropadding(1,1), stride=(2,2), channels=1024, activation=ReLu
    conv6 = tf.keras.layers.Conv2D(1024, 3, padding='same', strides=2, activation=relu, data_format='channels_last')(conv5_1)
    ## flatten final convoluted output
    flat = tf.keras.layers.Flatten(data_format='channels_last')(conv6)
    ## make mlp with 6 outpus for the 6 dof poses
    nInputNeurons = int((6 + (image_shape[0]/64)*(image_shape[1]/64)*1024)/2) # NOTE assuming image_shape is of the form (rows, cols, channels)
    dense_4 = tf.keras.layers.Dense(nInputNeurons, activation='linear')(flat)
    nInputNeurons = int((6 + nInputNeurons)/2)
    dense_2 = tf.keras.layers.Dense(nInputNeurons, activation='linear')(dense_4)
    if nInputNeurons > 500:
        nInputNeurons = 500
    dense_3 = tf.keras.layers.Dense(nInputNeurons, activation='linear')(dense_2)
    dense_4 = tf.keras.layers.Dense(100, activation='linear')(dense_3)
    out = tf.keras.layers.Dense(6, activation='linear')(dense_4) # output shape: 6 (trans. + euler)
    # set and compile model
    model = tf.keras.models.Model(inputs=input_layers, outputs=out)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    # return compiled model
    return model

def main():
    # parse input arguemts
    args = parse_args()
    conf = config.Config(args.config)
    # make header for the DNN
    layernames = make_layernames(conf.dataset_files[0], conf.seq_len)
    # create and compile model
    model = create_model(layernames, conf.image_shape)
    # write model to disk
    name = 'simple_cnn__in' + str(len(layernames)) \
             + '_seqlen' + str(conf.seq_len) \
             + '_imw'    + str(conf.image_shape[0]) \
             + '_imh'    + str(conf.image_shape[1]) \
             + '_out6'   + '.h5'
    model.save(os.path.join(args.model_out, name))
    # print final model
    model.summary()

if __name__=="__main__":
    main()
