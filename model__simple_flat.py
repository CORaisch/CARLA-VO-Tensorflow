# script creates simple flat dummy model that takes arbitrary input arguments and has 6 output neurons

import tensorflow as tf
import os, zipfile, argparse
import src.config as config

def parse_args():
    argparser = argparse.ArgumentParser(description="This scripts generates a simple network for debug/test usages.")
    argparser.add_argument('config', help="same config file as used for training with 'train_sequence.py'. Config file is required to set the input layers names as 'train_tfrec.py' expects it. Names will be generated from 'TRAINING_FILES' option, so ensure to use this option consistently.")
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
# NOTE 'image_shape' is tuple that holds resolution of input images, assuming all images to have the same size
def create_model(layernames, image_shape):
    # generate input layers
    input_layers   = []
    flatten_layers = []
    for layername in layernames:
        input_layer   = tf.keras.layers.Input(shape=image_shape, name=layername)
        input_layers.append(input_layer)
        flatten_layer = tf.keras.layers.Flatten()(input_layer)
        flatten_layers.append(flatten_layer)
    # concatenate output of all input layers to one flat tensor
    x = tf.keras.layers.concatenate(inputs=flatten_layers)
    # make mlp with 6 outpus for the 6 dof poses
    mid = tf.keras.layers.Dense(1000, activation='relu')(x)
    out = tf.keras.layers.Dense(6, activation='linear')(mid)
    # set and compile model
    model = tf.keras.models.Model(inputs=input_layers, outputs=out)
    # return compiled model
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
    name = 'simple_flat__in' + str(len(layernames)) \
             + '_tInputs'     + str(conf.input_timesteps) \
             + '_imw'        + str(conf.image_shape[0]) \
             + '_imh'        + str(conf.image_shape[1]) \
             + '_imc'        + str(conf.image_shape[2]) \
             + '_out6'       + '.h5'
    model.save(os.path.join(args.model_out, name))

if __name__=="__main__":
    main()
