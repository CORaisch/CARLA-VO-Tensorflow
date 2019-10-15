# script creates simple flat dummy model that takes arbitrary input arguments and has 6 output neurons

import tensorflow as tf
import os, zipfile

###################### config ######################
DATASET_FILES            = ['tfrec_sequences/sequence_00.zip', 'tfrec_sequences/sequence_01.zip']
SEQ_LEN                  = 2
MODEL_OUT                = 'models'
IMAGE_SHAPE              = [196, 196, 1]


###################### code ######################
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
    out = tf.keras.layers.Dense(6, activation='linear')(x) # output layer must have 6 output neurons for the 6 dof poses
    # set and compile model
    model = tf.keras.models.Model(inputs=input_layers, outputs=out)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    # return compiled model
    return model

def main():
    # parse input arguemts
    layernames = make_layernames(DATASET_FILES[0], SEQ_LEN)
    # create and compile model
    model = create_model(layernames, IMAGE_SHAPE)
    # write model to disk
    name = 'simple_flat__in' + str(len(layernames)) \
             + '_seqlen' + str(SEQ_LEN) \
             + '_imw'    + str(IMAGE_SHAPE[0]) \
             + '_imh'    + str(IMAGE_SHAPE[1]) \
             + '_out6'   + '.h5'
    model.save(os.path.join(MODEL_OUT, name))

if __name__=="__main__":
    main()
