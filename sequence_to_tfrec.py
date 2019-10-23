## sample call:
## python sequence_to_tfrec.py sequence_00 /home/claudio/Datasets/CARLA/sequence_00/relative_euler_poses_nulled.txt '/home/claudio/Datasets/CARLA/sequence_00/rgb/left/images=camera_left' '/home/claudio/Datasets/CARLA/sequence_00/rgb/right/images=camera_right' -imc 1 -imw 196 -imh 196


import tensorflow as tf
import numpy as np
import zipfile
import os, sys, time, argparse

# setups for testing
import matplotlib.pyplot as plt
tf.compat.v1.enable_eager_execution()


###################### helpers ######################
def parse_args():
    # create argparse instance
    argparser = argparse.ArgumentParser(description="converts a sequence to TFRecord files used for training with Tensorflow")
    # add positional arguments
    argparser.add_argument('archive_name', type=str, help="name of final archive where all data will be written to")
    argparser.add_argument('labels', type=str, help="path to labels file")
    argparser.add_argument('inputmap', type=str, nargs='*', metavar="\'PATH/TO/IMAGES=NAME\'", help="maps all input sequences to the desired name in the archive. The archive name will later be used as name for the inputlayer of the DNN")
    # add optional arguments
    argparser.add_argument('--image_channels', '-imc', type=int, default=1, help="1 = images are stored grayscale, 3 = images are stored RGB")
    argparser.add_argument('--image_width', '-imw', type=int, default=196, help="width of stored images")
    argparser.add_argument('--image_height', '-imh', type=int, default=196, help="height of stored images")
    # parse args
    args = argparser.parse_args()
    # convert inputmap to dict
    args.inputmap = { pair.split('=')[0] : pair.split('=')[1] for pair in args.inputmap }
    # crop extension of archive name if exists
    # NOTE assuming no points in name except for extension
    tmp = args.archive_name.split('.')
    if len(tmp) != 1:
        args.archive_name = tmp[0]
    # return parsed arguments
    return args


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# decodes and preprocess raw image
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

# loads, decodes and preprocess an image given a path
def load_and_preprocess_image(path, shape):
    image = tf.io.read_file(path)
    return preprocess_image(image, shape)

def load_dataset_paths(inputmap, labels_file):
    image_paths, num_images = load_paths_from_dict(inputmap)
    # load dummy 6 DoF labels for testing
    f = open(labels_file, 'r'); lines = f.readlines(); f.close();
    labels = []
    for i, line in enumerate(lines):
        # NOTE skip very first pose since it is not a relative pose (its the nulled inital pose)
        if i == 0:
            continue
        labels.append([ float(x.replace('\n','')) for x in line.split(' ') ])
    # NOTE labels[i] == t_i -> t_i+1
    # return dict of all image paths, labels (synced with indices of image_paths) and number of images per input
    return image_paths, labels, num_images

def load_paths_from_dict(inputmap):
    # NOTE on 'image_paths':
    # shape: (NUM_CAMERA_TYPES, NUM_IMAGES) -> accessing: image_paths['name']['t']
    # image_paths contains lists of all paths to all images belongin to one frame class (i.e., rgb left, depth right, etc.)
    # it is stored as a list of np.arrays since not every camera type will come with the same number of cameras mounted
    image_paths = dict()
    # iterate inputmap load all subpaths
    for path, name in inputmap.items():
        paths = [ os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) ]; paths.sort()
        image_paths[name] = np.array(paths)
    # get some useful information about the dataset
    num_images = list(image_paths.values())[0].shape[0]
    # return dict of paths and number of images per path
    return image_paths, num_images

# writes labels to compressed npz file
def write_labels_to_npz(filename, labels):
    np.savez_compressed(filename, labels=labels)

# NOTE paths: np.array of all absolute image paths
# NOTE output: TFRecord file that maps its absolute path to an serialized and already preprocessed image
def write_images_to_tfrec(paths, filename, inputshape, archname, compression_type='ZLIB'):
    with tf.io.TFRecordWriter(filename, options=compression_type) as writer:
        # add header as very firts record
        header_record = make_header_record(inputshape, num_images)
        writer.write(header_record.SerializeToString())
        # write images
        for path in paths:
            tf_example = make_image_record(path, inputshape, archname)
            writer.write(tf_example.SerializeToString())

# NOTE record format: {'id' : TIME, 'image' : SERIALIZED_IMAGE}
def make_image_record(path, shape, prefix):
    # load and rescale image (rescaling controlled by user)
    image = load_and_preprocess_image(path, shape)
    # store id and serialized image per record
    im_id = prefix + '_' + path.split('/')[-1].split('.')[0] # extract id from file name (relying on convention)
    record = { 'id' : _bytes_feature(bytes(im_id, 'utf-8')), 'image' : _bytes_feature(image.numpy().tostring()) }
    return tf.train.Example(features=tf.train.Features(feature=record))

def make_header_record(shape, num_images):
    record = {
        'height'          : _int64_feature(shape[0]),
        'width'           : _int64_feature(shape[1]),
        'channels'        : _int64_feature(shape[2]),
        'num_images'      : _int64_feature(num_images),
    }
    return tf.train.Example(features=tf.train.Features(feature=record))


###################### test code ######################
# force tensorflow to throw its inital messages on the very beginning of that script
tf.config.experimental_list_devices()

# parse arguemts
args = parse_args()

# setup image paths of CARLA sequence
image_paths, labels, num_images = load_dataset_paths(args.inputmap, args.labels)

## write dataset files and zip them into an archive for convenience
# write each input image class to a tfrecord file
print("archive name: ", args.archive_name)
for name, paths in image_paths.items():
    filename = name + '.tfrec'
    print("[INFO] writing images to TFRecord at \'{0}\' compressing with 'zlib'...".format(filename), end='', flush=True)
    write_images_to_tfrec(paths, filename, (args.image_width, args.image_height, args.image_channels), args.archive_name, compression_type='ZLIB')
    print(" done")
# 2) write labels to .npy file
labels_file = 'labels.npz'
print("[INFO] writing labels at \'{0}\'...".format(labels_file), end='', flush=True)
write_labels_to_npz(labels_file, labels)
print(" done")

# write combined files into zip archive
archivefilename = args.archive_name + '.zip'
print("[INFO] combine files into zip archive '{}'...".format(archivefilename), end='', flush=True)
with zipfile.ZipFile(archivefilename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    filename = 'labels.npz'
    zipf.write(filename)
    os.remove(filename)
    for name in image_paths.keys():
        filename = name + '.tfrec'
        zipf.write(filename)
        os.remove(filename)
print(" done")
