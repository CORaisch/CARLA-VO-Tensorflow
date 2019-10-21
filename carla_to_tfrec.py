## sample call:
## python carla_to_tfrec.py /home/claudio/Datasets/CARLA/sequence_00 'rgb_left' 'rgb_right' -imc 1 -imw 196 -imw 196

import argparse
import os

def parse_args():
    # create argparse instance
    argparser = argparse.ArgumentParser(description="converts a CARLA sequence to TFRecord files used for training with Tensorflow")
    # add positional arguments
    argparser.add_argument('base', metavar="/PATH/TO/CARLA/SEQUENCE", help="path to CARLA sequence base directory")
    argparser.add_argument('sensors', type=str, nargs='*', metavar="{rgb|depth|semantic_segmentation}_{left|right}", help="list of sensors that will converted to tfrec file. The sensor name will later be used as name for the inputlayer of the DNN")
    # add optional arguments
    argparser.add_argument('--labels', '-l', type=str, default="relative_euler_poses_nulled.txt", help="select label file")
    argparser.add_argument('--archive_name', '-out', type=str, default=None, help="name of final archive where all data will be written to. Default will be base name of CARLA sequence")
    argparser.add_argument('--image_channels', '-imc', type=int, default=1, help="1 = images are stored grayscale, 3 = images are stored RGB")
    argparser.add_argument('--image_width', '-imw', type=int, default=196, help="width of stored images")
    argparser.add_argument('--image_height', '-imh', type=int, default=196, help="height of stored images")
    # parse args
    args = argparser.parse_args()
    # set default archive name
    if args.archive_name == None:
        args.archive_name = args.base.split('/')[-1]
    # return parsed arguments
    return args

args = parse_args()

## get labels file
# add extension if not already there
if len(args.labels.split('.')) == 1:
    args.labels = args.labels + '.txt'
labels_file = os.path.join(args.base, args.labels)

## convert list of sensors to argument string needed by sequence_to_tfrec.py
sensormap = ''
for sensor in args.sensors:
    splt = sensor.split('_')
    sensormap += '\'' + os.path.join(args.base, splt[0], splt[1], 'images') + '=' + sensor + '\' '

## compose and call string
callstr = "python sequence_to_tfrec.py " + args.archive_name + " " + labels_file + " " + sensormap +\
          " -imc " + str(args.image_channels) + " -imw " + str(args.image_width) + " -imh " + str(args.image_height)
print("call:", callstr)
os.system(callstr)
