import tensorflow as tf
import numpy as np
import os, sys, glob, signal, time, math, argparse, zipfile
import src.config as config

# run eager execution
tf.compat.v1.enable_eager_execution()

###################### debugging tools ######################
# show images from numpy array
def show_image(image, label):
    import matplotlib.pyplot as plt
    if image.shape[2]==1:
        plt.imshow(image[:,:,0], cmap='gray')
    else:
        plt.imshow(image)
    plt.grid(False)
    plt.suptitle("label:\ntx = "+str(label[0])+", ty = "+str(label[1])+", tz = "+str(label[2])+",\nroll = "+str(label[3])+", pitch = "+str(label[4])+", yaw = "+str(label[5])) # write label to the super title
    plt.show()

# show images from list
def show_images(images, label, sequence_length):
    import matplotlib.pyplot as plt
    # get useful information on images
    num_images = len(images)
    fig_cols   = int(num_images/sequence_length) # NOTE 'num_images' is always a multiple of 'sequence_length' => no ceil needed here
    fig        = plt.figure(figsize=(sequence_length, fig_cols))
    # display images with pyplot.figure
    for row in range(sequence_length):
        for col in range(fig_cols):
            # check if image still exists for current cell
            idx = row * fig_cols + col
            if idx >= num_images:
                break
            # NOTE idx always points to valid data
            fig.add_subplot(sequence_length, fig_cols, idx+1)
            if images[0][0].shape[2] == 1: # case: GRAYSCALE images
                plt.imshow(images[idx][0][:,:,0], cmap='gray')
                plt.title(images[idx][1])
            else: # case: RGB images
                plt.imshow(images[idx][0])
                plt.title(images[idx][1])
    # set further pyplot properties
    plt.grid(False)
    plt.suptitle("label:\ntx = "+str(label[0])+", ty = "+str(label[1])+", tz = "+str(label[2])+",\nroll = "+str(rad2deg(label[3]))+"º, pitch = "+str(rad2deg(label[4]))+"º, yaw = "+str(rad2deg(label[5]))+"º") # write label to the super title
    plt.show()

# shows plots in tabbed sub-windows given an evo::plot_collection object
def show_tabbed_plots(pc):
    from PyQt5 import QtGui, QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    # make qt5 application
    app = QtGui.QGuiApplication.instance()
    if app == None:
        app = QtWidgets.QApplication([pc.title])
    # setup plot_collection
    pc.root_window = QtWidgets.QTabWidget()
    pc.root_window.setWindowTitle(pc.title)
    for name, fig in pc.figures.items():
        tab = QtWidgets.QWidget(pc.root_window)
        tab.canvas = FigureCanvasQTAgg(fig)
        vbox = QtWidgets.QVBoxLayout(tab)
        vbox.addWidget(tab.canvas)
        toolbar = NavigationToolbar2QT(tab.canvas, tab)
        vbox.addWidget(toolbar)
        tab.setLayout(vbox)
        for axes in fig.get_axes():
            if isinstance(axes, Axes3D):
                axes.mouse_init() # to allow for dragging 3D plot with mouse
        pc.root_window.addTab(tab, name)
    pc.root_window.show()
    app.exec_()
    plt.close('all')

def prepare_observations(obs, labels, layernames, seq_len, image_files, label_files, t0, t1):
    images_batch = obs[0]
    labels_batch = obs[1]
    # roll random batch entry
    batch_sz  = len(obs[1])
    batch_idx = np.random.randint(0, batch_sz)
    # pick random batch label
    label = labels_batch[batch_idx].numpy()
    # collect associated batch images
    images = []
    orig_label_written = False
    for i, im_class in enumerate(images_batch):
        im_dbg_info = im_class[0][batch_idx].numpy().decode('utf-8')
        im_no = im_dbg_info.split('_')[-1]
        im_id = im_dbg_info.split(im_no)[0]
        image = im_class[1][batch_idx].numpy()
        layer = layernames[i]
        images.append( (layer, im_no, image) )
        # pick original label for checking -> orig label == labels[T] where T is earliest time point in image ids
        if not orig_label_written and int(layer.split('_')[-1]) == 0:
            print("observation starting at t = {} (dataset: {})".format(im_no, im_id[:-1]))
            orig_label = combine(labels[im_id][int(im_no)+t0 : int(im_no)+t1, :])
            orig_label_written = True
    # check if label assignment is correct
    print("original label: ", orig_label)
    print("batch label   : ", label)
    clean_assert((orig_label == label).all(), image_files, label_files)
    # reshape observation data
    im_tmp = [ [] for x in range(seq_len) ]
    for im_data in images:
        name = im_data[0]
        t    = int(name.split('_')[-1])
        im_tmp[t].append( (im_data[2], name+"\n(t = {})".format(int(im_data[1]))) )
    # flatten im_tmp list
    image_data = [ e for sub in im_tmp for e in sub ]
    print("##########")
    return image_data, label

def prepare_observations_keras(obs, labels, layernames, seq_len, image_files, label_files):
    images_batch = obs[0]
    labels_batch = obs[1]
    # roll random batch entry
    batch_sz  = len(obs[1])
    batch_idx = np.random.randint(0, batch_sz)
    # pick random batch label
    label = labels_batch[batch_idx].numpy()
    # collect associated batch images
    images = []
    for i, (layername, im) in enumerate(images_batch.items()):
        image = im[batch_idx].numpy()
        layer = layername
        images.append( (layer, image) )
    # visualize observation
    im_tmp = [ [] for x in range(seq_len) ]
    for im_data in images:
        name = im_data[0]
        t    = int(name.split('_')[-1])
        im_tmp[t].append( (im_data[1], name) )
    # flatten im_tmp list
    image_data = [ e for sub in im_tmp for e in sub ]
    return image_data, label

def make_observations_figure(image_data, label, seq_len):
    import matplotlib.pyplot as plt
    # make observation figure
    num_images = len(image_data)
    figcols = int(num_images/seq_len)
    fig_obs, im_axes = plt.subplots(nrows=seq_len, ncols=figcols, squeeze=False)
    for row in range(seq_len):
        for col in range(figcols):
            idx = row * figcols + col
            if idx >= num_images:
                break
            # NOTE idx always points to valid data
            if image_data[0][0].shape[2] == 1: # case: GRAYSCALE images
                im_axes[row][col].imshow(image_data[idx][0][:,:,0], cmap='gray')
            else: # case: RGB images
                im_axes[row][col].imshow(image_data[idx][0])
            im_axes[row][col].set_title(image_data[idx][1])
            im_axes[row][col].grid(False)
    # add sample image figure to plot_collection
    fig_obs.suptitle("label:\ntx = "+str(label[0])+", ty = "+str(label[1])+", tz = "+str(label[2])+",\nroll = "+str(rad2deg(label[3]))+"º, pitch = "+str(rad2deg(label[4]))+"º, yaw = "+str(rad2deg(label[5]))+"º")
    # return observation figure
    return fig_obs

def make_evo_traj_figures(traj_file_path):
    import matplotlib.pyplot as plt
    from evo import main_traj
    from evo.tools import plot
    # fake evo traj kitti input arguments
    class evo_args:
        def __init__(self, path):
            self.subcommand  = 'kitti'
            self.pose_files = [path]
            self.ref         = None
    testArgs = evo_args(traj_file_path)
    # load trajectories using evo tools
    trajectories, _ = main_traj.load_trajectories(testArgs)
    # create and setup plot collection object
    fig_xyz, axarr_xyz = plt.subplots(3, sharex='col')
    fig_rpy, axarr_rpy = plt.subplots(3, sharex='col')
    fig_traj = plt.figure()
    plot_mode = plot.PlotMode['xyz']
    ax_traj = plot.prepare_axis(fig_traj, plot_mode)
    # make trajecory plots
    for name, traj in trajectories.items():
        color = next(ax_traj._get_lines.prop_cycler)['color']
        plot.traj(ax_traj, plot_mode, traj, '-', color, name)
        plot.traj_xyz(axarr_xyz, traj, '-', color, name, start_timestamp=None)
        plot.traj_rpy(axarr_rpy, traj, '-', color, name, start_timestamp=None)
        fig_rpy.text(0., 0.005, "euler_angle_sequence: {}".format('RPY'), fontsize=6)
    # return figures
    return fig_traj, fig_xyz, fig_rpy

# NOTE set keras_compat=True if tf.dataset is mapped to 'make_keras_compatible', else use keras_compat=False
def debug_vis(ds_final, labels, layernames, seq_len, image_files, label_files, keras_compat=True):
    from evo.tools import plot
    for obs in ds_final:
        # make figures for trajectory and observations
        if keras_compat:
            image_data, label = prepare_observations_keras(obs, labels, layernames, seq_len, image_files, label_files)
        else:
            image_data, label = prepare_observations(obs, labels, layernames, seq_len, image_files, label_files, conf.t0, conf.t1)
        fig_obs = make_observations_figure(image_data, label, seq_len)
        # create temporary pose file which holds 1) identity and 2) relative pose from label
        tmp_file_name = '.tmp_label.txt'
        with open(tmp_file_name, 'w') as f:
            f.write(mat2string(np.eye(4, dtype=np.float64)) + mat2string(euler2mat(label)))
        fig_traj, fig_xyz, fig_rpy = make_evo_traj_figures(tmp_file_name)
        # remove temporary pose file
        os.remove(tmp_file_name)
        # add figures to evo::plot_collection instance
        plot_collection = plot.PlotCollection("evo_traj - trajectory plot")
        plot_collection.add_figure("observations", fig_obs)
        plot_collection.add_figure("trajectories", fig_traj)
        plot_collection.add_figure("xyz_view", fig_xyz)
        plot_collection.add_figure("rpy_view", fig_rpy)
        # show all plots in tabbed window
        show_tabbed_plots(plot_collection)


###################### helpers ######################
# decodes and preprocess raw image
def preprocess_image(image, shape):
    # decod raw image
    image = tf.image.decode_image(image, channels=shape[2])
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

# get parameters from config file
def parse_args():
    argparser = argparse.ArgumentParser(description="This script trains a deep neural network on image sequences to learn to estimate the egomotion of the camera.")
    argparser.add_argument('config', help="Config file needs to be passed in order to specify the training setup. See 'configs/sample.conf' for an template.")
    argparser.add_argument('--unpack_to', '-u', type=str, default='.', help="If train on cluster set this to the remote directory like \'/scratch/X\'. All datasets used for training will be extracted to this directory.")
    return argparser.parse_args()

# generates constant image description to parse image from TFRecord file
def get_image_description():
    desc = {
        'id'    : tf.io.FixedLenFeature([], tf.string),
        'image' : tf.io.FixedLenFeature([], tf.string),
    }
    return desc

# generates constant header description to parse header from TFRecord file
def get_header_description():
    desc = {
        'height'          : tf.io.FixedLenFeature([], tf.int64),
        'width'           : tf.io.FixedLenFeature([], tf.int64),
        'channels'        : tf.io.FixedLenFeature([], tf.int64),
        'num_images'      : tf.io.FixedLenFeature([], tf.int64),
    }
    return desc

# NOTE map this function to each individual input image tf.data.Dataset to parse and decode the images within the pipeline
# NOTE 'image_shape' needs to be provided global
def parse_image_record(raw_record):
    record = tf.io.parse_single_example(raw_record, get_image_description())
    image  = tf.io.decode_raw(record['image'], tf.float32); image = tf.reshape(image, conf.image_shape);
    return (record['id'], image)

# this functions takes the n-tuple of input-image records, extract the images and returns a dictionary mapping
# the models layernames to the associated image
# NOTE map this function to the final tf.data.Dataset if you want to train using tf.keras.models.Model::fit()
# NOTE 'layernames' needs to be provided global
# NOTE the 'layernames' array was created simultanous with the input images dataset, it is the inverse mapping of 'names2idx'
def make_keras_compatible(*records):
    # use global layernames array to map tuple idx to layername
    ret = ({ layernames[i] : parse_image_record(raw_image)[1] for i, raw_image in enumerate(records[0]) }, records[1])
    return ret

def make_debug_compatible(*records):
    # use global layernames array to map tuple idx to layername
    ret = (tuple([parse_image_record(raw_image) for i, raw_image in enumerate(records[0])]), records[1])
    return ret

# NOTE combines 6 dof transformations (tx,ty,tz,r,p,y)
def combine(labels):
    # iterate 1st dim of labels (shape: (n,6)) and add 6 dof values
    if labels.shape[0] == 1:
        # NOTE case: no need to combine labels -> sequence length is 2 and labels are already in consecutive order
        return labels[0,:]
    else:
        # NOTE case: combine transformations -> assuming transformations are ordered consecutively in time
        # transform all relative euler poses into matrices and multiply them together
        T = euler2mat(labels[0,:])
        for i in range(1, labels.shape[0]):
            T = T * euler2mat(labels[i,:])
        # extract R from muliplied matrix and extract euler angles from R
        euler = rot2euler(T[:3,:3])[0] # NOTE always pick first solutions since second solution is unrealistic in context of carla data
        # t column of T and euler angles are new labels -> return them in correct shape
        return np.array([T[0,3], T[1,3], T[2,3], euler[0], euler[1], euler[2]])

# converts transformations from euler representation (tx,ty,tz,roll,pitch,yaw) to 4x4 matrix
def euler2mat(arr):
    # NOTE arr holds 6 dof poses: (tx,ty,tz,roll,pitch,yaw)
    def c(x):
        return math.cos(x)
    def s(x):
        return math.sin(x)
    tx = arr[0]; ty = arr[1]; tz = arr[2];
    r  = arr[3]; p  = arr[4]; y  = arr[5];
    # using: http://www.gregslabaugh.net/publications/euler.pdf -> gt_rotation = R_z(yaw)*R_y(pitch)*R_x(roll), where R are rotations about standard unit vectors
    return np.matrix([[c(y)*c(p), c(y)*s(p)*s(r)-s(y)*c(r), c(y)*s(p)*c(r)+s(y)*s(r), tx],
                      [s(y)*c(p), s(y)*s(p)*s(r)+c(y)*c(r), s(y)*s(p)*c(r)-c(y)*s(r), ty],
                      [-s(p),     c(p)*s(r),                c(p)*c(r),                tz],
                      [0.0,       0.0,                      0.0,                      1.0]])

# NOTE takes a 3x4 or 4x4 numpy matrix/array and returns a string by enrolling the upper left 3x4 sub-matrix in row major
def mat2string(mat):
    ret  = str(mat[0,0]) + " " + str(mat[0,1]) + " " + str(mat[0,2]) + " " + str(mat[0,3]) + " "
    ret += str(mat[1,0]) + " " + str(mat[1,1]) + " " + str(mat[1,2]) + " " + str(mat[1,3]) + " "
    ret += str(mat[2,0]) + " " + str(mat[2,1]) + " " + str(mat[2,2]) + " " + str(mat[2,3]) + "\n"
    return ret

# extracts euler angles from rotation matrix
# NOTE rotation matrix must be composed according to XYZ (roll-pitch-yaw) convention as in function euler2mat
# NOTE results are only valid iff roll and yaw angles are in range [-pi,pi] and pitch is in range [-pi/2,pi/2]
# NOTE function will return 2 solutions in the regular case and 1 sample solution from infinity many in the gimbal lock case
def rot2euler(R):
    if abs(R[2,0]) < 0.999998:
        # NOTE case: R20 != +-1 => pitch != 90º
        pitch1 = -math.asin(R[2,0])
        pitch2 = math.pi - pitch1
        roll1  = math.atan2(R[2,1]/math.cos(pitch1), R[2,2]/math.cos(pitch1))
        roll2  = math.atan2(R[2,1]/math.cos(pitch2), R[2,2]/math.cos(pitch2))
        yaw1   = math.atan2(R[1,0]/math.cos(pitch1), R[0,0]/math.cos(pitch1))
        yaw2   = math.atan2(R[1,0]/math.cos(pitch2), R[0,0]/math.cos(pitch2))
        # return the two remaining possible solutions
        return np.array([[roll1, pitch1, yaw1], [roll2, pitch2, yaw2]])
    else: # NOTE that case should not occur on our data
        print("Gimbal Lock Case!!\nTODO extend rot2euler function using prev. euler angles to determine best solution!")
        # NOTE case: Gimbal Lock since pitch==+-90º -> there are infinity many solutions !
        yaw = 0.0 # pick yaw arbitrary, since it is linked to roll
        # NOTE R20 can either be -1 or 1 in this case
        if (R[2,0] + 1.0) < 1e-6:
            # NOTE case: R20==-1
            pitch = math.pi/2.0
            roll  = yaw + math.atan2(R[0,1], R[0,2])
        else:
            # NOTE case: R20==1
            pitch = -math.pi/2.0
            roll  = -yaw + math.atan2(-R[0,1], -R[0,2])
        # return one sample solution in the gimbal lock case
        return np.array([[roll, pitch, yaw]])

def rad2deg(x):
    return x * 180.0 / math.pi

# removes unpacked files and exits code
def cleanup_and_exit(image_files_list, label_files_list):
    cleanup(image_files_list, label_files_list)
    exit()

def cleanup(image_files_list, label_files_list):
    print("[INFO] clean up...", end='', flush=True)
    # remove extracted image files
    for image_files in image_files_list:
        for f in image_files:
            os.remove(f)
    # remove extracted label file
    for label_files in label_files_list:
        for f in label_files:
            os.remove(f)
    print(" done")

# if cond fails: clean up all extracted files and exit
def clean_assert(condition, image_files_list, label_files_list):
    if not condition:
        cleanup(image_files_list, label_files_list)
        assert(condition) # call assert in order to get typical assert error (lazy code...)

def check_model_layout(model, layernames):
    # check if number of input layers of model is as expected
    tmp_input_layers = [ layer for layer in model.layers if isinstance(layer, tf.keras.layers.InputLayer) ]
    if len(tmp_input_layers) != len(layernames):
        print("[ERROR] wrong number if input layers! Expected {} but {} given.".format(len(layernames), tmp_input_layers))
        return False
    # check if layernames are as expected
    for lname in layernames:
        try:
            model.get_layer(name=lname)
        except ValueError:
            print("[ERROR] unexpected layername encounterd at loaded model, ensure to load the correct model and that it was built with the same config as currently set")
            return False
    return True

def signal_handler(sig, frame):
    print("\n[INFO] exit on Ctrl+C")
    cleanup_and_exit(train_im_files, train_label_files)

def tfrec_to_ds(_dataset_files, _unpack_dir, _im_shape_conf, _seq_len, _t0, _t1, _dataset_name):
    ## extract dataset archive
    num_obs_total  = 0
    final_datasets = []
    layernames     = [] # TODO when functions will be wrapped into dataset class, this variable should become class member
    image_files    = [ [] for i in range(len(_dataset_files)) ] # TODO when functions will be wrapped into dataset class, this variable should become class member
    label_files    = [ [] for i in range(len(_dataset_files)) ] # TODO when functions will be wrapped into dataset class, this variable should become class member
    label_list_dbg = dict() # NOTE only used for debugging
    for i_arch in range(len(_dataset_files)):
        arch_prefix = _dataset_files[i_arch].split('/')[-1].split('.')[0] + '_'
        fz = zipfile.ZipFile(_dataset_files[i_arch], 'r')
        # read filenames from archive
        if i_arch == 0:
            arch_prefix_init = arch_prefix # needed to cut the prefix in later iterations
            for im_file in fz.namelist():
                filename     = os.path.join(_unpack_dir, arch_prefix + im_file)
                filename_ext = fz.extract(im_file, path=_unpack_dir); os.rename(filename_ext, filename);
                if im_file == 'labels.npz':
                    label_files[i_arch].append(filename)
                else:
                    image_files[i_arch].append(filename)
        else:
            # if multiple archives are used it needs to be ensured that all archives holds the same files
            for im_file in image_files[0]:
                im_file = im_file.split(os.path.join(_unpack_dir, arch_prefix_init))[-1]
                try:
                    filename     = os.path.join(_unpack_dir, arch_prefix + im_file)
                    filename_ext = fz.extract(im_file, path=_unpack_dir); os.rename(filename_ext, filename);
                    image_files[i_arch].append(filename)
                except KeyError:
                    print("[ERROR] file {} is required but could not be found in {}".format(im_file, _dataset_files[i_arch]))
                    fz.close()
                    cleanup_and_exit(image_files, label_files)
            # extract labels
            filename     = os.path.join(_unpack_dir, arch_prefix + 'labels.npz')
            filename_ext = fz.extract('labels.npz', path=_unpack_dir); os.rename(filename_ext, filename);
            label_files[i_arch].append(filename)
        # close archive
        fz.close()

        ## compose dictionary of input image TFRecord files
        # dict( input_layername : TFRecordDataset )
        input_image_dict = { f.split('/')[-1].split('.')[0].split(arch_prefix)[-1] : tf.data.TFRecordDataset(f, compression_type='ZLIB') for f in image_files[i_arch] }

        ## read header information
        print("[INFO] loading header from {}...".format(_dataset_files[i_arch]), end='', flush=True)
        # NOTE all image TFRecord files have an header record as very first entry
        ds_header     = list(input_image_dict.values())[0]
        header_record = tf.io.parse_single_example(next(iter(ds_header)), get_header_description())
        # NOTE height, width and channels must be equal amongst all headers
        if i_arch == 0: # initially load headers without checking
            # retrive necessarry information from header
            im_height   = header_record['height'].numpy()
            im_width    = header_record['width'].numpy()
            im_channels = header_record['channels'].numpy()
            image_shape = (im_height, im_width, im_channels)
            clean_assert(image_shape == tuple(_im_shape_conf), image_files, label_files) # NOTE images in header needs to fit the shape given in config
        else: # for any further archive check if headers are compatible
            clean_assert(im_height == header_record['height'].numpy(), image_files, label_files)
            clean_assert(im_width == header_record['width'].numpy(), image_files, label_files)
            clean_assert(im_channels == header_record['channels'].numpy(), image_files, label_files)
        # compute information necessarry for further computation
        num_images       = header_record['num_images'].numpy()
        num_observations = num_images - (_seq_len - 1)
        num_obs_total   += num_observations
        del ds_header
        del header_record
        print(" done")

        ## read labels from .npz file
        print("[INFO] loading labels from {}...".format(_dataset_files[i_arch]), end='', flush=True)
        # NOTE labels are stored as numpy array with shape (OBSERVATION_LENGTH, 6)
        # NOTE accessing: labels[T] returns the 6 dof relative pose from time T to T+1
        labels = np.load(label_files[i_arch][0])['labels']
        clean_assert(num_images-1 == labels.shape[0], image_files, label_files) # NOTE number of training images must match the number of labels + 1 (since each pair of images needs one label)
        # prepare labels s.t. user can specify between which 2 timepoints within the sequence the rel. pose should be used as label
        # example: SEQ_LEN=4 -> [_t0,_t1,t2,t3], label_from=[1,2] => use rel. pose from _t1 to t2
        clean_assert(_seq_len>=2 and _t0<_t1 and _t0>=0 and _t1>0 and _t0<(_seq_len-1) and _t1<_seq_len, image_files, label_files) # NOTE check if _t0,_t1,_seq_len are valid
        observation_labels = [ combine(labels[i+_t0 : i+_t1, :]) for i in range(num_observations) ]
        observation_labels = np.array(observation_labels)
        # create tf.data.Dataset object for labels
        ds_labels = tf.data.Dataset.from_tensor_slices(observation_labels)
        label_list_dbg[arch_prefix] = labels # NOTE only used for debugging
        del labels
        print(" done")

        ## setup image data pipeline
        print("[INFO] loading image data from {}...".format(_dataset_files[i_arch]), end='', flush=True)
        # iterate all input images and create td.data.Datasets for each image in the observation sequence
        ds_list    = []
        for i, (name, ds) in enumerate(input_image_dict.items()):
            # skip the header
            dataset = ds.skip(1)
            # create observation sequence
            for t in range(_seq_len):
                sub_ds = dataset.skip(t).take(num_observations) # NOTE length of final dataset is: num_images - (_seq_len-1)
                # sub_ds = sub_ds.map(parse_image_record)
                # sub_ds = sub_ds.map(parse_image_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                ds_list.append(sub_ds)
                # at fist archive: fill datastructures to keep track of the order of input images
                if i_arch == 0: # NOTE only store layernames at the very first archive -> they must be equal throughout all final datasets
                    layername = name + '_' + str(t)
                    layernames.append(layername)
        del dataset
        del sub_ds
        # zip all images together to one dataset returning (im1, im2, ...)
        ds_imgs = tf.data.TFRecordDataset.zip(tuple(ds_list))
        # ds_imgs = ds_imgs.map(make_keras_compatible) # this function makes the final dataset return a tuple containing all input images in the order as given by layernames
        # ds_imgs = ds_imgs.map(make_keras_compatible, num_parallel_calls=tf.data.experimental.AUTOTUNE) # this function makes the final dataset return a tuple containing all input images in the order as given by layernames
        # combine images with labels to one dataset returning ((im1, im2, ...), labels)
        ds = tf.data.Dataset.zip((ds_imgs, ds_labels))
        # append to list of final datasets
        final_datasets.append(ds)
        del ds_imgs
        del ds
        print(" done")

    # concatenate all tf.data.Datasets from _dataset_files to one long tf.data.Dataset
    print("[INFO] concatenating {} datasets to dataset with {} observations...".format(len(final_datasets), num_obs_total), end='', flush=True)
    ds_final = final_datasets[0]
    for i in range(1,len(final_datasets)):
        ds_final = ds_final.concatenate(final_datasets[i])
    print(" done")

    # print info about dataset
    print("[INFO] information about {}: ".format(_dataset_name))
    print("\timage shape           : {}".format(image_shape))
    print("\tsequence length       : {}".format(_seq_len))
    print("\tinfere pose from (t0) : {}".format(_t0))
    print("\tinfere pose till (t1) : {}".format(_t1))
    print("\tnumber observations   : {}".format(num_obs_total))
    print("\tlabel format          : {}".format('(tx, ty, tz, roll, pitch, yaw)'))
    print("\tDNN input layer names : {}".format(layernames))

    ## data to return
    # meta-information about dataset extracted from header -> will be used for checking compatability with validation ds and NN
    # information needed for further processing of dataset
    info = (num_obs_total, layernames)
    # administrative stuff needed for debugging and cleaning up at the end # NOTE TODO when this function belongs to class these things will become class members
    clean_dbg_stuff = (image_files, label_files, label_list_dbg)
    # final concatenated dataset: on next() it will return a 2-tuple contaning 1) a n-tuple of n input images and 2) the appropriate label
    return ds_final, info, clean_dbg_stuff


###################### test code ######################
# force tensorflow to throw its inital messages on the very beginning of that script
tf.config.experimental_list_devices()

# define signal handler for clean exit on Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# give info about used tensorflow version
print("[INFO] using tensorflow version {}".format(tf.__version__))

# parse config file
args = parse_args()
conf = config.Config(args.config)

# make trainaing dataset from tfrec
ds_train, train_ds_info, train_ds_meta = tfrec_to_ds(conf.dataset_files, args.unpack_to, conf.image_shape, conf.seq_len, conf.t0, conf.t1, "training dataset")
train_obs_total  = train_ds_info[0]; layernames = train_ds_info[1];
train_im_files = train_ds_meta[0]; train_label_files = train_ds_meta[1]; train_label_list_dbg = train_ds_meta[2];

# TODO load validation dataset if requested
# TODO assert: check if layernames are consistent with the ones from ds_train

## setup dataset pipeline
# TODO play around with tf.contrib.data functions to make pipeline more effective -> read https://www.tensorflow.org/tutorials/load_data/images#performance
# TODO set size of shuffle_buffer s.t. it fits into local mem -> make it independent from num_images (maybe tf.data.experimental can help)
# NOTE pipeline info: 1) observations are split into training and validation sets 1.5) validation set will be cached in local mem since it is small enough 2) sets are mapped to preprocess function in parallel 3) sets are batched and repeated 4) sets will be prefetched
print("[INFO] setting up dataset pipeline (i.e. shuffling, batching, etc)...", end='', flush=True)
# shuffle observations
if conf.validation_split <= 0.0:
    # train on all observations
    ds_train = ds_train.shuffle(train_obs_total).map(make_keras_compatible, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(conf.batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
else:
    # TODO put into functions
    # split the final observations into training and validation sets
    num_validation_obs  = int(conf.validation_split * train_obs_total)
    if conf.debug:
        # NOTE do not map 'make_keras_compatible' for debug to get all debug information
        ds_train_validation = ds_train.take(num_validation_obs).shuffle(num_validation_obs).cache().map(make_debug_compatible, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().batch(conf.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        ds_train_training   = ds_train.skip(num_validation_obs).shuffle(train_obs_total-num_validation_obs).map(make_debug_compatible, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().batch(conf.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        ds_train_validation = ds_train.take(num_validation_obs).shuffle(num_validation_obs).cache().map(make_keras_compatible, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(conf.batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        ds_train_training   = ds_train.skip(num_validation_obs).shuffle(train_obs_total-num_validation_obs).map(make_keras_compatible, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(conf.batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)

print(" done")

## beg DEBUG visualize data without keras compatability
if conf.debug:
    ## visualize data from final dataset with extended debug informations and asserts
    # NOTE comment line where 'make_keras_compatible' is mapped to ds_train
    print("[INFO] visualizing random observations from batched dataset without mapping 'make_keras_compatible' to final_ds...")
    debug_vis(ds_train_training, train_label_list_dbg, layernames, conf.seq_len, train_im_files, train_label_files, keras_compat=False)
    cleanup_and_exit(train_im_files, train_label_files)

    # ## visualize data from final dataset
    # print("[INFO] visualizing random observations from batched final_ds dataset...")
    # debug_vis(ds_train_training, train_label_list_dbg, layernames, conf.seq_len, train_im_files, train_label_files, keras_compat=True)
    # cleanup_and_exit(train_im_files, train_label_files)
## end DEBUG

## savely load model from config path
model = tf.keras.models.load_model(conf.model_file)
clean_assert(check_model_layout(model, layernames), train_im_files, train_label_files)
# TODO repeat check for validation data

## print model informations
print("[INFO] information about the DNN model thats going to be trained:")
model.summary()

## setup tf.keras callbacks for training loop NOTE infos at https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard#class_tensorboard
print("[INFO] training the model, logs will be written to '{}' and checkpoints to '{}':".format(conf.log_dir, conf.checkpoint_dir))
# log tensorboard data NOTE inofs at https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard#class_tensorboard
tb_logger = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(conf.log_dir, 'tensorboard'))
# log into csv file NOTE infos at https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(conf.log_dir, 'csv', 'training.log'))
# setup training checkpointing NOTE infos at https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
timestamp_file = str(int(time.time()))
cpkt_filename = os.path.join(conf.checkpoint_dir, 'ckpt-'+timestamp_file+'-{epoch:05d}.ckpt')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cpkt_filename, save_weights_only=True, verbose=1, save_freq=conf.checkpoint_freq)
# NEXT delete: checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cpkt_filename, save_weights_only=True, verbose=1) # TODO save_freq not available on tf2.0a
# TODO check usefull options: mode, save_best_only

## train model using keras training loop
if conf.validation_split <= 0.0:
    history = model.fit(
        ds_train,
        steps_per_epoch=train_obs_total/conf.batch_size,
        epochs=conf.epoches,
        callbacks=[csv_logger, tb_logger, checkpoint_callback])
else:
    history = model.fit(
        ds_train_training,
        validation_data=ds_train_validation,
        validation_steps=num_validation_obs/conf.batch_size,
        steps_per_epoch=(train_obs_total-num_validation_obs)/conf.batch_size,
        epochs=conf.epoches,
        callbacks=[csv_logger, tb_logger, checkpoint_callback])

# save final model
print("[INFO] training done, final model is saved at '{}'".format(conf.model_out))
model.save(conf.model_out)

print("[INFO] training loss history:")
print(history.history['loss'])
if conf.validation_split > 0.0:
    print("[INFO] validation loss history:")
    print(history.history['val_loss'])

## clean up
# remove all extracted files from disk
cleanup_and_exit(train_im_files, train_label_files)

