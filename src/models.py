import tensorflow as tf

## substitutions
# activations
relu = tf.keras.activations.relu
linear = tf.keras.activations.linear
# layers
Conv2D = tf.keras.layers.Conv2D
TimeDistributed = tf.keras.layers.TimeDistributed
Flatten = tf.keras.layers.Flatten
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense

# FIXME rework input layers by time in order to get model compiled
class DeepVO(tf.keras.Model):
    ## INFO architecture from DeepVO (https://www.cs.ox.ac.uk/files/9026/DeepVO.pdf)

    def __init__(self, layernames, shape):
        super(DeepVO, self).__init__()

        ## add input layers
        # collect all input images in list
        self.input_layers = {}
        for layername in layernames:
            # input layers expect subsequenced data: (BATCH, SUBSEQ, imH, imW, imC) [BATCH=None will be added by keras automatically]
            self.input_layers[layername] = tf.keras.layers.Input(shape=(None, shape[0], shape[1], shape[2]), name=layername)

        ## add concatenation layer to stack input images NOTE assuming images are stored in channels_last format
        self.input_stacked = tf.keras.layers.Concatenate(axis=-1, name='stack_inputs')

        ## add FlowNet convolution layers: https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/flownet.pdf
        # Conv1: kernel=7x7, padding=zeropadding(3,3), stride=(2,2), channels=64, activation=ReLu
        self.conv1   = TimeDistributed(Conv2D(64, 7, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv1')
        # Conv2: kernel=5x5, padding=zeropadding(2,2), stride=(2,2), channels=128, activation=ReLu
        self.conv2   = TimeDistributed(Conv2D(128, 5, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv2')
        # Conv3: kernel=5x5, padding=zeropadding(2,2), stride=(2,2), channels=256, activation=ReLu
        self.conv3   = TimeDistributed(Conv2D(256, 5, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv3')
        # Conv3_1: kernel=3x3, padding=zeropadding(1,1), stride=(1,1), channels=256, activation=ReLu
        self.conv3_1 = TimeDistributed(Conv2D(256, 3, padding='same', strides=1, activation=relu, data_format='channels_last'), name='conv3_1')
        # Conv4: kernel=3x3, padding=zeropadding(1,1), stride=(2,2), channels=512, activation=ReLu
        self.conv4   = TimeDistributed(Conv2D(512, 3, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv4')
        # Conv4_1: kernel=3x3, padding=zeropadding(1,1), stride=(1,1), channels=512, activation=ReLu
        self.conv4_1 = TimeDistributed(Conv2D(512, 3, padding='same', strides=1, activation=relu, data_format='channels_last'), name='conv4_1')
        # Conv5: kernel=3x3, padding=zeropadding(1,1), stride=(2,2), channels=512, activation=ReLu
        self.conv5   = TimeDistributed(Conv2D(512, 3, padding='same', strides=2, activation=relu, data_format='channels_last'), name='conv5')
        # Conv5_1: kernel=3x3, padding=zeropadding(1,1), stride=(1,1), channels=512, activation=ReLu
        self.conv5_1 = TimeDistributed(Conv2D(512, 3, padding='same', strides=1, activation=relu, data_format='channels_last'), name='conv5_1')
        # Conv6: kernel=3x3, padding=zeropadding(1,1), stride=(2,2), channels=1024, activation=ReLu
        self.conv6   = TimeDistributed(Conv2D(1024, 3, padding='same', strides=2, activation=linear, data_format='channels_last'), name='conv6')

        ## add LSTMs to learn a mapping from the extracted features to 6 dof pose
        # flatten convolution output:
        # conv_6: (BATCH, SUBSEQ, shape[0]/64, shape[1]/64, 1024) -> (BATCH, SUBSEQ, shape[0]/64 * shape[1]/64 * 1024)
        self.flatten = TimeDistributed(Flatten(data_format='channels_last'), name='flatten_features')
        # add first LSTM layer with 1000 units
        self.lstm1 = LSTM(1000, return_sequences=True, name='LSTM1')
        # add second LSTM layer with 1000 units
        self.lstm2 = LSTM(1000, return_sequences=True, name='LSTM2')

        ## add final dense layer that maps LSTM outputs to 6 output neurons
        self.out = TimeDistributed(Dense(6, activation=linear), name='output')
        # NOTE output shape: (BATCH, SUBSEQ, 6)

    def call(self, inputs):
        ## distribute inputs over input layers
        # TODO verify inputs are coming as dict
        for name, data in inputs.items():
            self.input_layers[name](data)
        # stack input layers s.t. all input images are stacked together at their channel-dimension
        # TODO verify that stacking batches (of sequences) of images results in the same tensor as stacking the input image pairs beforehand
        x = self.input_stacked(self.input_layers)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv5(x)
        x = self.conv5_1(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.LSTM1(x)
        x = self.LSTM2(x)
        x = self.out(x)
        return x
