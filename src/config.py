## sample call:
## python Config.py configs/test.conf

import argparse, configparser, json

class Config:
    def __init__(self, configfile):
        conf = configparser.ConfigParser()
        conf.read(configfile)
        # set DATASET values
        self.dataset_files = json.loads(conf['DATASET']['DATASET_FILES'])
        self.image_shape = json.loads(conf['DATASET']['IMAGE_SHAPE'])
        self.seq_len = int(conf['DATASET']['SEQ_LEN'])
        self.t0 = int(conf['DATASET']['T0'])
        self.t1 = int(conf['DATASET']['T1'])
        # set TRAINING values
        self.batch_size = int(conf['TRAINING']['BATCH_SIZE'])
        self.epoches = int(conf['TRAINING']['EPOCHES'])
        self.validation_split = float(conf['TRAINING']['VALIDATION_SPLIT'])
        self.model_file = json.loads(conf['TRAINING']['MODEL_FILE'])
        self.checkpoint_dir = json.loads(conf['TRAINING']['CHECKPOINT_DIR'])
        self.checkpoint_freq = self._parse_ckpt_frq(json.loads(conf['TRAINING']['CHECKPOINT_FREQ']))
        self.log_dir = json.loads(conf['TRAINING']['LOG_DIR'])
        self.model_out = json.loads(conf['TRAINING']['MODEL_OUT'])
        self.on_cluster = self._parse_bool(conf['TRAINING']['ON_CLUSTER'])

    def _parse_ckpt_frq(self, val):
        # NOTE assuming val is encoding either string or integer
        if val.lower() == 'epoch':
            return val.lower()
        else:
            return int(val)

    def _parse_bool(self, strb):
        if strb.lower() == 'false':
            return False
        else:
            return True

def main():
    # read config file from command line
    argparser = argparse.ArgumentParser(description="Config argument parser. See test.conf for an example on how the config file must be set.")
    argparser.add_argument('config', type=str, help="path to config file")
    args = argparser.parse_args()

    # make config object from file
    conf = Config(args.config)

    # print parsed config parameters
    def my_print(s):
        print("val: ", s, ", type: ", type(s))

    print("DATASET:")
    my_print(conf.dataset_files)
    my_print(conf.image_shape)
    my_print(conf.seq_len)
    my_print(conf.t0)
    my_print(conf.t1)
    my_print(conf.batch_size)
    my_print(conf.epoches)
    my_print(conf.validation_split)
    print("TRAINING")
    my_print(conf.model_file)
    my_print(conf.checkpoint_dir)
    my_print(conf.checkpoint_freq)
    my_print(conf.log_dir)
    my_print(conf.on_cluster)

if __name__ == "__main__":
    main()

