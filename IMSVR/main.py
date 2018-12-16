import os
import scipy.misc
import numpy as np

from model import IMSVR

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1500, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.00005, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string("dataset", "02691156_hsp_vox", "The name of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("pretrained_model_dir", "./checkpoint/02691156_hsp_vox_only_train_64", "Root directory of pretrained_model")
flags.DEFINE_string("pretrained_z_dir", "./data/02691156_hsp_vox_only_train_z.hdf5", "Root directory of pretrained_model_z")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		imsvr = IMSVR(
				sess,
				is_training = FLAGS.train,
				dataset_name=FLAGS.dataset,
				checkpoint_dir=FLAGS.checkpoint_dir,
				pretrained_z_dir=FLAGS.pretrained_z_dir,
				sample_dir=FLAGS.sample_dir,
				data_dir=FLAGS.data_dir)

		#show_all_variables()

		if FLAGS.train:
			imsvr.train(FLAGS)
		else:
			imsvr.test(FLAGS)

if __name__ == '__main__':
	tf.app.run()
