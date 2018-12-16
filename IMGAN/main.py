import os
import scipy.misc
import numpy as np

from model import IMAE
from modelz import ZGAN

import tensorflow as tf
import h5py

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.00005, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string("dataset", "03001627_vox", "The name of dataset")
flags.DEFINE_integer("real_size", 64, "output point-value voxel grid size in training [64]")
flags.DEFINE_integer("batch_size_input", 32768, "training batch size (virtual, batch_size is the real batch_size) [32768]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("ae", False, "True for AE, False for zGAN [False]")
FLAGS = flags.FLAGS

def main(_):
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	#run_config = tf.ConfigProto(gpu_options=gpu_options)
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	if FLAGS.ae:
		with tf.Session(config=run_config) as sess:
			imae = IMAE(
					sess,
					FLAGS.real_size,
					FLAGS.batch_size_input,
					is_training = FLAGS.train,
					dataset_name=FLAGS.dataset,
					checkpoint_dir=FLAGS.checkpoint_dir,
					sample_dir=FLAGS.sample_dir,
					data_dir=FLAGS.data_dir)

			if FLAGS.train:
				imae.train(FLAGS)
			else:
				imae.get_z(FLAGS)
				#imae.test_interp(FLAGS)
				#imae.test(FLAGS)
	else:
		if FLAGS.train:
			with tf.Session(config=run_config) as sess_z:
				zgan = ZGAN(
						sess_z,
						is_training = FLAGS.train,
						dataset_name=FLAGS.dataset,
						checkpoint_dir=FLAGS.checkpoint_dir,
						sample_dir=FLAGS.sample_dir,
						data_dir=FLAGS.data_dir)
				zgan.train(FLAGS)
		else:
			
			#option 1 generate z
			with tf.Session(config=run_config) as sess_z:
				zgan = ZGAN(
						sess_z,
						is_training = FLAGS.train,
						dataset_name=FLAGS.dataset,
						checkpoint_dir=FLAGS.checkpoint_dir,
						sample_dir=FLAGS.sample_dir,
						data_dir=FLAGS.data_dir)
				generated_z = zgan.get_z(FLAGS, 16)
			tf.reset_default_graph()
			'''
			hdf5_file = h5py.File("temp_z.hdf5", mode='w')
			hdf5_file.create_dataset("zs", generated_z.shape, np.float32)
			hdf5_file["zs"][...] = generated_z
			hdf5_file.close()
			'''
			with tf.Session(config=run_config) as sess:
				imae = IMAE(
						sess,
						FLAGS.real_size,
						FLAGS.batch_size_input,
						is_training = FLAGS.train,
						dataset_name=FLAGS.dataset,
						checkpoint_dir=FLAGS.checkpoint_dir,
						sample_dir=FLAGS.sample_dir,
						data_dir=FLAGS.data_dir)
				imae.test_z(FLAGS, generated_z, 128)
			
			'''
			#option 2 use filtered z
			hdf5_file = h5py.File("temp_z.hdf5", mode='r')
			generated_z = hdf5_file["zs"][:]
			hdf5_file.close()
			z_num = generated_z.shape[0]
			filtered_z = np.copy(generated_z)
			t = 0
			for tt in range(z_num):
				if (os.path.exists(FLAGS.sample_dir+'/'+str(tt)+'_1t.png')):
					filtered_z[t] = generated_z[tt]
					t += 1
			filtered_z = filtered_z[:t]
			print('filtered',t)
			
			with tf.Session(config=run_config) as sess:
				imae = IMAE(
						sess,
						is_training = FLAGS.train,
						dataset_name=FLAGS.dataset,
						checkpoint_dir=FLAGS.checkpoint_dir,
						sample_dir=FLAGS.sample_dir,
						data_dir=FLAGS.data_dir)
				imae.test_z(FLAGS, filtered_z, 256)
			'''

if __name__ == '__main__':
	tf.app.run()
