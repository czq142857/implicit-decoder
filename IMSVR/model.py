import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import h5py
import cv2
import mcubes

from ops import *

class IMSVR(object):
	def __init__(self, sess, is_training = False, z_dim=128, ef_dim=64, gf_dim=128, dataset_name='default', checkpoint_dir=None, pretrained_z_dir=None, sample_dir=None, data_dir='./data'):
		"""
		Args:
			too lazy to explain
		"""
		self.sess = sess

		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16*2)
		#3-- (64, 32*32*32)
		#4-- (128, 32*32*32*4)
		
		self.real_size = 64
		self.batch_size_input = 32*32*32
		self.batch_size = 16*16*16
		self.z_batch_size = 64
		
		self.view_size = 137
		self.crop_size = 128
		self.view_num = 20
		self.crop_edge = self.view_size-self.crop_size

		self.z_dim = z_dim

		self.ef_dim = ef_dim
		self.gf_dim = gf_dim

		self.dataset_name = dataset_name
		self.dataset_load = dataset_name + '_train'
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir
		
		if not is_training:
			self.test_size = 16
			self.batch_size = self.test_size*self.test_size*self.test_size
			self.dataset_load = dataset_name + '_test'

		if os.path.exists(self.data_dir+'/'+self.dataset_load+'.hdf5'):
			self.data_dict = h5py.File(self.data_dir+'/'+self.dataset_load+'.hdf5', 'r')
			self.data_points = self.data_dict['points_'+str(self.real_size)][:]
			self.data_values = self.data_dict['values_'+str(self.real_size)][:]
			self.data_pixel = self.data_dict['pixels'][:]
			data_dict_z = h5py.File(pretrained_z_dir, 'r')
			self.data_z = data_dict_z['zs'][:]
			if self.batch_size_input!=self.data_points.shape[1]:
				print("error: batch_size!=data_points.shape")
				exit(0)
			if self.view_num!=self.data_pixel.shape[1] or self.view_size!=self.data_pixel.shape[2]:
				print("error: view_size!=self.data_pixel.shape")
				exit(0)
		else:
			if is_training:
				print("error: cannot load "+self.data_dir+'/'+self.dataset_load+'.hdf5')
				exit(0)
			else:
				print("warning: cannot load "+self.data_dir+'/'+self.dataset_load+'.hdf5')
		
		self.build_model()
	
	def build_model(self):
		#for test
		self.point_coord = tf.placeholder(shape=[self.batch_size,3], dtype=tf.float32)
		self.z_vector_test = tf.placeholder(shape=[1,self.z_dim], dtype=tf.float32)
		self.view_test = tf.placeholder(shape=[1,self.crop_size,self.crop_size,1], dtype=tf.float32)
		
		#for train
		self.view = tf.placeholder(shape=[self.z_batch_size,self.crop_size,self.crop_size,1], dtype=tf.float32)
		self.z_vector = tf.placeholder(shape=[self.z_batch_size,self.z_dim], dtype=tf.float32)
		
		self.E = self.encoder(self.view, phase_train=True, reuse=False)
		
		self.sE = self.encoder(self.view_test, phase_train=False, reuse=True)
		self.zG = self.generator(self.point_coord, self.z_vector_test, phase_train=False, reuse=False)
		
		self.loss = tf.reduce_mean(tf.square(self.z_vector - self.E))
		self.vars = tf.trainable_variables()
		self.g_vars = [var for var in self.vars if 'simple_net' in var.name]
		self.e_vars = [var for var in self.vars if 'encoder' in var.name]
		
	
	
	def generator(self, points, z, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			zs = tf.tile(z, [self.batch_size,1])
			pointz = tf.concat([points,zs],1)
			print("pointz",pointz.shape)
			
			h1 = lrelu(linear(pointz, self.gf_dim*16, 'h1_lin'))
			h1 = tf.concat([h1,pointz],1)
			
			h2 = lrelu(linear(h1, self.gf_dim*8, 'h4_lin'))
			h2 = tf.concat([h2,pointz],1)
			
			h3 = lrelu(linear(h2, self.gf_dim*4, 'h5_lin'))
			h3 = tf.concat([h3,pointz],1)
			
			h4 = lrelu(linear(h3, self.gf_dim*2, 'h6_lin'))
			h4 = tf.concat([h4,pointz],1)
			
			h5 = lrelu(linear(h4, self.gf_dim, 'h7_lin'))
			h6 = tf.nn.sigmoid(linear(h5, 1, 'h8_lin'))
			
			return tf.reshape(h6, [self.batch_size,1])
	
	def encoder(self, view, phase_train=True, reuse=False):
		with tf.variable_scope("encoder") as scope:
			if reuse:
				scope.reuse_variables()
			
			#mimic resnet
			def resnet_block(input, dim_in, dim_out, scope):
				if dim_in == dim_out:
					output = conv2d_nobias(input, shape=[3, 3, dim_out, dim_out], strides=[1,1,1,1], scope=scope+'_1')
					output = batch_norm(output, phase_train)
					output = lrelu(output)
					output = conv2d_nobias(output, shape=[3, 3, dim_out, dim_out], strides=[1,1,1,1], scope=scope+'_2')
					output = batch_norm(output, phase_train)
					output = output + input
					output = lrelu(output)
				else:
					output = conv2d_nobias(input, shape=[3, 3, dim_in, dim_out], strides=[1,2,2,1], scope=scope+'_1')
					output = batch_norm(output, phase_train)
					output = lrelu(output)
					output = conv2d_nobias(output, shape=[3, 3, dim_out, dim_out], strides=[1,1,1,1], scope=scope+'_2')
					output = batch_norm(output, phase_train)
					input_ = conv2d_nobias(input, shape=[1, 1, dim_in, dim_out], strides=[1,2,2,1], scope=scope+'_3')
					input_ = batch_norm(input_, phase_train)
					output = output + input_
					output = lrelu(output)
				return output
			
			view = 1.0 - view
			layer_0 = conv2d_nobias(view, shape=[7, 7, 1, self.ef_dim], strides=[1,2,2,1], scope='conv0')
			layer_0 = batch_norm(layer_0, phase_train)
			layer_0 = lrelu(layer_0)
			#no maxpool
			
			layer_1 = resnet_block(layer_0, self.ef_dim, self.ef_dim, 'conv1')
			layer_2 = resnet_block(layer_1, self.ef_dim, self.ef_dim, 'conv2')
			
			layer_3 = resnet_block(layer_2, self.ef_dim, self.ef_dim*2, 'conv3')
			layer_4 = resnet_block(layer_3, self.ef_dim*2, self.ef_dim*2, 'conv4')
			
			layer_5 = resnet_block(layer_4, self.ef_dim*2, self.ef_dim*4, 'conv5')
			layer_6 = resnet_block(layer_5, self.ef_dim*4, self.ef_dim*4, 'conv6')
			
			layer_7 = resnet_block(layer_6, self.ef_dim*4, self.ef_dim*8, 'conv7')
			layer_8 = resnet_block(layer_7, self.ef_dim*8, self.ef_dim*8, 'conv8')
			
			layer_9 = conv2d_nobias(layer_8, shape=[4, 4, self.ef_dim*8, self.ef_dim*8], strides=[1,2,2,1], scope='conv9')
			layer_9 = batch_norm(layer_9, phase_train)
			layer_9 = lrelu(layer_9)
			
			layer_10 = conv2d(layer_9, shape=[4, 4, self.ef_dim*8, self.z_dim], strides=[1,1,1,1], scope='conv10', padding="VALID")
			layer_10 = tf.nn.sigmoid(layer_10)
			
			return tf.reshape(layer_10, [-1,self.z_dim])
	
	def train(self, config):
		'''
		#resume training
		ae_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss, var_list=self.e_vars)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=10)
		
		batch_index_list = np.arange(len(self.data_points))
		batch_idxs = int(len(self.data_points)/self.z_batch_size)
		batch_num = int(self.batch_size_input/self.batch_size)
		if self.batch_size_input%self.batch_size != 0:
			print("batch_size_input % batch_size != 0")
			exit(0)
		
		counter = 0
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter+1
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
		start_time = time.time()
		'''
		#first time run
		ae_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, ).minimize(self.loss, var_list=self.e_vars)
		self.sess.run(tf.global_variables_initializer())
		
		self.saver = tf.train.Saver(self.g_vars)
		could_load, checkpoint_counter = self.load_pretrained(config.pretrained_model_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			exit(0)
		
		batch_index_list = np.arange(len(self.data_points))
		batch_idxs = int(len(self.data_points)/self.z_batch_size)
		batch_num = int(self.batch_size_input/self.batch_size)
		if self.batch_size_input%self.batch_size != 0:
			print("batch_size_input % batch_size != 0")
			exit(0)
		self.saver = tf.train.Saver(max_to_keep=10)
		counter = 0
		start_time = time.time()
		
		
		batch_view = np.zeros([self.z_batch_size,self.crop_size,self.crop_size,1], np.float32)
		for epoch in range(counter, config.epoch):
			np.random.shuffle(batch_index_list)
			avg_loss = 0
			avg_num = 0
			for idx in range(0, batch_idxs):
				for t in range(self.z_batch_size):
					dxb = batch_index_list[idx*self.z_batch_size+t]
					which_view = np.random.randint(self.view_num)
					batch_view_ = self.data_pixel[dxb,which_view]
					offset_x = np.random.randint(self.crop_edge)
					offset_y = np.random.randint(self.crop_edge)
					if np.random.randint(2)==0:
						batch_view_ = batch_view_[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
					else:
						batch_view_ = np.flip(batch_view_[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size], 1)
					batch_view[t] = np.reshape(batch_view_/255.0, [self.crop_size,self.crop_size,1])
				batch_z = self.data_z[batch_index_list[idx*self.z_batch_size:(idx+1)*self.z_batch_size]]
				
				# Update AE network
				_, errAE = self.sess.run([ae_optim, self.loss],
					feed_dict={
						self.view: batch_view,
						self.z_vector: batch_z,
					})
				
				avg_loss += errAE
				avg_num += 1
				
				if idx==batch_idxs-1:
					print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f, avgloss: %.8f" % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errAE, avg_loss/avg_num))
					
					if epoch%10 == 0:
						
						sample_z = self.sess.run(self.sE,
							feed_dict={
								self.view_test: batch_view[0:1],
							})

						model_float = np.zeros([self.real_size,self.real_size,self.real_size],np.float32)
						real_model_float = np.zeros([self.real_size,self.real_size,self.real_size],np.float32)
						dxb = batch_index_list[idx*self.z_batch_size]
						for minib in range(batch_num):
							batch_points_int = self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
							batch_points = (batch_points_int+0.5)/self.real_size*2.0-1.0
							batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
							
							model_out = self.sess.run(self.zG,
								feed_dict={
									self.z_vector_test: sample_z,
									self.point_coord: batch_points,
								})
							model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(model_out, [self.batch_size])
							real_model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [self.batch_size])
						img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
						img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
						img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
						cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_1t.png",img1)
						cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_2t.png",img2)
						cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_3t.png",img3)
						img1 = np.clip(np.amax(real_model_float, axis=0)*256, 0,255).astype(np.uint8)
						img2 = np.clip(np.amax(real_model_float, axis=1)*256, 0,255).astype(np.uint8)
						img3 = np.clip(np.amax(real_model_float, axis=2)*256, 0,255).astype(np.uint8)
						cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_1i.png",img1)
						cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_2i.png",img2)
						cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_3i.png",img3)
						img1 = (np.reshape(batch_view[0:1], [self.crop_size,self.crop_size])*255).astype(np.uint8)
						cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_v.png",img1)
						print("[sample]")
						
						self.save(config.checkpoint_dir, epoch)
	
	def test_interp(self, config):
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		interp_size = 8
		
		idx1 = 0
		idx2 = 1
		
		thres = 0.6
		
		add_out = "./out/"
		
		dim = 128
		dima = self.test_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		
		#get coords 64
		aux_x = np.zeros([dima,dima,dima],np.int32)
		aux_y = np.zeros([dima,dima,dima],np.int32)
		aux_z = np.zeros([dima,dima,dima],np.int32)
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					aux_x[i,j,k] = i*multiplier
					aux_y[i,j,k] = j*multiplier
					aux_z[i,j,k] = k*multiplier
		coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
					coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
					coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
		coords = (coords+0.5)/dim*2.0-1.0
		coords = np.reshape(coords,[multiplier3,self.batch_size,3])
		
		offset_x = int(self.crop_edge/2)
		offset_y = int(self.crop_edge/2)
		batch_view1 = self.data_pixel[idx1,0]
		batch_view1 = batch_view1[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
		batch_view1 = np.reshape(batch_view1/255.0, [1,self.crop_size,self.crop_size,1])
		batch_view2 = self.data_pixel[idx2,0]
		batch_view2 = batch_view2[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
		batch_view2 = np.reshape(batch_view2/255.0, [1,self.crop_size,self.crop_size,1])
		
		model_z1 = self.sess.run(self.sE,
			feed_dict={
				self.view_test: batch_view1,
			})
		model_z2 = self.sess.run(self.sE,
			feed_dict={
				self.view_test: batch_view2,
			})
		
		batch_z = np.zeros([interp_size,self.z_dim], np.float32)
		for i in range(interp_size):
			batch_z[i] = model_z2*i/(interp_size-1) + model_z1*(interp_size-1-i)/(interp_size-1)
		
		for t in range(interp_size):
			model_float = np.zeros([dim+2,dim+2,dim+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector_test: batch_z[t],
								self.point_coord: coords[minib],
							})
						model_float[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out, [dima,dima,dima])
			
			vertices, triangles = mcubes.marching_cubes(model_float, thres)
			mcubes.export_mesh(vertices, triangles, add_out+str(t)+".dae", str(t))
			print("[sample]")

	def test(self, config):
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		thres = 0.6
		
		add_out = "./out/"
		add_image = "./image/"
		
		dim = 128
		dima = self.test_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		
		#get coords 64
		aux_x = np.zeros([dima,dima,dima],np.int32)
		aux_y = np.zeros([dima,dima,dima],np.int32)
		aux_z = np.zeros([dima,dima,dima],np.int32)
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					aux_x[i,j,k] = i*multiplier
					aux_y[i,j,k] = j*multiplier
					aux_z[i,j,k] = k*multiplier
		coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
					coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
					coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
		coords = (coords+0.5)/dim*2.0-1.0
		coords = np.reshape(coords,[multiplier3,self.batch_size,3])
		
		offset_x = int(self.crop_edge/2)
		offset_y = int(self.crop_edge/2)
		
		#test_num = self.data_pixel.shape[0]
		test_num = 16
		for t in range(test_num):
			print(t,test_num)
			
			batch_view = self.data_pixel[t,0]
			batch_view = batch_view[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
			batch_view = np.reshape(batch_view/255.0, [1,self.crop_size,self.crop_size,1])
			
			model_z = self.sess.run(self.sE,
				feed_dict={
					self.view_test: batch_view,
				})
			
			model_float = np.zeros([dim+2,dim+2,dim+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector_test: model_z,
								self.point_coord: coords[minib],
							})
						model_float[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out, [dima,dima,dima])
			'''
			img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)
			img1 = (np.reshape(batch_view, [self.crop_size,self.crop_size])*255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_v.png",img1)
			'''
			vertices, triangles = mcubes.marching_cubes(model_float, thres)
			mcubes.export_mesh(vertices, triangles, add_out+str(t)+".dae", str(t))
			
			cv2.imwrite(add_image+str(t)+".png", self.data_pixel[t,0])
			
			print("[sample]")

	def test_image(self, config):
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		thres = 0.6
		
		add_out = "./out/"
		add_image = "./image/"
		
		dim = 128
		dima = self.test_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		
		#get coords 64
		aux_x = np.zeros([dima,dima,dima],np.int32)
		aux_y = np.zeros([dima,dima,dima],np.int32)
		aux_z = np.zeros([dima,dima,dima],np.int32)
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					aux_x[i,j,k] = i*multiplier
					aux_y[i,j,k] = j*multiplier
					aux_z[i,j,k] = k*multiplier
		coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
					coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
					coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
		coords = (coords+0.5)/dim*2.0-1.0
		coords = np.reshape(coords,[multiplier3,self.batch_size,3])
		
		offset_x = int(self.crop_edge/2)
		offset_y = int(self.crop_edge/2)
		
		for t in range(16):
			img_add = add_image+str(t)+".png"
			print(img_add)
			imgo_ = cv2.imread(img_add, cv2.IMREAD_GRAYSCALE)

			img = cv2.resize(imgo_, (self.view_size,self.view_size))
			batch_view = np.reshape(img,(1,self.view_size,self.view_size,1))
			batch_view = batch_view[:, offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size, :]
			batch_view = batch_view/255.0
			
			model_z = self.sess.run(self.sE,
				feed_dict={
					self.view_test: batch_view,
				})
			
			model_float = np.zeros([dim+2,dim+2,dim+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector_test: model_z,
								self.point_coord: coords[minib],
							})
						model_float[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out, [dima,dima,dima])
			
			vertices, triangles = mcubes.marching_cubes(model_float, thres)
			mcubes.export_mesh(vertices, triangles, add_out+str(t)+".dae", str(t))
			
			print("[sample]")

	@property
	def model_dir(self):
		return "{}_{}".format(
				self.dataset_name, self.crop_size)
			
	def save(self, checkpoint_dir, step):
		model_name = "IMSVR.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

	def load_pretrained(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

