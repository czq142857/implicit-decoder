import numpy as np
import cv2
import os
import h5py
from scipy.io import loadmat
import random
import json

class_name_list = [
"02828884_bench", 
"04090263_rifle", 
"04256520_couch", 
"04530566_vessel",
"02958343_car", 
]

def list_image(root, recursive, exts):
	image_list = []
	cat = {}
	for path, subdirs, files in os.walk(root):
		for fname in files:
			fpath = os.path.join(path, fname)
			suffix = os.path.splitext(fname)[1].lower()
			if os.path.isfile(fpath) and (suffix in exts):
				if path not in cat:
					cat[path] = len(cat)
				image_list.append((os.path.relpath(fpath, root), cat[path]))
	return image_list


for kkk in range(len(class_name_list)):
	if not os.path.exists(class_name_list[kkk]):
		os.makedirs(class_name_list[kkk])
	
	#class number
	class_name = class_name_list[kkk][:8]
	print(class_name_list[kkk])
	
	#dir of voxel models
	voxel_input = "F:\\hsp\\shapenet\\modelBlockedVoxels256\\"+class_name+"\\"
	
	#name of output file
	hdf5_path = class_name_list[kkk]+'\\'+class_name+'_vox.hdf5'

	#obj_list
	fout = open(class_name_list[kkk]+'\\obj_list.txt','w',newline='')
	
	#record statistics
	fstatistics = open(class_name_list[kkk]+'\\statistics.txt','w',newline='')
	exceed_32 = 0
	exceed_64 = 0


	dim = 64

	vox_size_1 = 16
	vox_size_2 = 32
	vox_size_3 = 64

	batch_size_1 = 16*16*16
	batch_size_2 = 16*16*16*2
	batch_size_3 = 32*32*32


	image_list = list_image(voxel_input, True, ['.mat'])
	name_list = []
	for i in range(len(image_list)):
		imagine=image_list[i][0]
		name_list.append(imagine[0:-4])
	name_list = sorted(name_list)
	name_num = len(name_list)
	#name_list contains all obj names
	
	hdf5_file = h5py.File(hdf5_path, 'w')
	hdf5_file.create_dataset("voxels", [name_num,dim,dim,dim,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_16", [name_num,batch_size_1,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_16", [name_num,batch_size_1,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_32", [name_num,batch_size_2,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_32", [name_num,batch_size_2,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_64", [name_num,batch_size_3,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_64", [name_num,batch_size_3,1], np.uint8, compression=9)

	
	for idx in range(name_num):
		print(idx)
		
		#get voxel models
		name_list_idx = name_list[idx]
		proper_name = "shape_data/"+class_name+"/"+name_list_idx
		try:
			voxel_model_mat = loadmat(voxel_input+name_list_idx+".mat")
		except:
			print("error in loading")
			name_list_idx = name_list[idx-1]
			try:
				voxel_model_mat = loadmat(voxel_input+name_list_idx+".mat")
			except:
				print("error in loading")
				name_list_idx = name_list[idx-2]
				try:
					voxel_model_mat = loadmat(voxel_input+name_list_idx+".mat")
				except:
					print("error in loading")
					exit(0)
		
		
		
		voxel_model_b = voxel_model_mat['b'][:].astype(np.int32)
		voxel_model_bi = voxel_model_mat['bi'][:].astype(np.int32)-1
		voxel_model_256 = np.zeros([256,256,256],np.uint8)
		for i in range(16):
			for j in range(16):
				for k in range(16):
					voxel_model_256[i*16:i*16+16,j*16:j*16+16,k*16:k*16+16] = voxel_model_b[voxel_model_bi[i,j,k]]
		#add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
		voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2,1,0)),2)
		
		
		
		
		
		#compress model 256 -> 64
		dim_voxel = 64
		voxel_model_64 = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		multiplier = int(256/dim_voxel)
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					voxel_model_64[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		hdf5_file["voxels"][idx,:,:,:,:] = np.reshape(voxel_model_64, (dim_voxel,dim_voxel,dim_voxel,1))
		
		#sample points near surface
		batch_size = batch_size_3
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		voxel_model_64_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		for i in range(3,dim_voxel-3):
			if (batch_size_counter>=batch_size): break
			for j in range(3,dim_voxel-3):
				if (batch_size_counter>=batch_size): break
				for k in range(3,dim_voxel-3):
					if (batch_size_counter>=batch_size): break
					if (np.max(voxel_model_64[i-3:i+4,j-3:j+4,k-3:k+4])!=np.min(voxel_model_64[i-3:i+4,j-3:j+4,k-3:k+4])):
						sample_points[batch_size_counter,0] = i
						sample_points[batch_size_counter,1] = j
						sample_points[batch_size_counter,2] = k
						sample_values[batch_size_counter,0] = voxel_model_64[i,j,k]
						voxel_model_64_flag[i,j,k] = 1
						batch_size_counter +=1
		if (batch_size_counter>=batch_size):
			print("64-- batch_size exceeded!")
			exceed_64 += 1
			batch_size_counter = 0
			voxel_model_64_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
			for i in range(0,dim_voxel,2):
				for j in range(0,dim_voxel,2):
					for k in range(0,dim_voxel,2):
						filled_flag = False
						for (i0,j0,k0) in [(i,j,k),(i+1,j,k),(i,j+1,k),(i+1,j+1,k),(i,j,k+1),(i+1,j,k+1),(i,j+1,k+1),(i+1,j+1,k+1)]:
							if voxel_model_64[i0,j0,k0]>0:
								filled_flag = True
								sample_points[batch_size_counter,0] = i0
								sample_points[batch_size_counter,1] = j0
								sample_points[batch_size_counter,2] = k0
								sample_values[batch_size_counter,0] = voxel_model_64[i0,j0,k0]
								voxel_model_64_flag[i0,j0,k0] = 1
								break
						if not filled_flag:
							sample_points[batch_size_counter,0] = i
							sample_points[batch_size_counter,1] = j
							sample_points[batch_size_counter,2] = k
							sample_values[batch_size_counter,0] = voxel_model_64[i,j,k]
							voxel_model_64_flag[i,j,k] = 1
						batch_size_counter +=1
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_64_flag[i,j,k] != 1: break
				sample_points[batch_size_counter,0] = i
				sample_points[batch_size_counter,1] = j
				sample_points[batch_size_counter,2] = k
				sample_values[batch_size_counter,0] = voxel_model_64[i,j,k]
				voxel_model_64_flag[i,j,k] = 1
				batch_size_counter +=1
		else:
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_64_flag[i,j,k] != 1: break
				sample_points[batch_size_counter,0] = i
				sample_points[batch_size_counter,1] = j
				sample_points[batch_size_counter,2] = k
				sample_values[batch_size_counter,0] = voxel_model_64[i,j,k]
				voxel_model_64_flag[i,j,k] = 1
				batch_size_counter +=1
		
		hdf5_file["points_64"][idx,:,:] = sample_points
		hdf5_file["values_64"][idx,:,:] = sample_values
		
		
		
		
		
		#compress model 256 -> 32
		dim_voxel = 32
		voxel_model_32 = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		multiplier = int(256/dim_voxel)
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					voxel_model_32[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		
		#sample points near surface
		batch_size = batch_size_2
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		voxel_model_32_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		for i in range(3,dim_voxel-3):
			if (batch_size_counter>=batch_size): break
			for j in range(3,dim_voxel-3):
				if (batch_size_counter>=batch_size): break
				for k in range(3,dim_voxel-3):
					if (batch_size_counter>=batch_size): break
					if (np.max(voxel_model_32[i-3:i+4,j-3:j+4,k-3:k+4])!=np.min(voxel_model_32[i-3:i+4,j-3:j+4,k-3:k+4])):
						sample_points[batch_size_counter,0] = i
						sample_points[batch_size_counter,1] = j
						sample_points[batch_size_counter,2] = k
						sample_values[batch_size_counter,0] = voxel_model_32[i,j,k]
						voxel_model_32_flag[i,j,k] = 1
						batch_size_counter +=1
		if (batch_size_counter>=batch_size):
			print("32-- batch_size exceeded!")
			exceed_32 += 1
			batch_size_counter = 0
			voxel_model_32_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
			for i in range(0,dim_voxel,2):
				for j in range(0,dim_voxel,2):
					for k in range(0,dim_voxel,2):
						filled_flag = False
						for (i0,j0,k0) in [(i,j,k),(i+1,j,k),(i,j+1,k),(i+1,j+1,k),(i,j,k+1),(i+1,j,k+1),(i,j+1,k+1),(i+1,j+1,k+1)]:
							if voxel_model_32[i0,j0,k0]>0:
								filled_flag = True
								sample_points[batch_size_counter,0] = i0
								sample_points[batch_size_counter,1] = j0
								sample_points[batch_size_counter,2] = k0
								sample_values[batch_size_counter,0] = voxel_model_32[i0,j0,k0]
								voxel_model_32_flag[i0,j0,k0] = 1
								break
						if not filled_flag:
							sample_points[batch_size_counter,0] = i
							sample_points[batch_size_counter,1] = j
							sample_points[batch_size_counter,2] = k
							sample_values[batch_size_counter,0] = voxel_model_32[i,j,k]
							voxel_model_32_flag[i,j,k] = 1
						batch_size_counter +=1
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_32_flag[i,j,k] != 1: break
				sample_points[batch_size_counter,0] = i
				sample_points[batch_size_counter,1] = j
				sample_points[batch_size_counter,2] = k
				sample_values[batch_size_counter,0] = voxel_model_32[i,j,k]
				voxel_model_32_flag[i,j,k] = 1
				batch_size_counter +=1
		else:
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_32_flag[i,j,k] != 1: break
				sample_points[batch_size_counter,0] = i
				sample_points[batch_size_counter,1] = j
				sample_points[batch_size_counter,2] = k
				sample_values[batch_size_counter,0] = voxel_model_32[i,j,k]
				voxel_model_32_flag[i,j,k] = 1
				batch_size_counter +=1
		
		hdf5_file["points_32"][idx,:,:] = sample_points
		hdf5_file["values_32"][idx,:,:] = sample_values
		
		
		
		
		
		#compress model 256 -> 16
		dim_voxel = 16
		voxel_model_16 = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		multiplier = int(256/dim_voxel)
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					voxel_model_16[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		
		#sample points near surface
		batch_size = batch_size_1
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					sample_points[batch_size_counter,0] = i
					sample_points[batch_size_counter,1] = j
					sample_points[batch_size_counter,2] = k
					sample_values[batch_size_counter,0] = voxel_model_16[i,j,k]
					batch_size_counter +=1
		if (batch_size_counter!=batch_size):
			print("batch_size_counter!=batch_size")
		
		hdf5_file["points_16"][idx,:,:] = sample_points
		hdf5_file["values_16"][idx,:,:] = sample_values
		
		fout.write(name_list_idx+"\n")
	
	fstatistics.write("non-repeat total: "+str(name_num)+"\n")
	fstatistics.write("exceed_32: "+str(exceed_32)+"\n")
	fstatistics.write("exceed_32_ratio: "+str(float(exceed_32)/name_num)+"\n")
	fstatistics.write("exceed_64: "+str(exceed_64)+"\n")
	fstatistics.write("exceed_64_ratio: "+str(float(exceed_64)/name_num)+"\n")
	
	fout.close()
	fstatistics.close()
	hdf5_file.close()
	print("finished")


