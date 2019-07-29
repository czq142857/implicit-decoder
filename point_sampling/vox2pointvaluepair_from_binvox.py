import numpy as np
import cv2
import os
import h5py
import binvox_rw
import random

class_name_list = [
"samplechair",
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

#hierarchical flood fill for 3D image: 32->64
def hierarchicalfloodFill(img64,dim):
	assert dim==64
	img64 = np.copy(img64)
	
	#compress model 64 -> 32
	dim_voxel = 32
	img32 = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
	multiplier = int(64/dim_voxel)
	for i in range(dim_voxel):
		for j in range(dim_voxel):
			for k in range(dim_voxel):
				img32[i,j,k] = np.max(img64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
	
	img32 = floodFill(img32, [(0,0,0),(31,0,0),(0,31,0),(31,31,0),(0,0,31),(31,0,31),(0,31,31),(31,31,31)], 32)
	for i in range(1,dim_voxel-1):
		for j in range(1,dim_voxel-1):
			for k in range(1,dim_voxel-1):
				occupied_flag = True
				for i0 in range(-1,2):
					for j0 in range(-1,2):
						for k0 in range(-1,2):
							if i0==0 and j0==0 and k0==0: continue
							if img32[i+i0,j+j0,k+k0]==0: occupied_flag = False
				if occupied_flag:
					img64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier] = np.ones([2,2,2],np.uint8)
	
	out64 = floodFill(img64, [(0,0,0),(63,0,0),(0,63,0),(63,63,0),(0,0,63),(63,0,63),(0,63,63),(63,63,63)], 64)
	
	print('filled', np.sum(np.abs(out64-img64)))
	return out64

#flood fill for 3D image
#input123: voxel model, initial points, size
#output1: voxel model, 0 for outside, 1 for inside
def floodFill(imgin, target_point_list,d):
	assert imgin.shape == (d,d,d)
	img = imgin+1
	queue = []
	for target_point in target_point_list:
		if img[target_point] == 1:
			img[target_point] = 0
			queue.append(target_point)
	while len(queue)>0:
		point = queue.pop(0)
		for i,j,k in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
			pi = point[0]+i
			pj = point[1]+j
			pk = point[2]+k
			if (pi<0 or pi>=d): continue
			if (pj<0 or pj>=d): continue
			if (pk<0 or pk>=d): continue
			if (img[pi,pj,pk] == 1):
				img[pi,pj,pk] = 0
				queue.append((pi,pj,pk))
	img = (img>0).astype(np.uint8)
	return img


for kkk in range(len(class_name_list)):
	if not os.path.exists(class_name_list[kkk]):
		os.makedirs(class_name_list[kkk])
	
	#class number
	class_name = class_name_list[kkk]
	print(class_name)
	
	#dir of voxel models
	voxel_input = "F:\\point_sampling\\data\\"+class_name+"\\"
	image_list = list_image(voxel_input, False, ['.binvox'])
	name_list = []
	for i in range(len(image_list)):
		imagine=image_list[i][0]
		name_list.append(imagine[0:-7])
	name_list = sorted(name_list)
	name_num = len(name_list)
	
	#record statistics
	fstatistics = open(class_name+'\\statistics.txt','w',newline='')
	exceed_32 = 0
	exceed_64 = 0

	dim = 64

	vox_size_1 = 16
	vox_size_2 = 32
	vox_size_3 = 64

	batch_size_1 = 16*16*16
	batch_size_2 = 16*16*16*2
	batch_size_3 = 32*32*32

	class_len_all_real = name_num

	hdf5_path = class_name+'\\'+class_name+'_vox.hdf5'
	fout = open(class_name+'\\'+class_name+'_vox.txt','w',newline='')
	
	hdf5_file = h5py.File(hdf5_path, 'w')
	hdf5_file.create_dataset("voxels", [class_len_all_real,dim,dim,dim,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_16", [class_len_all_real,batch_size_1,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_16", [class_len_all_real,batch_size_1,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_32", [class_len_all_real,batch_size_2,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_32", [class_len_all_real,batch_size_2,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_64", [class_len_all_real,batch_size_3,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_64", [class_len_all_real,batch_size_3,1], np.uint8, compression=9)


	counter = 0
	for idx in range(name_num):
		print(idx)
		#get voxel models
		try:
			voxel_model_file = open(voxel_input+name_list[idx]+".binvox", 'rb')
			voxel_model_64_crude = binvox_rw.read_as_3d_array(voxel_model_file).data.astype(np.uint8)
			#add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
			#Note stool is special!!!
			#voxel_model_64_crude = np.flip(np.transpose(voxel_model_64_crude, (1,2,0)),2)
			voxel_model_64 = hierarchicalfloodFill(voxel_model_64_crude, 64)
		except:
			print("error in loading")
			print(voxel_input+name_list[idx]+"\\model.binvox")
			exit(0)
		
		
		
		#compress model 64 -> 64
		dim_voxel = 64
		hdf5_file["voxels"][counter,:,:,:,:] = np.reshape(voxel_model_64, (dim_voxel,dim_voxel,dim_voxel,1))
		
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
		
		hdf5_file["points_64"][counter,:,:] = sample_points
		hdf5_file["values_64"][counter,:,:] = sample_values
		
		
		
		
		
		#compress model 64 -> 32
		dim_voxel = 32
		voxel_model_32 = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		multiplier = int(64/dim_voxel)
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					voxel_model_32[i,j,k] = np.max(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		
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
		
		hdf5_file["points_32"][counter,:,:] = sample_points
		hdf5_file["values_32"][counter,:,:] = sample_values
		
		#compress model 64 -> 16
		dim_voxel = 16
		voxel_model_16 = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		multiplier = int(64/dim_voxel)
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					voxel_model_16[i,j,k] = np.max(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
		
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
		
		hdf5_file["points_16"][counter,:,:] = sample_points
		hdf5_file["values_16"][counter,:,:] = sample_values
		
		fout.write(name_list[idx]+"\n")
		counter += 1
	
	assert counter==class_len_all_real
	
	fstatistics.write("total: "+str(class_len_all_real)+"\n")
	fstatistics.write("exceed_32: "+str(exceed_32)+"\n")
	fstatistics.write("exceed_32_ratio: "+str(float(exceed_32)/class_len_all_real)+"\n")
	fstatistics.write("exceed_64: "+str(exceed_64)+"\n")
	fstatistics.write("exceed_64_ratio: "+str(float(exceed_64)/class_len_all_real)+"\n")
	
	fout.close()
	fstatistics.close()
	hdf5_file.close()
	print("finished")


