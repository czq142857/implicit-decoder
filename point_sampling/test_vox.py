import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import h5py
import cv2
import mcubes

data_dict = h5py.File('samplechair/samplechair_vox.hdf5', 'r')
data_points16 = data_dict['points_16'][:]
data_values16 = data_dict['values_16'][:]
data_points32 = data_dict['points_32'][:]
data_values32 = data_dict['values_32'][:]
data_points64 = data_dict['points_64'][:]
data_values64 = data_dict['values_64'][:]
data_voxels = data_dict['voxels'][:]


dxb = 0

batch_voxels = data_voxels[dxb:dxb+1]
batch_voxels = np.reshape(batch_voxels,[64,64,64])
img1 = np.clip(np.amax(batch_voxels, axis=0)*256, 0,255).astype(np.uint8)
img2 = np.clip(np.amax(batch_voxels, axis=1)*256, 0,255).astype(np.uint8)
img3 = np.clip(np.amax(batch_voxels, axis=2)*256, 0,255).astype(np.uint8)
cv2.imwrite(str(dxb)+"_vox_1.png",img1)
cv2.imwrite(str(dxb)+"_vox_2.png",img2)
cv2.imwrite(str(dxb)+"_vox_3.png",img3)
vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
mcubes.export_mesh(vertices, triangles, str(dxb)+"_vox.dae", str(dxb))


batch_points_int = data_points16[dxb,:]
batch_values = data_values16[dxb,:]
real_model = np.zeros([16,16,16],np.uint8)
real_model[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
img1 = np.clip(np.amax(real_model, axis=0)*256, 0,255).astype(np.uint8)
img2 = np.clip(np.amax(real_model, axis=1)*256, 0,255).astype(np.uint8)
img3 = np.clip(np.amax(real_model, axis=2)*256, 0,255).astype(np.uint8)
cv2.imwrite(str(dxb)+"_16_1.png",img1)
cv2.imwrite(str(dxb)+"_16_2.png",img2)
cv2.imwrite(str(dxb)+"_16_3.png",img3)
vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
mcubes.export_mesh(vertices, triangles, str(dxb)+"_16.dae", str(dxb))


batch_points_int = data_points32[dxb,:]
batch_values = data_values32[dxb,:]
real_model = np.zeros([32,32,32],np.uint8)
real_model[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
img1 = np.clip(np.amax(real_model, axis=0)*256, 0,255).astype(np.uint8)
img2 = np.clip(np.amax(real_model, axis=1)*256, 0,255).astype(np.uint8)
img3 = np.clip(np.amax(real_model, axis=2)*256, 0,255).astype(np.uint8)
cv2.imwrite(str(dxb)+"_32_1.png",img1)
cv2.imwrite(str(dxb)+"_32_2.png",img2)
cv2.imwrite(str(dxb)+"_32_3.png",img3)
vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
mcubes.export_mesh(vertices, triangles, str(dxb)+"_32.dae", str(dxb))


batch_points_int = data_points64[dxb,:]
batch_values = data_values64[dxb,:]
real_model = np.zeros([64,64,64],np.uint8)
real_model[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [-1])
img1 = np.clip(np.amax(real_model, axis=0)*256, 0,255).astype(np.uint8)
img2 = np.clip(np.amax(real_model, axis=1)*256, 0,255).astype(np.uint8)
img3 = np.clip(np.amax(real_model, axis=2)*256, 0,255).astype(np.uint8)
cv2.imwrite(str(dxb)+"_64_1.png",img1)
cv2.imwrite(str(dxb)+"_64_2.png",img2)
cv2.imwrite(str(dxb)+"_64_3.png",img3)
vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
mcubes.export_mesh(vertices, triangles, str(dxb)+"_64.dae", str(dxb))
