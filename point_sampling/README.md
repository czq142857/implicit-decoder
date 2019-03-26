# point sampling code for implicit-decoder

See [project page](https://www.sfu.ca/~zhiqinc/imgan/Readme.html) for detailed sampling method.

## Usage

In our paper we used the voxel data from HSP(https://github.com/chaene/hsp). You can also use your own voxelized data. We only tested our code on Windows. You may need to change some lines if you wish to run it on Ubuntu.

If you have downloaded the voxelized data from HSP(https://github.com/chaene/hsp), please change the directories in the code and use the following command:
```
python vox2pointvaluepair_from_hsp.py
```

If you prepared your own voxel data (.binvox), please change the directories in the code and use the following command:
```
python vox2pointvaluepair_from_binvox.py
```

You can visualize some shapes in the output hdf5 file by the following command, to see if the code works correctly:
```
python test_vox.py
```

Some samples are in folder "samplechair".