# point sampling code for implicit-decoder

See [project page](https://www.sfu.ca/~zhiqinc/imgan/Readme.html) for detailed sampling method.

## Usage

In our paper we used the voxel data from HSP(https://github.com/chaene/hsp).

You can also use your own voxelized data. We recommand you to normalize each shape so that the diagonal of its bounding box equals to unit length. If you use binvox you are likely to place the shape in a unit cube so that the longest edge in its bounding box is unit length. This could cause some trouble since in the code we rarely sample points near the cube border.

If you have downloaded the voxelized data from HSP(https://github.com/chaene/hsp), please change the directories in the code and use the following command:
```
python vox2pointvaluepair_from_hsp.py
```

If you have prepared your own voxel data (64^3 .binvox), please change the directories in the code and use the following command:
```
python vox2pointvaluepair_from_binvox.py
```

You can visualize some shapes in the output hdf5 file by the following command, to see if the code works correctly:
```
python test_vox.py
```

Some samples are in folder "samplechair".