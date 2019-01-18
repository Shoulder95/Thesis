import detectron.utils.densepose_methods as dp_utils
import pickle
import numpy as np


DP = dp_utils.DensePoseMethods()
pkl_file = open('../DensePoseData/demo_data/demo_dp_single_ann.pkl', 'rb')
Demo = pickle.load(pkl_file)

collected_x = np.zeros(Demo['x'].shape)
collected_y = np.zeros(Demo['x'].shape)
collected_z = np.zeros(Demo['x'].shape)

for i, (ii,uu,vv) in enumerate(zip(Demo['I'],Demo['U'],Demo['V'])):
    # Convert IUV to FBC (faceIndex and barycentric coordinates.)
    FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(ii,uu,vv)
    # Use FBC to get 3D coordinates on the surface.
    p = DP.FBC2PointOnSurface( FaceIndex, bc1,bc2,bc3,Vertices )
    #
    collected_x[i] = p[0]
    collected_y[i] = p[1]
    collected_z[i] = p[2]


collected_x = [ 1, 2, 3]
collected_y = [4,5,6]
collected_z = [7,8,9]
f = open("points.txt","w")
i = 0
while i < len(collected_x):
	f.write(str(collected_x[i]))
	f.write(", ")
	f.write(str(collected_y[i]))
	f.write(", ")
	f.write(str(collected_z[i]))
	f.write("\n")
	i += 1
f.close()
