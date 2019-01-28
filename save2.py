import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pickle
import detectron.utils.densepose_methods as dp_utils
import time

im  = cv2.imread('../DensePoseData/demo_data/im.jpg')
IUV = cv2.imread('../DensePoseData/demo_data/IUV.png')
INDS = cv2.imread('../DensePoseData/demo_data/INDS.png',  0)

# Now read the smpl model.
with open('../DensePoseData/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
    data = pickle.load(f)
    Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
    X,Y,Z = [Vertices[:,0], Vertices[:,1],Vertices[:,2]]

DP = dp_utils.DensePoseMethods()

pick_idx = 1    # PICK PERSON INDEX!

C = np.where(INDS == pick_idx)
# C[0] is x-coords  np.array([23,  23,   24, ..])
# C[1] is y-coords  np.array([127, 128, 130, ..])
print('num pts on picked person:', C[0].shape)
IUV_pick = IUV[C[0], C[1], :]  # boolean indexing
IUV_pick = IUV_pick.astype(np.float)
IUV_pick[:, 1:3] = IUV_pick[:, 1:3] / 255.0
print(IUV_pick.shape)
collected_x = np.zeros(C[0].shape)
collected_y = np.zeros(C[0].shape)
collected_z = np.zeros(C[0].shape)

start = time.time()
# for i, (ii,uu,vv) in enumerate(zip(Demo['I'],Demo['U'],Demo['V'])):
for i in range(IUV_pick.shape[0]):
    # Convert IUV to FBC (faceIndex and barycentric coordinates.)
    FaceIndex,bc1,bc2,bc3 = DP.IUV2FBC(IUV_pick[i, 0], IUV_pick[i, 1], IUV_pick[i, 2])
    # Use FBC to get 3D coordinates on the surface.
    p = DP.FBC2PointOnSurface( FaceIndex, bc1,bc2,bc3,Vertices )
    #
    collected_x[i] = p[0]
    collected_y[i] = p[1]
    collected_z[i] = p[2]
print(time.time() - start , 'secs')

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





















