
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import numpy as np
import pickle


def smpl_view_set_axis_full_body(ax,azimuth=0):
    ## Manually set axis 
    ax.view_init(0, azimuth)
    max_range = 0.55
    ax.set_xlim( - max_range,   max_range)
    ax.set_ylim( - max_range,   max_range)
    ax.set_zlim( -0.2 - max_range,   -0.2 + max_range)
    ax.axis('off')
    
def smpl_view_set_axis_face(ax, azimuth=0):
    ## Manually set axis 
    ax.view_init(0, azimuth)
    max_range = 0.1
    ax.set_xlim( - max_range,   max_range)
    ax.set_ylim( - max_range,   max_range)
    ax.set_zlim( 0.45 - max_range,   0.45 + max_range)
    ax.axis('off')

f = open("points.txt","r")
collected_x = []
collected_y = []
collected_z = []
i = 0
for line in f:
	l = line.strip("\n")
	collected_x[i] = int(l.split(",")[0])
	collected_y[i] = int(l.split(",")[1])
	collected_z[i] = int(l.split(",")[2])
f.close()

pkl_file = open('../DensePoseData/demo_data/demo_dp_single_ann.pkl', 'rb')
Demo = pickle.load(pkl_file)
fig = plt.figure(figsize=[15,5])

# Visualize the image and collected points.
ax = fig.add_subplot(131)
ax.imshow(Demo['ICrop'])
ax.scatter(Demo['x'],Demo['y'],11, np.arange(len(Demo['y']))  )
plt.title('Points on the image')
ax.axis('off'), 

## Visualize the full body smpl male template model and collected points
ax = fig.add_subplot(132, projection='3d')
ax.scatter(Z,X,Y,s=0.02,c='k')
ax.scatter(collected_z,  collected_x,collected_y,s=25,  c=  np.arange(len(Demo['y']))    )
smpl_view_set_axis_full_body(ax)
plt.title('Points on the SMPL model')

## Now zoom into the face.
ax = fig.add_subplot(133, projection='3d')
ax.scatter(Z,X,Y,s=0.2,c='k')
ax.scatter(collected_z,  collected_x,collected_y,s=55,c=np.arange(len(Demo['y'])))
smpl_view_set_axis_face(ax)
plt.title('Points on the SMPL model')
#
plt.show()
