#from ipywidgets import interactive, fixed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import numpy as np
import cv2
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

im  = cv2.imread('../Thesis/12_infer/12.jpg')
#print(im)
IUV = cv2.imread('../Thesis/12_infer/12_IUV.png')
#print(IUV)
INDS = cv2.imread('../Thesis/12_infer/12_INDS.png',  0)

pkl_file = open('../Thesis/12_infer/test_vis_12.pkl', 'rb')
Demo = pickle.load(pkl_file)
#print(Demo['im'])
#print(Demo['cls_boxes'])

f = open("points.txt","r")
collected_x = []
collected_y = []
collected_z = []
i = 0
for line in f:
	l = line.strip("\n")
	#print(l.split(",")[0])
	collected_x.append(float(l.split(",")[0]))
	collected_y.append(float(l.split(",")[1]))
	collected_z.append(float(l.split(",")[2]))
f.close()

# Now read the smpl model.
with open('../Thesis/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', 'rb') as f:
    data = pickle.load(f)
    Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
    X,Y,Z = [Vertices[:,0], Vertices[:,1],Vertices[:,2]]



pick_idx = 1
# Color of each (U,V) point.
C = np.where(INDS == pick_idx)
# C[0] is x-coords  np.array([23,  23,   24, ..])
# C[1] is y-coords  np.array([127, 128, 130, ..])
print('num pts on picked person:', C[0].shape)
#print(C[0])
#print(C[1])
#person_color = im[C[0], C[1], ::-1]  # boolean indexing.   ::-1 to make the RGB/BGR convention suitable for plotting.
#print(person_color.shape)

## Visualization with colors
#fig = plt.figure(figsize=[20,10])

## Visualize the full body smpl male template model and collected points
#ax = fig.add_subplot(121, projection='3d')
## ax.scatter(Z,X,Y,s=0.02,c='k')
#ax.scatter(collected_z,  collected_x,collected_y,  c= person_color/255.0   )  ##s=__ size
#smpl_view_set_axis_full_body(ax)
#plt.title('Points on the SMPL model')

## Now zoom into the face.
#ax = fig.add_subplot(122, projection='3d')
## ax.scatter(Z,X,Y,s=0.2,c='k')
#ax.scatter(collected_z,  collected_x,collected_y,c=person_color/255.0) #s=__ size
#smpl_view_set_axis_face(ax)
#plt.title('Points on the SMPL model')
#plt.show()

#print('num pts on picked person:', C[0].shape)
IUV_pick = IUV[C[0], C[1], :]  # boolean indexing
IUV_pick = IUV_pick.astype(np.float)
IUV_pick[:, 1:3] = IUV_pick[:, 1:3] / 255.0
#print(IUV_pick.shape)

print("Plotting...")

fig = plt.figure(figsize=[15,5])

cls_keyps = Demo['kp']
keyps = [k for klist in cls_keyps for k in klist]
kps = keyps[0]
x = [kps[0,i] for i in range(len(kps[0]))]
y = [kps[1,i] for i in range(len(kps[1]))]

# Visualize the image and collected points.
ax = fig.add_subplot(131)
ax.imshow(Demo['im'])
ax.scatter(x,y,11, np.arange(len(y))  )
plt.title('Points on the image')
ax.axis('off'), 

## Visualize the full body smpl male template model and collected points
ax = fig.add_subplot(132, projection='3d')
ax.scatter(Z,X,Y,s=0.02,c='k')
ax.scatter(collected_z,  collected_x,collected_y,s=25,  c=  np.arange(len(collected_y))    )
smpl_view_set_axis_full_body(ax)
plt.title('Points on the SMPL model')

## Now zoom into the face.
ax = fig.add_subplot(133, projection='3d')
ax.scatter(Z,X,Y,s=0.2,c='k')
ax.scatter(collected_z,  collected_x,collected_y,s=55,c=np.arange(len(collected_y)))
smpl_view_set_axis_face(ax)
plt.title('Points on the SMPL model')
#
plt.show()


















