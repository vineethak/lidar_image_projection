import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d

def quaternion_to_matrix(x, y, z, w):
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return matrix

'''Folder structure 
   |
    Parent folder
        |
        |Image folder
        |Lidar folder
'''

path = "/media/dheeraj/c702663a-a4ab-4510-aef9-96645e129569/vattikonda/vineetha_datasets/tuesday_afternoon/extracted/"

sensor = 'camera'
img_filename = '1710878890205040896.png'
img_filepath = os.path.join(path,sensor,img_filename)

# Load the camera image
camera_image = cv2.imread(img_filepath)
camera_image_rgb = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

# rostopic /sensor_msgs/camera_info
# matrix form [[fx, 0, cx],
#              [0,  fy, cy],
#              [0, 0,   1]]       
camera_matrix = np.array([[1297.672904,0.0,620.914026],
                                [0.0, 1298.631344, 238.280325],
                                [0.0, 0.0,1.0,]])

dist_coeffs = np.zeros((5, 1))

sensor = "lidar"
filename = "1710878890155900416.csv"
filepath = os.path.join(path, sensor, filename)

# Load the LiDAR data
data = pd.read_csv(filepath)
lidar_points = data[['x', 'y', 'z']].values # Shape: (N, 3) for N points with (x, y, z)

# extrinsic matrix obtained from lookup trnasform
x,y,z,w = 0.46627404384091853, 0.4663075237828769, -0.5315897169831801, 0.5315620209359032
rmat = quaternion_to_matrix(x,y,z,w)
rvec,_ = cv2.Rodrigues(rmat)
tvec = np.array([0.0013692428073066587, 0.5182643825008042, 0.026478499929228433], dtype=np.float32)  # Translation vector (example values)


# convert np array to pcd o3d format to visualize the point cloud
# use this to identify the points behind the camera and filter them
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(lidar_points)
# colors = np.zeros((lidar_points.shape))
# colors[lidar_points[:,2] > 0] = [0,1,0]
# pcd.colors  = o3d.utility.Vector3dVector(colors)

# o3d.visualization.draw_geometries([pcd])

valid_indices = lidar_points[:,0] > 0 #filtering the points infront of the camera
valid_lidar_points = lidar_points[valid_indices]

# Project LiDAR points onto the 2D camera plane using OpenCV
lidar_points = lidar_points.astype(np.float32)
image_points, _ = cv2.projectPoints(lidar_points, rvec, tvec, camera_matrix, dist_coeffs)
valid_image_points, _ = cv2.projectPoints(valid_lidar_points, rvec, tvec, camera_matrix, dist_coeffs)

# Reshape image_points to remove the extra dimension
image_points = image_points.squeeze()
valid_image_points = valid_image_points.squeeze()


# Draw the points on the camera image
overlay_image = camera_image_rgb.copy()
for i, (u, v) in enumerate(valid_image_points):
    x, y = int(u), int(v)
    if ( (0 <= x < overlay_image.shape[1] and 0 <= y < overlay_image.shape[0])):
        cv2.circle(overlay_image, (x, y), 2, (255, 0, 0), -1)  # Red color

# Display the image with the overlay 
cv2.imshow('overlay_image', overlay_image)

# Function to color LiDAR points according to camera pixels
def color_lidar_points(lidar_points_2d,valid_index, image):
    h, w, _ = image.shape
    count = 0
    lidar_colors = np.zeros((lidar_points_2d.shape[0], 3))
    for i, (x, y) in enumerate(lidar_points_2d):
        # camera is facing -x direction
        if ((valid_index[i] == False) and (0 <= int(y) < h and 0 <= int(x) < w)):
            #print(x,y)
            lidar_colors[i] = image[int(y), int(x)] / 255.0
            count += 1
        else:
            lidar_colors[i] = [0, 0, 0]
    print(count)
    return lidar_colors

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_points)
colors = color_lidar_points(image_points, valid_indices, camera_image_rgb)
pcd.colors  = o3d.utility.Vector3dVector(colors)
bbox_min   = np.min(valid_lidar_points, axis = 0)
bbox_max   = np.max(valid_lidar_points, axis = 0)
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
#pcd.bounding_box = bbox
o3d.visualization.draw_geometries([pcd, bbox])