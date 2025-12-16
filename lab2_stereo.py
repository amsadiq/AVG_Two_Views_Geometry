import numpy as np
np.typeDict = np.sctypeDict  # compatibility alias for packages expecting np.typeDict
np.int = int  # compatibility alias for packages expecting np.int
np.float = float  # compatibility alias for packages expecting np.float
import cv2
import open3d as o3d
import os


K = np.array([
    [7.215377e2,        0,              6.095593e2],
    [0,                 7.215377e2,     1.728540e2],
    [0,                 0,              1]
])  # intrinsic matrix of camera

Bf = 3.875744e+02  # base line * focal length

# load images
left = cv2.imread('/Users/sadiq/Desktop/M2/Lecture notes/AVG/AVG_Two_Views_Geometry/scripts_and_file/img/left.png', 0)
right = cv2.imread('/Users/sadiq/Desktop/M2/Lecture notes/AVG/AVG_Two_Views_Geometry/scripts_and_file/img/right.png', 0)
left_color = cv2.imread('/Users/sadiq/Desktop/M2/Lecture notes/AVG/AVG_Two_Views_Geometry/scripts_and_file/img/left_color.png')

# compute disparity
stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=5)
#stereo = cv2.StereoSGBM_create(numDisparities=192, blockSize=5, P1=600, P2=2400)
disparity = stereo.compute(left, right)  # an image of the same size with "left" (or "right")
disparity = disparity.astype(np.float32) / 16.0  # StereoBM output is fixed-point scaled by 16

try:
    print('Disparity min:', float(disparity.min()), 'max:', float(disparity.max()), 'mean:', float(disparity.mean()))
    print('Valid disparity pixels:', int((disparity > 0).sum()), 'of', disparity.size)
except Exception:
    pass

# TODO: compute depth of every pixel whose disparity is positive
# hint: assume d is the disparity of pixel (u, v)
# hint: the depth Z of this pixel is Z = Bf / d
valid = disparity > 0
depth = np.zeros_like(disparity, dtype=np.float32)
np.divide(Bf, disparity, out=depth, where=valid)
print("Depth map (zeros where disparity>=5):\n", depth)

# TODO: compute normalized coordinate of every pixel whose disparity is positive
# hint: the normalized coordinate of pixel [u, v, 1] is K^(-1) @ [u, v, 1]
K_inv = np.linalg.inv(K)
flat_indices = np.flatnonzero(valid)
v_coords, u_coords = np.divmod(flat_indices, disparity.shape[1])
homogeneous_pixels = np.stack((u_coords, v_coords, np.ones_like(u_coords)), axis=1).astype(np.float32)
normalized_coord = homogeneous_pixels @ K_inv.T


# TODO: compute 3D coordinate of every pixel whose disparity is positive
# hint: 3D coordinate of pixel (u, v) is the product of Z and its normalized cooridnate
depth_valid = depth.ravel()[flat_indices]
all_3d = normalized_coord * depth_valid[:, None]  # this is matrix storing 3D coordinate of every pixel whose disparity is positive
# the shape of all_3d is <num_pixels_positive_disparity, 3>
# each row of all_3d is a 3D coordinate [X, Y, Z]
# you need to change the value of all_3d with your computation of 3D coordinate of every pixel whose disparity is positive

# TODO: get color for 3D points
all_color = left_color.reshape(-1, 3)[flat_indices][:, ::-1]  # this is matrix storing color of every pixel whose disparity is positive
# the shape of all_color is <num_pixels_positive_disparity, 3>
# each row of all_color is [R, G, B] value

# TODO: THE ORDER OF all_color IS THE SAME WITH all_3d
# the shape of all_color is <num_pixels_positive_disparity, 3>
# each row of all_color is [R, G, B] value
all_color = all_color.astype(np.float64) / 255.0

# Display pointcloud
cloud = o3d.geometry.PointCloud()  # create pointcloud object
cloud.points = o3d.utility.Vector3dVector(all_3d)
cloud.colors = o3d.utility.Vector3dVector(all_color)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])  # create frame object
try:
    o3d.visualization.draw_geometries(
        [cloud, mesh_frame], window_name="PointCloud", width=1024, height=768
    )
except Exception as e:
    print("[Open3D] Visualization failed:", e)
    out_path = os.path.join(os.path.dirname(__file__), "cloud_output.ply")
    try:
        o3d.io.write_point_cloud(out_path, cloud)
        print("Saved point cloud to", out_path)
    except Exception as e2:
        print("Failed to save point cloud:", e2)
