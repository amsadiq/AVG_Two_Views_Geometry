import numpy as np
import cv2


# part 1


def normalize_transformation(points: np.ndarray) -> np.ndarray:
    """
    Compute a similarity transformation matrix that translate the points such that
    their center is at the origin & the avg distance from the origin is sqrt(2)
    :param points: <float: num_points, 2> set of key points on an image
    :return: (sim_trans <float, 3, 3>)
    """
    center = points[:, :2].mean(axis=0)  # TODO: find center of the set of points by computing mean of x & y
    dist = np.linalg.norm(points[:, :2] - center, axis=1, keepdims=True) # TODO: matrix of distance from every point to the origin, shape: <num_points, 1>
    s = np.sqrt(2) / dist.mean() # TODO: scale factor the similarity transformation = sqrt(2) / (mean of dist)
    sim_trans = np.array([
        [s,     0,      -s * center[0]],
        [0,     s,      -s * center[1]],
        [0,     0,      1]
    ])
    return sim_trans


def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate
    :param points: <float: num_points, num_dim>
    :return: <float: num_points, 3>
    """
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


# read image & put them in grayscale
img1 = cv2.imread('/Users/sadiq/Desktop/M2/Lecture notes/AVG/AVG_Two_Views_Geometry/scripts_and_file/img/chapel00.png', 0)  # queryImage
img2 = cv2.imread('/Users/sadiq/Desktop/M2/Lecture notes/AVG/AVG_Two_Views_Geometry/scripts_and_file/img/chapel01.png', 0)  # trainImage

# detect kpts & compute descriptor
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# match kpts
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# organize key points into matrix, each row is a point
query_kpts = np.array([kp1[m.queryIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>
train_kpts = np.array([kp2[m.trainIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>

# normalize kpts
T_query = normalize_transformation(query_kpts)  # get the similarity transformation for normalizing query kpts
normalized_query_kpts = (T_query @ homogenize(query_kpts).T).T # TODO: apply T_query to query_kpts to normalize them

T_train = normalize_transformation(train_kpts)  # get the similarity transformation for normalizing train kpts
normalized_train_kpts = (T_train @ homogenize(train_kpts).T).T # TODO: apply T_train to train_kpts to normalize them

# construct homogeneous linear equation to find fundamental matrix
x, y = normalized_query_kpts[:, 0], normalized_query_kpts[:, 1]
xp, yp = normalized_train_kpts[:, 0], normalized_train_kpts[:, 1]
A = np.column_stack((xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, np.ones_like(x))) # TODO: construct A according to Eq.(3) in lab subject

# TODO: find vector f by solving A f = 0 using SVD
# hint: perform SVD of A using np.linalg.svd to get u, s, vh (vh is the transpose of v)
# hint: f is the last column of v
_, _, vh = np.linalg.svd(A, full_matrices=False)
f = vh[-1] # TODO: find f

# arrange f into 3x3 matrix to get fundamental matrix F
F = f.reshape(3, 3)

# TODO: force F to have rank 2
# hint: perform SVD of F using np.linalg.svd to get u, s, vh
# hint: set the smallest singular value of F to 0
# hint: reconstruct F from u, new_s, vh
u, s, vh = np.linalg.svd(F, full_matrices=False)
s[-1] = 0
F = u @ np.diag(s) @ vh
assert np.linalg.matrix_rank(F) == 2, 'Fundamental matrix must have rank 2'
print('rank F: ', np.linalg.matrix_rank(F))  # should be = 2

# TODO: de-normlaize F
# hint: last line of Algorithme 1 in the lab subject
F = T_train.T @ F @ T_query
F_gt = np.loadtxt('chapel.00.01.F')
print(F - F_gt)
