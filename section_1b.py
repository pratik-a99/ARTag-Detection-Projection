import cv2 as cv
import numpy as np

orig_img = cv.imread('test.png')
img = cv.imread('test.png', 0)

ref_size_x = 120
ref_size_y = 120


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return np.int0(rect)


def homography_mat(x1, x2, x3, x4, y1, y2, y3, y4, xp1, xp2, xp3, xp4, yp1, yp2, yp3, yp4):
    # Initializing the matrix
    A = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
                   [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
                   [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
                   [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
                   [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
                   [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
                   [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
                   [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]])

    # Calculating the AA^t and A^tA matrices
    W = A * A.T
    X = A.T * A

    # Finding the eigen values and eigen vectors of the calculated matrices
    eig_u, U = np.linalg.eig(W)
    eig_v, V = np.linalg.eig(X)


    X_sort = np.argsort(eig_v)[::-1]
    eig_v = eig_v[X_sort]
    V = V[:, X_sort]

    # Displaying the homography matrix
    return np.asmatrix(V[:, eig_v.argmin()]).reshape(3, 3)

######## REF_IMG ###########

rxp1 = 0
rxp2 = ref_size_x
rxp3 = ref_size_x
rxp4 = 0

ryp1 = 0
ryp2 = 0
ryp3 = ref_size_y
ryp4 = ref_size_y

ref_corners_ordered = np.array([[rxp1, ryp1], [rxp2, ryp2], [rxp3, ryp3], [rxp4, ryp4]])

#############################


# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if ret:
th, img_bin = cv.threshold(img, 180, 255, cv.THRESH_BINARY)
corners = cv.goodFeaturesToTrack(img_bin, 9, 0.1, 100)
corners = np.int0(corners)

remove = np.array([np.argmin(corners[:, :, 0]), np.argmax(corners[:, :, 0]),
                   np.argmin(corners[:, :, 1]), np.argmax(corners[:, :, 1])])

corners = np.delete(corners, remove, 0)

tag_corners_itr = np.array([np.argmin(corners[:, :, 0]), np.argmin(corners[:, :, 1]),
                            np.argmax(corners[:, :, 0]), np.argmax(corners[:, :, 1])])

tag_corners = corners[tag_corners_itr]
tag_corners_ordered = order_points(tag_corners.reshape(4, 2))

warped_image = np.zeros([ref_size_x, ref_size_y, 3], dtype=np.uint8)


def warp(src_corners, dst_corners, img_src, img_dst):
    h_matrix = homography_mat(src_corners[0][0], src_corners[1][0], src_corners[2][0], src_corners[3][0],
                              src_corners[0][1], src_corners[1][1], src_corners[2][1], src_corners[3][1],
                              dst_corners[0][0], dst_corners[1][0], dst_corners[2][0], dst_corners[3][0],
                              dst_corners[0][1], dst_corners[1][1], dst_corners[2][1], dst_corners[3][1])
    h_matrix = h_matrix / h_matrix[-1, -1]

    min_src_cor = min(src_corners.flatten())
    max_src_cor = max(src_corners.flatten())

    min_dst_cor_x = min(dst_corners[:, 0])
    max_dst_cor_x = max(dst_corners[:, 0])

    min_dst_cor_y = min(dst_corners[:, 1])
    max_dst_cor_y = max(dst_corners[:, 1])

    for i_x in range(min_src_cor, max_src_cor + 1):
        for i_y in range(min_src_cor, max_src_cor + 1):
            orig_pos = np.array([i_x, i_y, 1]).reshape(3, 1)
            new_pos = np.matmul(h_matrix, orig_pos)

            new_x = int(new_pos[0] / new_pos[2])
            new_y = int(new_pos[1] / new_pos[2])

            if (min_dst_cor_x <= new_x < max_dst_cor_x) and (min_dst_cor_y <= new_y < max_dst_cor_y):
                img_dst[new_y][new_x] = img_src[i_y][i_x]

    # kernel = np.ones((3, 3), np.uint8)
    # img_dst = cv.dilate(img_dst, kernel)
    # img_dst = cv.erode(img_dst, kernel)

    return img_dst


warped = warp(tag_corners_ordered, ref_corners_ordered, img_bin,
              warped_image)

################# Decoding #####################

gap_x = int(ref_size_x / 8)
gap_y = int(ref_size_y / 8)


def grid_part(i_part, j_part, bin_img):
    return bin_img[i_part * gap_x: (i_part + 1) * gap_x,
           j_part * gap_y: (j_part + 1) * gap_y]


def rotation(bin_img):
    rot_corners = [cv.countNonZero(grid_part(5, 5, bin_img)), cv.countNonZero(grid_part(5, 2, bin_img)),
                   cv.countNonZero(grid_part(2, 2, bin_img)), cv.countNonZero(grid_part(2, 5, bin_img))]

    return np.argmax(rot_corners)


def decode(bin_img):
    decoded_grid = [int(cv.countNonZero(grid_part(3, 3, bin_img)) > 100),
                    int(cv.countNonZero(grid_part(3, 4, bin_img)) > 100),
                    int(cv.countNonZero(grid_part(4, 4, bin_img)) > 100),
                    int(cv.countNonZero(grid_part(4, 3, bin_img)) > 100)]

    rot_num = rotation(bin_img)

    decoded_grid = (decoded_grid[rot_num:] + decoded_grid[:rot_num])
    decoded_grid.reverse()

    number = ''.join(str(bit) for bit in decoded_grid)

    return rot_num, number


warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
_, warped_binary = cv.threshold(warped, 127, 255, cv.THRESH_BINARY)

print("(Rotation/90, Decoded Binary)")
print(decode(warped_binary))

cv.imshow('Corners', warped_binary)

cv.imwrite("flattened_tag.png", warped_binary)

#     if cv.waitKey(250) & 0xFF == ord('q'):
#         break
# else:
#     break


cv.waitKey()
#########################################################################################################################