import cv2 as cv
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# orig_img = cv.imread('test.png')
# img = cv.imread('test.png', 0)

cap = cv.VideoCapture('1tagvideo.mp4')
corner_tag_new = []

ref_size_x = 160
ref_size_y = 160

K_Matrix = np.array([[1346.100595, 0, 932.1633975],
                     [0, 1355.933136, 654.8986796],
                     [0, 0, 1]])

result = cv.VideoWriter('result.mp4',
                        cv.VideoWriter_fourcc(*'mp4v'),
                        10, (1920, 1080))

def unique_generator(lst):
    seen = set()
    for item in lst:
        tupled = tuple(item)
        if tupled not in seen:
            seen.add(tupled)
            yield item


def get_corners(img):
    global corner_tag_new

    corners_ = cv.goodFeaturesToTrack(img, 15, 0.05, 50)
    corners_ = np.int0(corners_)

    x_min = np.argmin(corners_[:, :, 0])
    x_max = np.argmax(corners_[:, :, 0])
    y_min = np.argmin(corners_[:, :, 1])
    y_max = np.argmax(corners_[:, :, 1])

    corner_tag = []

    polygon = Polygon([(corners_[x_min][0][0], corners_[x_min][0][1]), (corners_[y_min][0][0], corners_[y_min][0][1]),
                       (corners_[x_max][0][0], corners_[x_max][0][1]), (corners_[y_max][0][0], corners_[y_max][0][1])])

    for corner in corners_:
        x_pnt, y_pnt = corner.ravel()
        point = Point(x_pnt, y_pnt)
        if polygon.contains(point):
            corner_tag.append([x_pnt, y_pnt])

    if not corner_tag:
        return corner_tag_new

    corner_tag = np.asarray(corner_tag)
    x_min = np.argmin(corner_tag[:, 0])
    x_max = np.argmax(corner_tag[:, 0])
    y_min = np.argmin(corner_tag[:, 1])
    y_max = np.argmax(corner_tag[:, 1])

    corner_tag_ = np.array([corner_tag[x_min], corner_tag[y_min], corner_tag[x_max], corner_tag[y_max]])

    lst = list(unique_generator(corner_tag_))
    if len(lst) < 4:
        return corner_tag_new

    corner_tag_new = corner_tag_

    corner_tag_new = order_points(corner_tag_new)

    return corner_tag_new

def find_fft(img_ip):
    img_gray = cv.cvtColor(img_ip, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img_gray, (7, 7), cv.BORDER_DEFAULT)

    morphed = cv.morphologyEx(blur, cv.MORPH_OPEN, np.ones((7, 7)))
    r, image_fft = cv.threshold(morphed, 170, 255, cv.THRESH_BINARY)

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(image_fft)

    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols, _ = img_ip.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols, 2), np.uint8)
    r = 100

    center = [crow, ccol]

    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    # apply mask and inverse DFT
    fshift = dft_shift * mask

    fshift_mask_mag = 2000 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    # ret, img_back = cv.threshold(img_back,170,255,cv.THRESH_BINARY)
    img_back = cv.GaussianBlur(img_back, (7, 7), cv.BORDER_DEFAULT)

    return img_back


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

    # Sorting the eigen values and vectors
    # W_sort = np.argsort(eig_u)[::-1]
    # eig_u = eig_u[W_sort]
    # U = U[:, W_sort]

    X_sort = np.argsort(eig_v)[::-1]
    eig_v = eig_v[X_sort]
    V = V[:, X_sort]

    # Calculating the S matrix
    # S = np.diag(np.sqrt(eig_u))
    # S = np.hstack((S, np.zeros((8, 1))))

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
                if (0 <= i_x < 1920) and (0 <= i_y < 1080):
                    img_dst[new_y][new_x] = img_src[i_y][i_x]

    return img_dst


def warp_inv(src_corners, dst_corners, img_src, img_dst, rot):
    dst_corners = np.concatenate((dst_corners[rot:], dst_corners[:rot]), axis=0)

    img_dst_cp = img_dst.copy()
    h_matrix = homography_mat(src_corners[0][0], src_corners[1][0], src_corners[2][0], src_corners[3][0],
                              src_corners[0][1], src_corners[1][1], src_corners[2][1], src_corners[3][1],
                              dst_corners[0][0], dst_corners[1][0], dst_corners[2][0], dst_corners[3][0],
                              dst_corners[0][1], dst_corners[1][1], dst_corners[2][1], dst_corners[3][1])
    h_matrix = h_matrix / h_matrix[-1, -1]

    # min_src_cor = min(src_corners.flatten())
    # max_src_cor = max(src_corners.flatten())

    min_src_cor_x = min(src_corners[:, 0])
    max_src_cor_x = max(src_corners[:, 0])

    min_src_cor_y = min(src_corners[:, 1])
    max_src_cor_y = max(src_corners[:, 1])

    for i_x in range(min_src_cor_x, max_src_cor_x):
        for i_y in range(min_src_cor_y, max_src_cor_y):
            orig_pos = np.array([i_x, i_y, 1]).reshape(3, 1)
            new_pos = np.matmul(h_matrix, orig_pos)

            new_x = int(new_pos[0] / new_pos[2])
            new_y = int(new_pos[1] / new_pos[2])

            # if (min_dst_cor_x <= new_x < max_dst_cor_x) and (min_dst_cor_y <= new_y < max_dst_cor_y):
            img_dst_cp[new_y][new_x] = img_src[i_y][i_x]

    # kernel = np.ones((3, 3), np.uint8)
    # img_dst = cv.dilate(img_dst, kernel)
    # img_dst = cv.erode(img_dst, kernel)

    return img_dst_cp


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

    rot_num_ = rotation(bin_img)

    decoded_grid = (decoded_grid[rot_num_:] + decoded_grid[:rot_num_])
    decoded_grid.reverse()

    number = ''.join(str(bit) for bit in decoded_grid)

    return rot_num_, number


##########################################################################

################## TESTUDO ###################

testudo = cv.imread('testudo.png')

testudo_x = testudo.shape[1]
testudo_y = testudo.shape[0]

testudo_corners_ordered = np.array([[0, 0], [testudo_x, 0], [testudo_x, testudo_y], [0, testudo_y]])


##############################################


#################### Projection Matrix #########################

def cube_projection(src_corners, dst_corners, cube_dimension, img_dst, rot, k_mat):
    dst_corners = np.concatenate((dst_corners[rot:], dst_corners[:rot]), axis=0)

    img_dst_cp = img_dst.copy()
    h_matrix = homography_mat(src_corners[0][0], src_corners[1][0], src_corners[2][0], src_corners[3][0],
                              src_corners[0][1], src_corners[1][1], src_corners[2][1], src_corners[3][1],
                              dst_corners[0][0], dst_corners[1][0], dst_corners[2][0], dst_corners[3][0],
                              dst_corners[0][1], dst_corners[1][1], dst_corners[2][1], dst_corners[3][1])
    h_matrix = h_matrix / h_matrix[-1, -1]

    K_inv = np.linalg.inv(k_mat)

    scale_fac = 2 / (np.linalg.norm(K_inv.dot(h_matrix[:, 0])) + np.linalg.norm(K_inv.dot(h_matrix[:, 1])))

    B_tilde = np.array(scale_fac * K_inv.dot(h_matrix))

    B = B_tilde if (np.linalg.det(B_tilde) > 0) else (- B_tilde)

    ## Rotation and translation Vectors

    r1 = B[:, 0]
    r2 = B[:, 1]

    r3 = np.cross(r1, r2)

    t = B[:, 2]

    rot_mat = np.vstack((r1, r2, r3, t)).T

    p_matrix = k_mat.dot(rot_mat)

    ## Cube

    cube_pts = [[0, 0, 0, 1],
                [0, cube_dimension, 0, 1],
                [cube_dimension, cube_dimension, 0, 1],
                [cube_dimension, 0, 0, 1],
                [0, 0, -cube_dimension, 1],
                [0, cube_dimension, -cube_dimension, 1],
                [cube_dimension, cube_dimension, -cube_dimension, 1],
                [cube_dimension, 0, -cube_dimension, 1]]

    new_cube = np.array([p_matrix.dot(cube_itr) for cube_itr in cube_pts])

    dst_cube = new_cube[:, :-1] / new_cube[:, 2:]

    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[0].astype(int)), tuple(dst_cube[1].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[1].astype(int)), tuple(dst_cube[2].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[2].astype(int)), tuple(dst_cube[3].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[3].astype(int)), tuple(dst_cube[0].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[4].astype(int)), tuple(dst_cube[5].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[5].astype(int)), tuple(dst_cube[6].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[6].astype(int)), tuple(dst_cube[7].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[7].astype(int)), tuple(dst_cube[4].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[0].astype(int)), tuple(dst_cube[4].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[1].astype(int)), tuple(dst_cube[5].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[2].astype(int)), tuple(dst_cube[6].astype(int)), (0, 0, 255), 2)
    img_dst_cp = cv.line(img_dst_cp, tuple(dst_cube[3].astype(int)), tuple(dst_cube[7].astype(int)), (0, 0, 255), 2)

    for pts in dst_cube:
        cv.circle(img_dst_cp, (int(pts[0]), int(pts[1])), 3, (180, 0, 0), -1)

    return img_dst_cp


################################################################
count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        count += 1
        print(count)
        img_edges = find_fft(frame)

        corners = get_corners(img_edges)
        corners = np.int0(corners)

        warped = warp(corners, ref_corners_ordered, frame,
                      warped_image)

        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        _, warped_binary = cv.threshold(warped, 127, 255, cv.THRESH_BINARY)

        rot_num, decoded = decode(warped_binary)

        testudo_warp = warp_inv(testudo_corners_ordered, corners, testudo, frame, rot_num)

        # projected = cube_projection(ref_corners_ordered, corners, ref_size_x, frame, rot_num, K_Matrix)

        # cv.imshow('orig', orig_img)
        # cv.imshow('warp', warped)
        cv.imshow('test', testudo_warp)
        # result.write(testudo_warp)
        # cv.imshow('projected', projected)
        # result.write(projected)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

result.release()
cv.waitKey()
