import cv2
import numpy as np
import random
try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii
import matplotlib.pyplot as plt


def generate_random_data(height, width, count):
    x, y, defects_gt, gt, trans = zip(*[generate_img_and_rot_img(height, width) for i in range(0, count)])

    X = np.asarray(x) * 255
    X = X.repeat(1, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    # cv2.imshow("haha", X[0, :])
    # cv2.waitKey(3000)
    Y = np.asarray(y) * 255
    Y = Y.repeat(1, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    # cv2.imshow("haha", Y[0, :])
    # cv2.waitKey(3000)    
    defects_GT = np.asarray(defects_gt) * 255
    defects_GT = defects_GT.repeat(1, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    # cv2.imshow("haha", defects_GT[0, :])
    # cv2.waitKey(3000)    

    return X, Y, defects_GT, gt, trans

def generate_img_and_rot_img(height, width):
    shape = (height, width)

    triangle_location = get_random_location(*shape)
    triangle_location1 = get_random_location(*shape)
    triangle_location2 = get_random_location(*shape)
    triangle_location3 = get_random_location(*shape, zoom=0.5)
    circle_location1 = get_random_location(*shape, zoom=0.7)
    circle_location2 = get_random_location(*shape, zoom=0.5)
    circle_location3 = get_random_location(*shape, zoom=0.9)
    circle_location4 = get_random_location(*shape, zoom=0.8)
    mesh_location = get_random_location(*shape)
    square_location = get_random_location(*shape, zoom=0.8)
    square_location2 = get_random_location(*shape, zoom=0.5)
    plus_location = get_random_location(*shape, zoom=1.2)
    plus_location1 = get_random_location(*shape, zoom=1.2)
    plus_location2 = get_random_location(*shape, zoom=1.2)
    plus_location3 = get_random_location(*shape, zoom=1.2)
    plus_location4 = get_random_location(*shape, zoom=0.6)

    


    # Create input image
    # arr = np.random.rand(height, width)
    arr = np.zeros((height, width),dtype=bool)
    
    arr = add_triangle(arr, *triangle_location)
    arr = add_triangle(arr, *triangle_location1)
    arr = add_triangle(arr, *triangle_location2)
    arr = add_circle(arr, *circle_location1)
    arr = add_circle(arr, *circle_location2, fill=True)
    arr = add_circle(arr, *circle_location3)
    arr = add_mesh_square(arr, *mesh_location)
    arr = add_filled_square(arr, *square_location)
    arr = add_plus(arr, *plus_location)
    arr = add_plus(arr, *plus_location1)
    arr = add_plus(arr, *plus_location2)
    arr = add_plus(arr, *plus_location3)

    arr = np.reshape(arr, (1, height, width)).astype(np.float32)
    

    # create an arr1 to generate a couple of same arr with vibrations.
    arr1 = np.zeros((height, width))
    
    arr1 = add_triangle(arr1, *tuple(np.asarray(triangle_location)+np.asarray([np.random.rand()*2,np.random.rand()*2,0]).astype(int)))
    arr1 = add_triangle(arr1, *triangle_location1)
    arr1 = add_triangle(arr1, *triangle_location2)
    arr1 = add_circle(arr1, *tuple(np.asarray(circle_location1)+np.asarray([np.random.rand()*2,np.random.rand()*2,0]).astype(int)))
    arr1 = add_circle(arr1, *circle_location2, fill=True)
    arr1 = add_circle(arr1, *circle_location3)
    arr1 = add_mesh_square(arr1, *tuple(np.asarray(mesh_location)+np.asarray([np.random.rand()*2,np.random.rand()*2,0]).astype(int)))
    arr1 = add_filled_square(arr1, *square_location)
    arr1 = add_plus(arr1, *plus_location)
    arr1 = add_plus(arr1, *plus_location1)
    arr1 = add_plus(arr1, *plus_location2)
    arr1 = add_plus(arr1, *plus_location3)

    arr1 = np.reshape(arr1, (1, height, width)).astype(np.float32)


    # create a rotated arrays that rotates arrays

    angle = np.random.rand() * 10
    t_x = (np.random.randn()*0.0)
    t_y = (np.random.randn()*0.0)
    trans = np.array((t_y, t_x))

    if angle < -180.0:
        angle = angle + 360.0
    elif angle > 180.0:
        angle = angle - 360.0

    (_, h, w) = arr.shape
    (cX, cY) = (w//2, h//2)
    rot = arr[0,]

    N = np.float32([[1,0,t_x],[0,1,t_y]])
    rot = cv2.warpAffine(rot, N, (w, h))

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rot = cv2.warpAffine(rot, M, (w, h))
    rot = cv2.resize(rot, (h, w), interpolation=cv2.INTER_CUBIC)


    rot = rot[np.newaxis, :]
# for dynamic obstacles, comment out if you dont want any dynamic obstacles
    arr[0,] = add_triangle(arr[0,], *triangle_location3)
    arr[0,] = add_plus(arr[0,], *plus_location4)
    arr[0,] = add_filled_square(arr[0,], *square_location2)
    arr[0,] = add_circle(arr[0,], *circle_location4)
    arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    defects_gt = np.zeros(shape, dtype=bool)
    defects_gt = add_triangle(defects_gt, *triangle_location3)
    defects_gt = add_plus(defects_gt, *plus_location4)   
    defects_gt = add_filled_square(defects_gt, *square_location2)
    defects_gt = add_circle(defects_gt, *circle_location4)
    defects_gt = np.reshape(defects_gt, (1, height, width)).astype(np.float32)

    return arr, rot, defects_gt, angle, trans

def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x-s,y-s:y+s] = True
    arr[x+s,y-s:y+s] = True
    arr[x-s:x+s,y-s] = True
    arr[x-s:x+s,y+s] = True

    return arr

def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))

def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array

def add_mesh_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))

def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

    return arr

def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

    return new_arr

def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x-1:x+1,y-s:y+s] = True
    arr[x-s:x+s,y-1:y+1] = True

    return arr

def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.22, 0.78))
    y = int(height * random.uniform(0.22, 0.78))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return (x, y, size)