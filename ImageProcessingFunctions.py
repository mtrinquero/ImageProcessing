# Mark Trinquero
# Image Processing Helper Functions

def image_to_matrix(image_file, grays=False):
    img = image.imread(image_file)
    if(len(img.shape) == 3 and img.shape[2] > 3):
        height, width, depth = img.shape
        new_img = np.zeros([height, width, 3])
        for r in range(height):
            for c in range(width):
                new_img[r,c,:] = img[r,c,0:3]
        img = np.copy(new_img)
    if(grays and len(img.shape) == 3):
        height, width = img.shape[0:2]
        new_img = np.zeros([height, width])
        for r in range(height):
            for c in range(width):
                new_img[r,c] = img[r,c,0]
        img = new_img
    if(len(img.shape) == 2):
        zeros = np.where(img == 0)[0]
        img[zeros] += 1e-7
    return img

def matrix_to_image(image_matrix, image_file):
    cMap = None
    if(len(image_matrix.shape) < 3):
        cMap = cm.Greys_r
    image.imsave(image_file, image_matrix, cmap=cMap)
    
def flatten_image_matrix(image_matrix):
    if(len(image_matrix.shape) == 3):
        height, width, depth = image_matrix.shape
    else:
        height, width = image_matrix.shape
        depth = 1
    flattened_values = np.zeros([height*width,depth])
    for i, r in enumerate(image_matrix):
        for j, c in enumerate(r):
            flattened_values[i*width+j,:] = c
    return flattened_values

def unflatten_image_matrix(image_matrix, width):
    heightWidth = image_matrix.shape[0]
    height = int(heightWidth / width)
    if(len(image_matrix.shape) > 1):
        depth = image_matrix.shape[-1]
        unflattened_values = np.zeros([height, width, depth])
        for i in range(height):
            for j in range(width):
                unflattened_values[i,j,:] = image_matrix[i*width+j,:]
    else:
        depth = 1
        unflattened_values = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                unflattened_values[i,j] = image_matrix[i*width+j]
    return unflattened_values

def image_difference(image_values_1, image_values_2):
    flat_vals_1 = flatten_image_matrix(image_values_1)
    flat_vals_2 = flatten_image_matrix(image_values_2)
    N, depth = flat_vals_1.shape
    dist = 0.
    point_thresh = 0.005
    for i in range(N):
        if(depth > 1):
            new_dist = sum(abs(flat_vals_1[i] - flat_vals_2[i]))
            if(new_dist > depth * point_thresh):
                dist += new_dist
        else:
            new_dist = abs(flat_vals_1[i] - flat_vals_2[i])
            if(new_dist > point_thresh):
                dist += new_dist
    return dist


