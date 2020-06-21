#!/usr/share/anaconda3/bin/python3

from PIL import Image, ImageDraw
import sys
import numpy as np
from time import time
from math import pi, atan
from os import listdir
from os.path import isfile, join
from threading import Thread, Lock

margin = 20 # margin size to enlarge picture
crop_margin = 10 # margin to crop final image
scale = 4 # scale down value to work on reduced size
background_threshold = 5
definite_foreground = 100
background = np.array([255, 255, 255])
quality = 80 # JPEG saving quality
theta_angle = 5 # threshold for linear regression in angle search
theta_cropping = 10 # threshold for linear regression in bounding box search
min_picture_length = 3 # in cm, minimal picture length
dpi = 300 # dots per inch
save_image = True
threadLock = Lock() # semaphore, necessary for console output and saving


def show_small(image):
    image.resize((image.size[0] // scale, image.size[1] // scale)).show()


def get_suppressed_matrix(image):
    list_red = [abs(i - background[0]) for i in range(256)]
    list_green = [abs(i - background[1]) for i in range(256)]
    list_blue = [abs(i - background[2]) for i in range(256)]
    R, G, B = image.split()
    R_m = np.asarray(R.point(list_red), dtype="int32")
    G_m = np.asarray(G.point(list_green), dtype="int32")
    B_m = np.asarray(B.point(list_blue), dtype="int32")
    return R_m + G_m + B_m
    

def add_margin(image, margin, center = True):
    width, height = image.size
    new_size = (width + 2 * margin, height + 2 * margin)
    image_margin = Image.new("RGB", new_size)
    ImageDraw.Draw(image_margin).rectangle((0, 0) + new_size, fill = tuple(background))
    image_margin.paste(image, (margin, margin) if center else (0, 0))
    return image_margin


def rotated_crop_boundary(M):
    # 10% of the picture must be at least of non background color
    min_pixels = (min_picture_length / 2.54 * dpi)**2 * 0.1 
    M = np.rot90(M, k = 2)
    min_h, min_w = height, width = M.shape
    
    while True:
        horizontals = np.maximum.accumulate(M, axis = 1)
        verticals = np.maximum.accumulate(M, axis = 0)
        D = np.maximum.accumulate(verticals, axis = 1)
        potentials = np.nonzero(np.multiply(np.multiply(horizontals <= background_threshold, verticals <= background_threshold), D >= definite_foreground))
        
        if len(potentials[0]) > 0:
            h, w = potentials[0][0],  potentials[1][0]
            if np.sum(M[:h, :w] >= definite_foreground) > min_pixels:
                min_h, min_w = height - h, width - w
                break
            else:
                M[:h, :w] = 0
        else:
            break
            
    return [min_w, min_h, width, height]


# method using linear regression for finding a bounding box
def crop_boundaries(image, margin = 0, theta = 1):
    image = add_margin(image, margin, center = False)
    width, height = image.size
    boundaries, M = [0, 0, 0, 0], get_suppressed_matrix(image)
    
    # crop for all four sides of a picture
    for pos in range(4):
        last_line = M[height - 1 : height,  : ].copy()
        M[height - 1 : height,  : ] = 255
        
        Y = np.argmin(M < background_threshold, axis = 0)
        M[height - 1 : height,  : ] = last_line
        beta = np.matrix([[np.median(Y)]]) # initial parameter is median position of all points
        
        # linear regression
        X = np.arange(width)
        J_matrix = np.matrix(np.ones((1, width))) / width
        
        def E(YY, theta): 
            YY[YY > theta] = theta
            return YY
        
        for rng in range(10): # iterative optimization of parameter towards local optimum
            delta_Y = E(Y - beta[0, 0], theta)
            beta += J_matrix * delta_Y.reshape(width, 1)
        boundaries[pos] = int(beta[0, 0])
        if pos < 3: M = np.rot90(M)
        width, height = height, width
        
    min_h = max(min(boundaries[0] + margin, height), 0)
    max_w = max(min(width - (boundaries[1] + margin), width), 0)
    max_h = max(min(height - (boundaries[2] + margin), height), 0)
    min_w = max(min(boundaries[3] + margin, width), 0)
                
    return [min_w, min_h, max_w, max_h]


# method using linear regression for edge detection to find a proper angle for correction
def find_angle(M, theta = 10):
    height, width = M.shape
    line = M[height - 1 : height, : ].copy()
    M[height - 1 : height, : ] = 255
    Y = np.argmin(M < background_threshold, axis = 0)
    M[height - 1 : height, : ] = line
    beta = np.matrix([[0], [height * 0.1]])
    
    # linear regression
    X = np.arange(width)
    J_trans = np.matrix([X, np.ones(width)])
    J = np.transpose(J_trans)
    J_matrix = (np.linalg.inv(J_trans * J) * J_trans)
    
    def f(X, beta): return X * beta[0, 0] + beta[1, 0] # first order polynomial, i.e. straight line
    def E(YY, theta):
        YY[YY > theta] = theta
        return YY
    
    for rng in range(10): # iterative optimization of parameters towards local optimum
        delta_Y = E(Y - f(X, beta), theta)
        beta += J_matrix * delta_Y.reshape(width, 1)
        
    y1, y2 = beta[1, 0], beta[0, 0] * width + beta[1, 0]
    angle = atan(abs(y1 - y2) / width) / pi * 180 * (1 if y1 < y2 else -1)
    least_squares = np.sum(np.square(E(Y - f(X, beta), theta))) / width
    return [angle, least_squares]


# once a picture was found, a proper angle is determined, the picture is rotated and cropped again
class crop_and_rotate(Thread):
    def __init__(self, image, folder_name = "."):
        Thread.__init__(self)
        self.image = image
        self.folder_name = folder_name
    
    def run(self):
        cropped_image = add_margin(self.image, margin)
        width, height = cropped_image.size
        num_pixels = width * height
        output = "found picture"
        
        # find the correct angle using all 4 edges of a picture and choosing via smallest least squares value
        small_cropped_image = cropped_image.resize((width // scale, height // scale))
        M = get_suppressed_matrix(small_cropped_image)
        angle, least_squares = find_angle(M, theta_angle)
        for rng in range(3):
            M = np.rot90(M)
            ang, lsq = find_angle(M, theta_angle)
            if least_squares > lsq: angle, least_squares = ang, lsq
        output += "\nangle: %0.3f" % angle
        rotated_image = cropped_image.rotate(angle, resample=Image.BILINEAR)
        
        # search and crop boundaries after rotation
        data = np.array(rotated_image)
        mask = (data[:,:,0] == 0) & (data[:,:,1] == 0) & (data[:,:,2] == 0)
        data[:,:,:3][mask] = background
        colored_image = Image.fromarray(data)
        colored_width, colored_heigt = colored_image.size
        
        small_colored_image = colored_image.resize((colored_width // scale, colored_heigt // scale))
        cb = crop_boundaries(small_colored_image, crop_margin, theta_cropping)
        for i in range(len(cb)): cb[i] *= scale
        final_image = rotated_image.crop(cb)
        final_width, final_height = final_image.size
        output += "\nimage size: %i x %i" % (final_width, final_height)
        
        # find next free image number
        threadLock.acquire()
        onlyfiles = [f for f in listdir(self.folder_name) if isfile(join(self.folder_name, f))]
        img_files = [f.lower() for f in onlyfiles if f[:3].lower() == "img"]
        
        img_number = 1
        while True:
            file_name = str(img_number)
            while len(file_name) < 4: file_name = "0" + file_name
            file_name = "img" + file_name + ".jpg"
            if file_name in img_files: img_number += 1
            else: break
        if self.folder_name[-1] != "/": self.folder_name += "/"
        file_name = self.folder_name + file_name.replace("img", "IMG")
        
        if save_image:
            output += "\nsaving in file %s" % file_name
            final_image.save(file_name, "JPEG", quality = quality)
        else:
            show_small(final_image)
        print(output + "\n")
        threadLock.release()


def print_help_text():
    print("FSC - Fast Scan Cropper")
    print("A tool for extracting pictures from scans images with white background.\n")
    print("usage:", sys.argv[0], "[options] infile-image")
    print()
    print("options:")
    print("\t-p, --preview\t\tshow thumbnail(s) of cropped picture(s), do not store")
    print("\t-h, --help\t\tshow this help text")
    exit()
    
    
# parse commands from command line
if len(sys.argv) < 2:
    print_help_text()
args = 1
if sys.argv[args] in ["-p", "--preview"]:
    save_image = False
    args += 1
elif sys.argv[args] in ["-h", "--help"]:
    print_help_text()
elif sys.argv[args][0] == "-":
    print_help_text()
if len(sys.argv) <= args:
    print_help_text()
    
# read in file and preparing folder name for storing
image_raw = Image.open(sys.argv[args])
dpi = image_raw.info['dpi'][0] if image_raw.info['dpi'][0] > 0 else dpi
folder_name = sys.argv[args]
f_i = len(folder_name) - 1
while f_i > 0 and folder_name[f_i - 1] != "/": f_i -= 1
folder_name = folder_name[ : f_i]
if folder_name == "": folder_name = "."

# adding margin, to ensure well separation
# shrinking for faster processing to cropping original image again
image_raw = add_margin(image_raw, scale * margin, center = False)
width_raw, height_raw = image_raw.size
image_margin = image_raw.resize((width_raw // scale, height_raw // scale))
image_draw = ImageDraw.Draw(image_margin)
image_draw_orig = ImageDraw.Draw(image_raw)

# searching for rectangles of well separated pictures
image_processing_threads = []
while True:
    M = get_suppressed_matrix(image_margin)
    horizontals = np.maximum.accumulate(M, axis = 1)
    verticals = np.maximum.accumulate(M, axis = 0)
    D = np.maximum.accumulate(verticals, axis = 1)
    potentials = np.nonzero(np.multiply(np.multiply(horizontals <= background_threshold, verticals <= background_threshold), D >= definite_foreground))

    if len(potentials[0]) == 0: break
    
    h, w = potentials[0][0], potentials[1][0]
    bbox_crop = rotated_crop_boundary(M[:h, :w].copy())
    for i in range(len(bbox_crop)): bbox_crop[i] *= scale
    crop_margin_image = image_raw.crop(bbox_crop)
    if crop_margin_image.size[0] > min_picture_length / 2.54 * dpi and crop_margin_image.size[1] > min_picture_length / 2.54 * dpi:
        image_processing_threads.append(crop_and_rotate(crop_margin_image.copy(), folder_name))
        image_processing_threads[-1].start()
    
    image_draw.rectangle([0, 0, w, h], fill = tuple(background))
    image_draw_orig.rectangle([0, 0, min(w * scale, width_raw), min(h * scale, height_raw)], fill = tuple(background))
    
# waiting for all threads
for image_processing_thread in image_processing_threads:
    image_processing_thread.join()
