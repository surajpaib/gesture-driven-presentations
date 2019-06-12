#from video_data import VideoData
import os
import matplotlib.pyplot as plt
import cv2
import numpy

# class Classifier:
# VideoData.load_xml_file()
folders = './data1'
data2 = 'data2'
for folder in os.listdir(folders):
    # print('folder', folder)
    for file in os.listdir(folders + '/' + folder):
        file_path = folders + '/' + folder + '/' + file
        write_path =  data2 + '/' + folder + '/' + file + '.npy'
        img = cv2.imread(file_path)
        img = numpy.array(im)
        cv2.imwrite(write_path, img)
