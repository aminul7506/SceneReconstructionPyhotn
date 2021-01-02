from PIL import Image
import cv2
import numpy as np

from Source import utils

# take inputs
input_directory = "../InputData"
image_input_file_name = "background_image"
image_input_file_extension = ".png"
image_input_file_path = input_directory + "/" + image_input_file_name + image_input_file_extension
background_image = Image.open(image_input_file_path)

image_input_file_name_of_the_object = "zahid2"
image_input_file_path_of_the_object = input_directory + "/" + image_input_file_name_of_the_object\
                                      + image_input_file_extension
input_image = Image.open(image_input_file_path_of_the_object)

output_file_directory = "../OutputData"
pixel_input_file_name = image_input_file_name + "_with_path"
pixel_input_file_extension = ".txt"
pixel_input_file_path = output_file_directory + "/" + pixel_input_file_name + pixel_input_file_extension
total_distance, pixel_array = utils.read_distance_pixel_data_from_path_specified_file(pixel_input_file_path)

# construct image array for inserting into video file
image_array = []
for point in pixel_array:
    back_im = background_image.copy()
    back_im.paste(input_image, (point[0], point[1]))
    cv2_image = cv2.cvtColor(np.array(back_im), cv2.COLOR_RGB2BGR)
    image_array.append(cv2_image)

# insert into video file
output_video_file_name = pixel_input_file_name
output_video_file_extension = ".avi"
output_video_file_path = output_file_directory + "/" + \
                         output_video_file_name + output_video_file_extension
image_width, image_height = background_image.size
utils.create_video_file_from_images(image_array, output_video_file_path, image_width, image_height, frame_rate=20)
