import cv2

from Source import utils


# click event for mouse
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Point clicked in pixel (" + str(x) + ",", str(y) + ")")
        global points_on_path_size
        points_on_path.append([x, y])
        points_on_path_size += 1
        if points_on_path_size > 1:
            cv2.line(image, (points_on_path[points_on_path_size - 2][0], points_on_path[points_on_path_size - 2][1]),
                     (points_on_path[points_on_path_size - 1][0], points_on_path[points_on_path_size - 1][1]),
                     (255, 0, 0), 2)
        elif points_on_path_size == 1:
            cv2.line(image, (points_on_path[points_on_path_size - 1][0], points_on_path[points_on_path_size - 1][1]),
                     (points_on_path[points_on_path_size - 1][0], points_on_path[points_on_path_size - 1][1]),
                     (255, 0, 0), 2)

        cv2.imshow(input_file_name, image)

    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.imshow(input_file_name, image)


# init variables
points_on_path = []
points_on_path_size = 0
total_distance_in_an_unit = 0

# read image
input_directory = "../InputData"
input_file_name = "background_image"
input_file_extension = ".png"
input_file_path = input_directory + "/" + input_file_name + input_file_extension
image = cv2.imread(input_file_path)
cv2.imshow(input_file_name, image)

# mouse callback
cv2.setMouseCallback(input_file_name, click_event)

# clear windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# determining polynomial coefficients for x and y
polynomial_coefficients_in_x, polynomial_coefficients_in_y = \
    utils.determine_polynomial_coefficient_in_x_y("../InputData/pixel_to_unit.txt")

# calculate total path distance pointed by mouse in an unit
for i in range(points_on_path_size):
    if i > 0:
        total_distance_in_an_unit += utils.calculate_difference_of_two_points_in_an_unit(polynomial_coefficients_in_x,
                                                                                         polynomial_coefficients_in_y,
                                                                                         points_on_path[i - 1][0],
                                                                                         points_on_path[i - 1][1],
                                                                                         points_on_path[i][0],
                                                                                         points_on_path[i][1])

# save the image with path specification
output_file_directory = "../OutputData"
output_file_name = input_file_name + "_with_path"
output_file_path = output_file_directory + "/" + output_file_name + input_file_extension
cv2.imwrite(output_file_path, image)

# save pixel info into a .txt file
path_pixel_output_file_path = output_file_directory + "/" + output_file_name + ".txt"
file_for_output_path_info = open(path_pixel_output_file_path, "w")

# first write total distance of the path in an unit and then pixels of the path
total_path_length_in_an_unit_upto_two_decimal_in_str = str(round(total_distance_in_an_unit, 2)) + "\n"
file_for_output_path_info.write(total_path_length_in_an_unit_upto_two_decimal_in_str)
for point in points_on_path:
    point_in_str = str(point[0]) + " " + str(point[1]) + "\n"
    file_for_output_path_info.write(point_in_str)

file_for_output_path_info.close()

print("Total distance covered by the path in an unit :", total_distance_in_an_unit)
