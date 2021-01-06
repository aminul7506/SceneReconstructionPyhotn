import numpy as np
import cv2
import math


def coefficients_of_polynomial(x_values_array, y_values_array, degree):
    return np.polyfit(x_values_array, y_values_array, degree)


def get_value_from_four_degree_polynomial(polynomial_coefficients, x):
    a = polynomial_coefficients[0]
    b = polynomial_coefficients[1]
    c = polynomial_coefficients[2]
    d = polynomial_coefficients[3]
    e = polynomial_coefficients[4]

    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def get_real_positive_valid_root_from_four_degree_polynomial(polynomial_coefficients,
                                                             right_hand_side_value, max_value):
    a = polynomial_coefficients[0]
    b = polynomial_coefficients[1]
    c = polynomial_coefficients[2]
    d = polynomial_coefficients[3]
    e = polynomial_coefficients[4] - right_hand_side_value
    coefficients_array = [a, b, c, d, e]
    values_of_roots = np.roots(coefficients_array)
    real_roots = []
    for root in values_of_roots:
        if np.isreal(root):
            if 0 <= root <= max_value:
                real_roots.append(float(root))
    if len(real_roots) > 0:
        return real_roots[0]
    return -1


def get_angle_with_x_axis_of_a_straight_line_passing_two_points(x1, y1, x2, y2):
    if x2 == x1:
        return 3.1416 / 2
    return math.atan((y2 - y1) / (x2 - x1))


def get_x_y_value_in_pixel_from_distance_in_unit_and_angle_with_x_axis(angle_with_x_axis_in_radian,
                                                                       polynomial_coefficients_in_x,
                                                                       polynomial_coefficients_in_y,
                                                                       distance_in_unit, max_value_in_x,
                                                                       max_value_in_y):
    x_in_a_unit = distance_in_unit * math.cos(angle_with_x_axis_in_radian)
    y_in_a_unit = distance_in_unit * math.sin(angle_with_x_axis_in_radian)
    if x_in_a_unit < 0:
        x_in_a_unit = -x_in_a_unit
    if y_in_a_unit < 0:
        y_in_a_unit = -y_in_a_unit
    x_in_pixel = get_real_positive_valid_root_from_four_degree_polynomial(polynomial_coefficients_in_x, x_in_a_unit,
                                                                          max_value_in_x)
    y_in_pixel = get_real_positive_valid_root_from_four_degree_polynomial(polynomial_coefficients_in_y, y_in_a_unit,
                                                                          max_value_in_y)
    return x_in_pixel, y_in_pixel


def construct_object_movement_array_from_path_array(path_array, total_frame_count, polynomial_coefficients_in_x,
                                                    polynomial_coefficients_in_y, distance_in_unit_per_frame,
                                                    max_value_in_x, max_value_in_y):
    length_path_array = len(path_array)
    movement_array = []
    if length_path_array > 1:
        current_pixel_from_path = path_array[0]
        next_pixel_from_path = path_array[1]
        index_in_next_pixel_in_path_array = 1
        for i in range(total_frame_count):
            x1, y1 = current_pixel_from_path[0], current_pixel_from_path[1]
            x2, y2 = next_pixel_from_path[0], next_pixel_from_path[1]
            angle_with_x_axis_in_radian = get_angle_with_x_axis_of_a_straight_line_passing_two_points(
                x1, y1, x2, y2)
            x, y = get_x_y_value_in_pixel_from_distance_in_unit_and_angle_with_x_axis(
                angle_with_x_axis_in_radian, polynomial_coefficients_in_x,
                polynomial_coefficients_in_y, distance_in_unit_per_frame, max_value_in_x, max_value_in_y)
            change_next = False
            if x2 > x1 and y2 > y1:
                x = x1 + x
                y = y1 + y
                if x2 < x and y2 < y:
                    change_next = True
            elif x2 < x1 and y2 > y1:
                x = x1 - x
                y = y1 + y
                if x2 > x and y2 < y:
                    change_next = True
            elif x2 > x1 and y2 < y1:
                x = x1 + x
                y = y1 - y
                if x2 < x and y2 > y:
                    change_next = True
            elif x2 < x1 and y2 < y1:
                x = x1 - x
                y = y1 - y
                if x2 > x and y2 > y:
                    change_next = True
            movement_array.append([x, y])
            current_pixel_from_path = [x, y]
            if change_next and index_in_next_pixel_in_path_array < length_path_array - 1:
                index_in_next_pixel_in_path_array += 1
                next_pixel_from_path = path_array[index_in_next_pixel_in_path_array]
            print("x & y value in pixel in object movement array : (x,y) (" + str(x) + "," + str(y) + ")")
    return movement_array


def get_frame_per_second_of_a_video(video, major_ver):
    if int(major_ver) < 3:
        return video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        return video.get(cv2.CAP_PROP_FPS)


def get_video_width_height(video_file):
    width = video_file.get(3)
    height = video_file.get(4)
    return int(width), int(height)


def pixel_to_unit_converter_data(file_name):
    file = open(file_name, "r")
    array = []
    for line in file:
        res = [float(i) for i in line.split()]
        array.append(res)
    file.close()
    return array


def determine_polynomial_coefficient_in_x_y(file_name):
    array_of_pixel_to_unit_data = pixel_to_unit_converter_data(file_name)
    input_values_in_camera_in_pixel_in_x = array_of_pixel_to_unit_data[0]
    output_values_in_camera_in_an_unit_in_x = array_of_pixel_to_unit_data[1]
    input_values_in_camera_in_pixel_in_y = array_of_pixel_to_unit_data[2]
    output_values_in_camera_in_an_unit_in_y = array_of_pixel_to_unit_data[3]
    polynomial_degree = 4
    polynomial_coefficients_in_x = coefficients_of_polynomial(input_values_in_camera_in_pixel_in_x,
                                                              output_values_in_camera_in_an_unit_in_x,
                                                              polynomial_degree)
    polynomial_coefficients_in_y = coefficients_of_polynomial(input_values_in_camera_in_pixel_in_y,
                                                              output_values_in_camera_in_an_unit_in_y,
                                                              polynomial_degree)
    return polynomial_coefficients_in_x, polynomial_coefficients_in_y


def calculate_difference_of_two_points_in_an_unit(polynomial_coefficients_in_x, polynomial_coefficients_in_y, x1, y1,
                                                  x2, y2):
    x2_in_an_unit = get_value_from_four_degree_polynomial(polynomial_coefficients_in_x, x2)
    y2_in_an_unit = get_value_from_four_degree_polynomial(polynomial_coefficients_in_y, y2)
    x1_in_an_unit = get_value_from_four_degree_polynomial(polynomial_coefficients_in_x, x1)
    y1_in_an_unit = get_value_from_four_degree_polynomial(polynomial_coefficients_in_y, y1)

    return ((x2_in_an_unit - x1_in_an_unit) ** 2 + (y2_in_an_unit - y1_in_an_unit) ** 2) ** 0.5


def compute_first_detection_boundary_box(video, input_image, show_image):
    MIN_MATCH_COUNT = 20
    THRESHOLD_DIFF_IN_BOUNDARY_BOX = 20

    detector = cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDITREE = 0
    flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
    flann = cv2.FlannBasedMatcher(flannParam, {})

    trainKP, trainDesc = detector.detectAndCompute(input_image, None)

    while True:
        ret, QueryImgBGR = video.read()
        if not ret:
            break

        QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
        queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
        matches = flann.knnMatch(queryDesc, trainDesc, k=2)

        goodMatch = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatch.append(m)
        if len(goodMatch) > MIN_MATCH_COUNT:
            tp = []
            qp = []
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp, qp = np.float32((tp, qp))
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
            h, w = input_image.shape
            trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
            queryBorder = cv2.perspectiveTransform(trainBorder, H)
            cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 5)

            diff1 = abs(queryBorder[0][0][0] - queryBorder[0][1][0])
            diff2 = abs(queryBorder[0][2][0] - queryBorder[0][3][0])
            diff3 = abs(queryBorder[0][0][1] - queryBorder[0][3][1])
            diff4 = abs(queryBorder[0][1][1] - queryBorder[0][2][1])

            if diff1 <= THRESHOLD_DIFF_IN_BOUNDARY_BOX and diff2 <= THRESHOLD_DIFF_IN_BOUNDARY_BOX \
                    and diff3 <= THRESHOLD_DIFF_IN_BOUNDARY_BOX and diff4 <= THRESHOLD_DIFF_IN_BOUNDARY_BOX:
                top_left_coordinate_x = int(queryBorder[0][0][0])
                top_left_coordinate_y = int(queryBorder[0][0][1])
                width_of_bbox = abs(int(queryBorder[0][2][0] - queryBorder[0][0][0]))
                height_of_bbox = abs(int(queryBorder[0][1][1] - queryBorder[0][0][1]))
                return top_left_coordinate_x, top_left_coordinate_y, width_of_bbox, height_of_bbox

            print("Enough match found but object is not in an rectangular shape.")
        else:
            print("Not Enough match found.")

        if show_image:
            cv2.imshow('result', QueryImgBGR)
            if cv2.waitKey(10) == ord('v'):
                break

    return [0, 0, 0, 0]


def save_distance_along_with_path_info_in_a_txt_file(file_name, points_on_path, total_distance_in_an_unit,
                                                     polynomial_coefficients_in_x, polynomial_coefficients_in_y):
    file_for_output_path_info = open(file_name, "w")

    total_path_length_in_an_unit_up_to_two_decimal_in_str = str(round(total_distance_in_an_unit, 2)) + "\n"
    file_for_output_path_info.write(total_path_length_in_an_unit_up_to_two_decimal_in_str)

    polynomial_coefficients_in_x_in_string = str(polynomial_coefficients_in_x[0]) + " " \
                                             + str(polynomial_coefficients_in_x[1]) + " " \
                                             + str(polynomial_coefficients_in_x[2]) + " " \
                                             + str(polynomial_coefficients_in_x[3]) + " " \
                                             + str(polynomial_coefficients_in_x[4]) + "\n"
    file_for_output_path_info.write(polynomial_coefficients_in_x_in_string)

    polynomial_coefficients_in_y_in_string = str(polynomial_coefficients_in_y[0]) + " " \
                                             + str(polynomial_coefficients_in_y[1]) + " " \
                                             + str(polynomial_coefficients_in_y[2]) + " " \
                                             + str(polynomial_coefficients_in_y[3]) + " " \
                                             + str(polynomial_coefficients_in_y[4]) + "\n"
    file_for_output_path_info.write(polynomial_coefficients_in_y_in_string)

    for point in points_on_path:
        point_in_str = str(int(point[0])) + " " + str(int(point[1])) + "\n"
        file_for_output_path_info.write(point_in_str)
    file_for_output_path_info.close()


def create_video_file_from_images(image_array, file_path, image_width, image_height, frame_rate):
    dimension = (image_width, image_height)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(file_path, fourcc, frame_rate, dimension)
    for image in image_array:
        video.write(image)
    video.release()


def read_distance_pixel_data_from_path_specified_file(file_path):
    file = open(file_path, "r")
    total_distance = 0
    polynomial_coefficients_in_x = []
    polynomial_coefficients_in_y = []
    pixel_array = []
    index = 0
    for line in file:
        if index == 0:
            res = [float(i) for i in line.split()]
            total_distance = res[0]
        elif index == 1:
            polynomial_coefficients_in_x = [float(i) for i in line.split()]
        elif index == 2:
            polynomial_coefficients_in_y = [float(i) for i in line.split()]
        else:
            res = [int(i) for i in line.split()]
            pixel_array.append(res)
        index += 1
    file.close()
    return total_distance, polynomial_coefficients_in_x, polynomial_coefficients_in_y, pixel_array
