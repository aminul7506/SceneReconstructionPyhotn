import numpy as np
import cv2


def coefficients_of_polynomial(x_values_array, y_values_array, degree):
    return np.polyfit(x_values_array, y_values_array, degree)


def get_value_from_four_degree_polynomial(polynomial_coefficients, x):
    a = polynomial_coefficients[0]
    b = polynomial_coefficients[1]
    c = polynomial_coefficients[2]
    d = polynomial_coefficients[3]
    e = polynomial_coefficients[4]

    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def get_frame_per_second_of_a_video(video, major_ver):
    if int(major_ver) < 3:
        return video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        return video.get(cv2.CAP_PROP_FPS)


def pixel_to_unit_converter_data(file_name):
    file = open(file_name, "r")
    array = []
    for line in file:
        res = [float(i) for i in line.split()]
        array.append(res)
    file.close()
    return array


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

            if diff1 <= THRESHOLD_DIFF_IN_BOUNDARY_BOX and diff2 <= THRESHOLD_DIFF_IN_BOUNDARY_BOX\
                    and diff3 <= THRESHOLD_DIFF_IN_BOUNDARY_BOX and diff4 <= THRESHOLD_DIFF_IN_BOUNDARY_BOX:
                top_left_coordinate_x = int(queryBorder[0][0][0])
                top_left_coordinate_y = int(queryBorder[0][0][1])
                width_of_bbox = abs(int(queryBorder[0][2][0] - queryBorder[0][0][0]))
                height_of_bbox = abs(int(queryBorder[0][1][1] - queryBorder[0][0][1]))
                return top_left_coordinate_x, top_left_coordinate_y, width_of_bbox, height_of_bbox

            print("Enough match found object is not in rectangular shape.")
        else:
            print("Not Enough match found.")

        if show_image:
            cv2.imshow('result', QueryImgBGR)
            if cv2.waitKey(10) == ord('v'):
                break

    return [0, 0, 0, 0]
