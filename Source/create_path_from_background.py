import cv2

from Source import utils


# click event for mouse
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ",", y)
        global points_on_path_size
        points_on_path.append([x, y])
        points_on_path_size += 1
        if points_on_path_size > 1:
            cv2.line(image, (points_on_path[points_on_path_size - 2][0], points_on_path[points_on_path_size - 2][1]),
                     (points_on_path[points_on_path_size - 1][0], points_on_path[points_on_path_size - 1][1]),
                     (255, 0, 0), 2)
        cv2.imshow("image", image)

    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.imshow("image", image)


# init variables
points_on_path = []
points_on_path_size = 0
total_distance_in_an_unit = 0

# read image
image = cv2.imread("../InputData/background_image.png")
cv2.imshow("image", image)

# mouse callback
cv2.setMouseCallback("image", click_event)

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

print("Total distance covered by the path in an unit:", total_distance_in_an_unit)
