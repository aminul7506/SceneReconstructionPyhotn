import cv2
import sys

from Source import utils

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    # trackers
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[6]

    # set tracker
    if int(major_ver) < 4 and int(minor_ver) < 3:
        tracker = cv2.cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    if tracker is None:
        tracker = cv2.TrackerMIL_create()

    # Open video
    video = cv2.VideoCapture("../InputData/cc2fCut.avi")
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    print("------------------------------------- frame reading started -----------------------------------------")

    # detection started
    # detect input object in video
    input_image = cv2.imread("../InputData/zahid1.png", 0)
    bbox = utils.compute_first_detection_boundary_box(video, input_image, True)

    # tracking started
    # Read next frame after detection
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    change_x = 0
    change_y = 0
    distance_covered_in_a_unit = 0
    bbox_prev = bbox

    # determining polynomial coefficients for x and y
    array_of_pixel_to_unit_data = utils.pixel_to_unit_converter_data("../InputData/pixel_to_unit.txt")
    input_values_in_camera_in_pixel_in_x = array_of_pixel_to_unit_data[0]
    output_values_in_camera_in_a_unit_in_x = array_of_pixel_to_unit_data[1]
    input_values_in_camera_in_pixel_in_y = array_of_pixel_to_unit_data[2]
    output_values_in_camera_in_a_unit_in_y = array_of_pixel_to_unit_data[3]
    polynomial_degree = 4
    polynomial_coefficients_in_x = utils.coefficients_of_polynomial(input_values_in_camera_in_pixel_in_x,
                                                                    output_values_in_camera_in_a_unit_in_x,
                                                                    polynomial_degree)
    polynomial_coefficients_in_y = utils.coefficients_of_polynomial(input_values_in_camera_in_pixel_in_y,
                                                                    output_values_in_camera_in_a_unit_in_y,
                                                                    polynomial_degree)
    frame_count_for_object = 0
    frame_per_second = utils.get_frame_per_second_of_a_video(video, major_ver)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            # calculate centroid diff in current frame in pixel
            centroid_x_of_bbox = bbox[0] + bbox[2] / 2
            centroid_y_of_bbox = bbox[1] + bbox[3] / 2

            # calculate centroid diff in prev frame in pixel
            centroid_x_of_bbox_prev = bbox_prev[0] + bbox_prev[2] / 2
            centroid_y_of_bbox_prev = bbox_prev[1] + bbox_prev[3] / 2

            # calculate centroid diff in x and y in a unit
            centroid_x_of_bbox_in_a_unit = utils.get_value_from_four_degree_polynomial(polynomial_coefficients_in_x,
                                                                                       centroid_x_of_bbox)
            centroid_y_of_bbox_in_a_unit = utils.get_value_from_four_degree_polynomial(polynomial_coefficients_in_y,
                                                                                       centroid_y_of_bbox)
            centroid_x_of_bbox_in_a_unit_prev = utils.get_value_from_four_degree_polynomial(
                polynomial_coefficients_in_x,
                centroid_x_of_bbox_prev)
            centroid_y_of_bbox_in_a_unit_prev = utils.get_value_from_four_degree_polynomial(
                polynomial_coefficients_in_y,
                centroid_y_of_bbox_prev)
            change_of_centroid_in_x_in_a_unit = abs(centroid_x_of_bbox_in_a_unit - centroid_x_of_bbox_in_a_unit_prev)
            change_of_centroid_in_y_in_a_unit = abs(centroid_y_of_bbox_in_a_unit - centroid_y_of_bbox_in_a_unit_prev)

            # calculate total diff in current frame in a unit
            distance_covered_in_a_unit += ((centroid_x_of_bbox_in_a_unit - centroid_x_of_bbox_in_a_unit_prev) ** 2 +
                                           (centroid_y_of_bbox_in_a_unit - centroid_y_of_bbox_in_a_unit_prev)
                                           ** 2) ** 0.5
            # update variables
            bbox_prev = bbox
            frame_count_for_object += 1
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    print("------------------------------------- frame reading ended -----------------------------------------")

    # calculate velocity
    total_present_time_in_seconds_for_object = frame_count_for_object / frame_per_second
    average_velocity_of_the_input_object = distance_covered_in_a_unit / total_present_time_in_seconds_for_object

    # print info
    print("Total distance covered in a unit by the object :", distance_covered_in_a_unit)
    print("Total present time in seconds in the video by the object :", total_present_time_in_seconds_for_object)
    print("Average velocity of the object in the video file in unit per second :", average_velocity_of_the_input_object)