import cv2
import sys

from Source import utils

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    # trackers
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]

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

    # Read video
    video = cv2.VideoCapture("../InputData/cc2fCut.avi")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (407, 70, 40, 120)  # input here the bbox from object detection bounding box

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    change_x = 0
    change_y = 0
    change_total_in_a_unit = 0
    bbox_prev = bbox

    # determining polynomial coefficients for x and y
    input_values_in_camera_in_pixel_in_x = [114, 228, 342, 456, 570, 684, 798]
    output_values_in_camera_in_a_unit_in_x = [0.8, 1.7, 2.7, 3.8, 4.8, 5.7, 6.5]
    input_values_in_camera_in_pixel_in_y = [50, 100, 150, 200, 250, 300, 350, 400, 450]
    output_values_in_camera_in_a_unit_in_y = [0.8, 1.65, 2.55, 3.5, 4.5, 5.55, 6.65, 7.8, 9]
    polynomial_degree = 4
    polynomial_coefficients_in_x = utils.coefficients_of_polynomial(input_values_in_camera_in_pixel_in_x,
                                                                    output_values_in_camera_in_a_unit_in_x,
                                                                    polynomial_degree)
    polynomial_coefficients_in_y = utils.coefficients_of_polynomial(input_values_in_camera_in_pixel_in_y,
                                                                    output_values_in_camera_in_a_unit_in_y,
                                                                    polynomial_degree)
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
            change_total_in_a_unit += ((centroid_x_of_bbox_in_a_unit - centroid_x_of_bbox_in_a_unit_prev) ** 2 +
                                       (centroid_y_of_bbox_in_a_unit - centroid_y_of_bbox_in_a_unit_prev)
                                       ** 2) ** 0.5
            print(change_total_in_a_unit)

            bbox_prev = bbox
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
