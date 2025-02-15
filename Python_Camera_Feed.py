import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import time

# Constants
ROBOT_MARKER_ID = 0  # ID for the robot's marker
ESP32_BASE_URL = "http://192.168.12.183"
DISTANCE_THRESHOLD = 70  # Stop when within 50 pixels of the object
ANGLE_TOLERANCE = 15  # Tolerance for rotation (up to 10 degrees of deviation)
MOVEMENT_DURATION = 0.01  # Movement duration in seconds

# HSV color range for detecting the orange object
LOWER_ORANGE = np.array([5, 150, 150])
UPPER_ORANGE = np.array([15, 255, 255])

# Movement commands for the ESP32
COMMANDS = {
    "FORWARD": "/move_forward?speed=85",
    "BACKWARD": "/move_backward?speed=85",
    "RIGHT": "/turn_right?speed=70",
    "LEFT": "/turn_left?speed=70",
    "STOP": "/stop"
}

def send_command(command, duration=None):
    """Send a movement command to the ESP32."""
    if command in COMMANDS:
        url = ESP32_BASE_URL + COMMANDS[command]
        try:
            response = requests.get(url, timeout=0.5)
            print(f"Command sent: {command}, Response: {response.text}")
            if duration:
                time.sleep(duration)
                stop_response = requests.get(ESP32_BASE_URL + COMMANDS["STOP"], timeout=0.5)
                print(f"Command sent: STOP, Response: {stop_response.text}")
        except requests.RequestException as e:
            print(f"Error sending command: {e}")


# Camera calibration parameters
CAMERA_MATRIX = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((1, 5), dtype=np.float32)

# Initialize ArUco dictionaries
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
aruco_params = aruco.DetectorParameters_create()

def detect_orange_object(frame):
    """Detect the orange square and return its centroid."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, LOWER_ORANGE, UPPER_ORANGE)

    # Use morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, frame

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)

    if M["m00"] == 0:
        return None, frame

    # Compute the centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Draw the contour and centroid
    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    return (cx, cy), frame


def detect_robot(frame):
    """Detect the robot's AruCo marker and return its position and orientation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is None or ROBOT_MARKER_ID not in ids:
        return None, None, frame

    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, CAMERA_MATRIX, DIST_COEFFS)
    index = np.where(ids == ROBOT_MARKER_ID)[0][0]
    position = tvecs[index][0]
    orientation = rvecs[index][0]

    # Calculate the center of the marker in pixel coordinates
    marker_center = corners[index][0].mean(axis=0)

    # Draw the marker
    aruco.drawDetectedMarkers(frame, corners)
    cv2.circle(frame, tuple(marker_center.astype(int)), 5, (0, 0, 255), -1)

    # Draw the ArUco axes
    frame = draw_aruco_axes(frame, orientation, position)

    return marker_center, orientation, frame


def draw_aruco_axes(frame, rvec, tvec):
    """Draw the ArUco marker's axes on the frame."""
    axis_length = 0.1  # Length of the axes
    axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length], [0, 0, 0]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)

    origin = tuple(imgpts[3].ravel().astype(int))
    frame = cv2.line(frame, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 2)  # X-axis (Red)
    frame = cv2.line(frame, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 2)  # Y-axis (Green)
    frame = cv2.line(frame, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 2)  # Z-axis (Blue)

    return frame


def calculate_movement(robot_center, robot_ori, object_pos, frame):
    """Calculate the movement required to approach the object."""
    if robot_center is None or object_pos is None:
        return "STOP", frame

    # Compute offsets
    robot_x, robot_y = robot_center
    object_x, object_y = object_pos

    dx = object_x - robot_x
    dy = object_y - robot_y

    # Calculate the angle of the line connecting the robot and object
    angle_to_object = np.degrees(np.arctan2(dy, dx))

    # Extract the robot's Y-axis direction from the orientation (yaw rotation)
    robot_yaw = np.degrees(robot_ori[2])  # Assuming yaw (Z-axis rotation)

    # Calculate the angle difference between robot's front (Y-axis) and line to object
    angle_difference = (angle_to_object - robot_yaw + 360) % 360
    if angle_difference > 180:
        angle_difference -= 360

    print(f"Angle to Object: {angle_to_object}, Robot Yaw: {robot_yaw}, Angle Difference: {angle_difference}")

    # If the angle difference is larger than tolerance, rotate the robot
    if abs(angle_difference) > ANGLE_TOLERANCE:
        if angle_difference > 0:
            return "RIGHT", frame  # Rotate right
        else:
            return "LEFT", frame  # Rotate left

    # Move forward towards the object if aligned (within tolerance)
    if np.sqrt(dx ** 2 + dy ** 2) > DISTANCE_THRESHOLD:
        return "FORWARD", frame

    # Stop if reached
    return "STOP", frame


# Open the camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading from camera.")
        break

    # Detect the orange object
    object_pos, frame = detect_orange_object(frame)

    # Detect the robot's position and orientation
    robot_center, robot_ori, frame = detect_robot(frame)

    # Calculate the movement command
    command, frame = calculate_movement(robot_center, robot_ori, object_pos, frame)

    # Send the command to ESP32
    send_command(command, duration=MOVEMENT_DURATION)

    # Display the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
