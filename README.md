# Autonomous-Robot-with-OpenCV-ESP32

This project demonstrates an autonomous robot that uses OpenCV for color detection and ArUco markers for navigation. The robot detects a colored object, aligns itself, and moves toward it using commands sent wirelessly via an ESP32 microcontroller.

Key Features:
ArUco Marker Detection: Identifies the robot's position and orientation in real-time.

Color Detection: Uses HSV color filtering to detect a colored object (e.g., orange).

Wireless Control: Commands are sent to the ESP32 to control the robot's movement.

Dynamic Navigation: The robot adjusts its path to align with the object and stops when it reaches the target.

Tech Stack:
OpenCV for computer vision (ArUco marker detection and color filtering).

ESP32 for wireless communication and motor control.

L298N Motor Driver to control the robot's motors.

Python for the main logic and OpenCV integration.
