import cv2
import numpy as np
import mediapipe as mp
import os
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils

# Define indices for 12 key points
keypoint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Function to calculate angles between joints
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points: a, b, c.
    a, b, c are (x, y) coordinates.
    The angle is formed at point b.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba == 0 or mag_bc == 0:
        return 0
    angle = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle)

# Define joint triplets for angle calculation
joint_triplets = [
    (0, 2, 4),  # Left elbow
    (1, 3, 5),  # Right elbow
    (6, 0, 2),  # Left upper body
    (7, 1, 3),  # Right upper body
    (6, 8, 10),  # Left leg
    (7, 9, 11),  # Right leg
    (0, 6, 8),  # Left side of torso
    (1, 7, 9)   # Right side of torso
]

# Path to the folder containing images
folder_path = "/Users/alazartegegnework/Documents/Year 3/EE375/Final Project/Data/Raw_Images"

# Initialize arrays to store key points and labels
all_keypoints = []
labels = []
i = 1
# Process each image
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    # Determine the label based on the file name
    if "baby" in file_name.lower():
        label = "Baby freeze"
    elif "windmill" in file_name.lower():
        label = "Windmill"
    else:
        continue  # Skip files without valid labels
    
    # Process the image to extract key points
    keypoints = []
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        height, width, _ = image.shape
        for idx in keypoint_indices:
            landmark = results.pose_landmarks.landmark[idx]
            keypoints.append((landmark.x * width, landmark.y * height))
        all_keypoints.append(keypoints)
        labels.append(label)

# Calculate angles for all images
all_angles = []
for i, keypoints in enumerate(all_keypoints):
    angles = []
    keypoints = np.array(keypoints).reshape(12, 2)  # Convert to (12, 2)
    for joint_triplet in joint_triplets:
        a_idx, b_idx, c_idx = joint_triplet
        a, b, c = keypoints[a_idx], keypoints[b_idx], keypoints[c_idx]
        angle = calculate_angle(a, b, c)
        angles.append(angle)
    angles.append(labels[i])  # Append the label
    all_angles.append(angles)

# Convert angles to a NumPy array and save to CSV
angles_array = np.array(all_angles, dtype=object)
angles_csv = "/Users/alazartegegnework/Documents/Year 3/EE375/Final Project/Data/Joint_angles.csv"
header="Left elbow, Right elbow, Left shoulder, Right shoulder, Left knee, Right knee, Left hip, Right hip, Label"
np.savetxt(angles_csv, angles_array, fmt='%s', delimiter=",", header=header)
print(f"Angles with labels saved to {angles_csv}")