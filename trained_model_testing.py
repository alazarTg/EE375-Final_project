# We first start by import basic libraries and also the mediapipe library for pose estimation
import mediapipe as mp
import pandas as pd
import cv2
import math
import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from autograd import grad 

image_path = "/Users/alazartegegnework/Documents/Year 3/EE375/Final Project/rando.jpeg"  # Path to the new image

def feature_transforms(x_pass, theta):
    # Generate transformed features
    f = np.vstack([x_pass**d for d in range(1, (theta.size // 8) + 1)])
    return f

def model(x_pass, theta):
    f = feature_transforms(x_pass, theta)
    a = np.dot(f.T, theta)
    return a.T

def least_squares(x_pass,y_pass,w):
    cost = np.sum((model(x_pass,w) - y_pass)**2)
    return cost/(2*y_pass.size) # scaled to avoid overflow

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points: a, b, c.
    a, b, c are (x, y) coordinates.
    The angle is formed at point b.
    """
    # Create vectors
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    # Calculate dot product and magnitude
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    # Prevent division by zero
    if mag_ba == 0 or mag_bc == 0:
        return 0
    
    # Calculate the angle in radians and convert to degrees
    angle = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle)

def process_new_image(image_path):

    mp_pose = mp.solutions.pose.Pose()

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = mp_pose.process(image_rgb)

    # Define indices for 12 key points
    keypoint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    # Extract key points
    keypoints_12 = []
    if results.pose_landmarks:
        height, width, _ = image.shape
        for idx in keypoint_indices:
            landmark = results.pose_landmarks.landmark[idx]
            keypoints_12.append((int(landmark.x * width), int(landmark.y * height)))

    joint_triplets = [
        (11, 13, 15),  # Left elbow
        (12, 14, 16),  # Right elbow
        (23, 11, 13),  # Left shoulder
        (24, 12, 14),  # Right shoulder
        (23, 25, 27),  # Left leg
        (24, 26, 28),  # Right leg
        (11, 23, 25),  # Left side of torso
        (12, 24, 26)   # Right side of torso
    ]

    # Extract angles
    angles = []
    if results.pose_landmarks:
        height, width, _ = image.shape
        landmarks = results.pose_landmarks.landmark
        for joint_triplet in joint_triplets:
            a_idx, b_idx, c_idx = joint_triplet
            
            # Get the (x, y) positions of the joints
            a = (landmarks[a_idx].x * width, landmarks[a_idx].y * height)
            b = (landmarks[b_idx].x * width, landmarks[b_idx].y * height)
            c = (landmarks[c_idx].x * width, landmarks[c_idx].y * height)
            
            # Calculate angle and append
            angle = calculate_angle(a, b, c)
            angles.append(angle)
    return np.array(angles)

def normalize_features(features, mean, std):
    # Prevent division by zero
    std = std if std != 0 else 1
    return (features - mean) / std

def classify_image(image_path, weights, mean_train, std_train):
    # Preprocess the image
    features = process_new_image(image_path)

    # Normalize the features
    normalized_features = normalize_features(features, mean_train, std_train)

    # Apply feature transformations
    transformed_features = feature_transforms(normalized_features, weights)

    # Make predictions
    score = model(transformed_features, weights)
    prediction = 1 if score >= 0 else -1  # Threshold at 0

    return prediction

new_image_features = process_new_image(image_path)
x_new = np.array(new_image_features).reshape(-1, 1)
x_mean = np.mean(x_new)
x_std = np.std(x_new)
x_new_norm = (x_new - x_mean) / x_std

csvname = '/Users/alazartegegnework/Documents/Year 3/EE375/Final Project/best_weights.csv'
weights = np.asarray(pd.read_csv(csvname, header = None))
print(weights)

# Step 4: Compute the model output
model_output = model(x_new_norm, weights).flatten()[0]  # Extract scalar value
print(model_output)
# Step 5: Make a prediction
prediction = 1 if model_output >= 0 else -1

# Step 6: Map the prediction to the class name
label_map = { -1: 'Baby freeze', 1: 'Windmill' }
predicted_label = label_map[prediction]

print(f"The predicted class for the new image is: {predicted_label}")