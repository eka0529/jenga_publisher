import pyrealsense2 as rs
from ultralytics import YOLO
import supervision as sv
import cv2
import yaml
import numpy as np
import torch
import math
import random
import time

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('747612060071')  # Specify the serial number for camera 1
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Configure the stream
pipeline.start(config)

# Load YOLO model
model = YOLO('best.pt')  # loading the YOLO model

# Load camera intrinsic matrix from YAML file
with open('newcalib_cam2.yaml', 'r') as f:
    dataDict = yaml.load(f, Loader=yaml.SafeLoader)
camera_matrix = np.array(dataDict['camera_matrix'])
dist_coeff = np.array(dataDict['dist_coeff'])
rvecs = np.array(dataDict['rvecs'])
tvecs = np.array(dataDict['tvecs'])

# Physical real-world parameters
SF = 83  # distance of the calibration surface plane from the camera in cm, this is the scaling factor
c_X_w = 20  # x coordinate of world frame in the camera frame
c_Y_w = 57 # y coordinate of world frame in the camera frame
c_Z_w = -83  # z coordinate of the world frame in the camera frame

# Rotation matrices
#theta_x = math.pi  # Rotation around the x-axis by 180 degrees
theta_z = math.pi   # Rotation around the z-axis by 90 degrees
#theta_y = -math.pi / 2  # Rotation around the z-axis by 90 degrees

#R2 = np.array([[math.cos(theta_x), 0, math.sin(theta_x)],
#               [0, 1, 0],
#               [-math.sin(theta_x), 0, math.cos(theta_x)]])

#R2 = np.array([[1, 0, 0],
#               [0, math.cos(theta_x), -math.sin(theta_x)],
#               [0, math.sin(theta_x), math.cos(theta_x)]])

R = np.array([[math.cos(theta_z), -math.sin(theta_z), 0],
               [math.sin(math.pi), math.cos(math.pi), 0],
               [0, 0, 1]])

#R = np.matmul(R1, R2)  # Combined rotation matrix
#R = np.matmul(R1, R2).T

#R = np.array([[math.cos(theta_y), 0, math.sin(theta_y)],
#             [0, 1, 0],
#             [-math.sin(theta_y), 0, math.cos(theta_y)]])


# Translation vector
t = np.array([[c_X_w], [c_Y_w], [c_Z_w]])  # translation in cm

# RotMat from world to camera
w_R_c = np.concatenate([R, t], 1)

e = np.array([[0, 0, 0, 1]])  # unit row vector
w_R_c = np.concatenate([w_R_c, e])  # complete 4x4 homogenous transformation from world to camera

# Function to calculate the World Pose from Pixel Coordinates
def world_coordinates(scale_factor, obb_xyxyxyxy, homTransfm):
    print(f"obb_xyxyxyxy shape: {obb_xyxyxyxy.shape}")
    
    # Using the camera matrix to go to Camera Frame from Pixel frame. Remember p = A*P_c
    p_lr = np.array([obb_xyxyxyxy[0][0][0], obb_xyxyxyxy[0][0][1], 1])
    P_c_lr = np.matmul(np.linalg.inv(camera_matrix), p_lr)
    P_c_lr = scale_factor * P_c_lr

    p_ur = np.array([obb_xyxyxyxy[0][1][0], obb_xyxyxyxy[0][1][1], 1])
    P_c_ur = np.matmul(np.linalg.inv(camera_matrix), p_ur)
    P_c_ur = scale_factor * P_c_ur

    p_ul = np.array([obb_xyxyxyxy[0][2][0], obb_xyxyxyxy[0][2][1], 1])
    P_c_ul = np.matmul(np.linalg.inv(camera_matrix), p_ul)
    P_c_ul = scale_factor * P_c_ul

    p_ll = np.array([obb_xyxyxyxy[0][3][0], obb_xyxyxyxy[0][3][1], 1])
    P_c_ll = np.matmul(np.linalg.inv(camera_matrix), p_ll)
    P_c_ll = scale_factor * P_c_ll

    # Calculate world coordinates
    P_w_lrT = np.matmul(homTransfm, np.array([P_c_lr[0], P_c_lr[1], P_c_lr[2], 1]))
    P_w_lr = np.array([P_w_lrT[1], P_w_lrT[0], -P_w_lrT[2], P_w_lrT[3]])
    
    P_w_urT = np.matmul(homTransfm, np.array([P_c_ur[0], P_c_ur[1], P_c_ur[2], 1]))
    P_w_ur = np.array([P_w_urT[1], P_w_urT[0], -P_w_urT[2], P_w_urT[3]])
    
    P_w_ulT = np.matmul(homTransfm, np.array([P_c_ul[0], P_c_ul[1], P_c_ul[2], 1]))
    P_w_ul = np.array([P_w_ulT[1], P_w_ulT[0], -P_w_ulT[2], P_w_ulT[3]])
    
    P_w_llT = np.matmul(homTransfm, np.array([P_c_ll[0], P_c_ll[1], P_c_ll[2], 1]))
    P_w_ll = np.array([P_w_llT[1], P_w_llT[0], -P_w_llT[2], P_w_llT[3]])
    
    print(P_w_lr)
    print(P_w_ur)
    print(P_w_ul)
    print(P_w_ll)
    
  
    
    
    
    
       


    


#def calculate_orientation(corners):
    # Calculate midpoints of bottom and top edges
    x_bottom = (obb_xyxyxyxy[0][3][0] + obb_xyxyxyxy[0][0][0]) / 2
    y_bottom = (obb_xyxyxyxy[0][3][1] + obb_xyxyxyxy[0][0][1]) / 2
    x_top = (obb_xyxyxyxy[0][2][0] + obb_xyxyxyxy[0][1][0]) / 2
    y_top = (obb_xyxyxyxy[0][2][1] + obb_xyxyxyxy[0][1][1]) / 2
    
    	

    # Calculate the orientation angle theta
    theta = math.atan2(y_top - y_bottom, x_top - x_bottom)
    theta_degreesT = math.degrees(theta)
    
    if theta_degreesT < 0:
        theta_degrees = -theta_degreesT
    else:
        theta_degrees = 180 - theta_degreesT

    
 
    
    print(theta)
    print(theta_degrees)
    
    yll = P_w_ll[0]
    ylr = P_w_lr[0]
    yul = P_w_ul[0]
    yur = P_w_ur[0]
    
    minim = yll
    minimx = P_w_ll[1]
    if ylr < minim:
        minim = ylr
        minimx = P_w_lr[1]
    if yul < minim:
        minim = yul
        minimx = P_w_ul[1]
    if yur < minim:
        minim = yur
        minimx = P_w_ur[1]
    
    l =  7.1
    cent_x = minimx + (l * math.sin(theta))
    cent_y = minim + (l * math.cos(theta))
    
    print(cent_x)
    print(cent_y)
    
    return [P_w_lr, P_w_ur, P_w_ul, P_w_ll]

#def centroid_calc():
 #   length = obb_xyxyxyxy[0][3][0] + obb_xyxyxyxy[0][0][0]
  #  breadth = obb_xyxyxyxy[0][3][1] + obb_xyxyxyxy[0][0][1]
   # print(length/2)
    #print(breadth/2)

# Function to display coordinates in a separate window
def display_coordinates(coordinates):
    display_image = np.zeros((600, 800, 3), dtype=np.uint8)  # Increase the size of the blank image
    y0, dy = 30, 30
    for i, coord in enumerate(coordinates):
        text = f"Block {i+1} Coordinates:"
        y = y0 + i * dy * 5  # Increase the spacing between blocks
        cv2.putText(display_image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        corner_names = ["Lower Right", "Upper Right", "Upper Left", "Lower Left"]
        for j, corner in enumerate(coord):
            corner_text = f" {corner_names[j]}: [{corner[0]:.2f}, {corner[1]:.2f}, {corner[2]:.2f}]"
            cv2.putText(display_image, corner_text, (10, y + (j + 1) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Coordinates', display_image)

# Generate random colors for bounding boxes
def generate_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors

# Main loop
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        img = np.asanyarray(color_frame.get_data())

        # Perform inference on the current frame
        results = model.predict(img, conf=0.79)  # performing inference

        # Process the results
        block = 0  # only fetching the box for a "{block}" type detection corresponding to the key value "0"
        coordinates = []

        if results[0].obb is not None:
            obb_boxes = results[0].obb
            colors = generate_colors(len(obb_boxes))  # Generate colors for each bounding box
            for i, obb in enumerate(obb_boxes):
                obb_xyxyxyxy = obb.xyxyxyxy.cpu().numpy()  # Ensure obb data is in numpy format
                print(f"obb_xyxyxyxy: {obb_xyxyxyxy}")  # Debugging statement to print the content of obb_xyxyxyxy
                scale_factor = SF  # Directly use the scaling factor
                
              

                # Calculate world coordinates
                corners = world_coordinates(scale_factor, obb_xyxyxyxy, w_R_c)
                coordinates.append([corner[:3] for corner in corners])
                
                # Calculate orientation
              #  angle = calculate_orientation(obb_xyxyxyxy)
                
                

                # Draw bounding box
                color = colors[i]
                for j in range(len(corners)):
                    p1 = (int(obb_xyxyxyxy[0][j][0]), int(obb_xyxyxyxy[0][j][1]))
                    p2 = (int(obb_xyxyxyxy[0][(j+1) % 4][0]), int(obb_xyxyxyxy[0][(j+1) % 4][1]))
                    cv2.line(img, p1, p2, color, 2)
                    
                    
                   

            # Display coordinates
            display_coordinates(coordinates)

        # Show images
        cv2.imshow('RealSense', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

