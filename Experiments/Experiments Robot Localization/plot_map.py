import cv2
import os
import pandas
import numpy as np
import copy
import matplotlib.pyplot as plt



def antenna_tranformation(id,robot_amcl):
    

    if id==1:
        x = 0.0
        y = 0.095
        z = 0.633

    elif id==2:

        x = 0.0
        y = -0.095
        z = 0.633

    elif id==3:

        x = 0.0
        y = -0.095
        z = 1.073

    elif id==4:

        x = 0.0
        y = 0.095
        z = 1.073

    DA = np.array([x,y,z])

    antenna_amcl = copy.copy(robot_amcl)
    
    for i in range(len(robot_amcl)):
        
        x_r = robot_amcl[i,1]
        y_r = robot_amcl[i,2]
        theta_r = robot_amcl[i,4] + np.pi

        A_r = np.transpose(np.array([x_r,y_r,0]))

        rotation = np.zeros((3,3))
        rotation[0,0] = np.cos(theta_r)
        rotation[0,1] = - np.sin(theta_r)
        rotation[0,2] = 0
        rotation[1,0] = np.sin(theta_r)
        rotation[1,1] = np.cos(theta_r)
        rotation[1,2] = 0
        rotation[2,0] = 0
        rotation[2,1] = 0
        rotation[2,2] = 1 

        antenna_pose =  A_r + np.matmul(rotation,np.transpose(DA))
        antenna_amcl[i,1] = antenna_pose[0]
        antenna_amcl[i,2] = antenna_pose[1]
    
    return antenna_amcl




img = cv2.imread('csal_2023_03_27_0.01.pgm')

height = 1249
width = 1461
resolution = 0.01





path = "4\\"

filename_odom= path + "odometry.txt"

if (os.path.exists(filename_odom)) and ( not (os.stat(filename_odom).st_size == 0)):

        robot_odometry=pandas.read_csv(filename_odom, sep=",",engine ='python', header=None)
   
        robot_odometry=(robot_odometry).to_numpy()

        robot_odometry[:,1] = robot_odometry[:,1]/resolution
        robot_odometry[:,2] = height - robot_odometry[:,2]/resolution

filename_amcl= path + "pose_robot.txt"

if (os.path.exists(filename_amcl)) and ( not (os.stat(filename_amcl).st_size == 0)):

    robot_amcl=pandas.read_csv(filename_amcl, sep=",",engine ='python', header=None)

    robot_amcl=(robot_amcl).to_numpy()


    antenna_amcl = antenna_tranformation(1,robot_amcl)
    antenna_amcl[:,1] = antenna_amcl[:,1]/resolution
    antenna_amcl[:,2] = height - antenna_amcl[:,2]/resolution

  


    robot_amcl[:,1] = robot_amcl[:,1]/resolution
    robot_amcl[:,2] = height - robot_amcl[:,2]/resolution


filename_tags=  "tags_trans.csv"   

if (os.path.exists(filename_tags)) and ( not (os.stat(filename_tags).st_size == 0)):
        
        coords=pandas.read_csv(filename_tags, header=None)
        
        if len(coords):                     
            tag_coords = (coords.select_dtypes(include=['float64','int64'])).to_numpy()                
            tag_epcs=(coords.select_dtypes(include=['object'])).to_numpy()


color = ( 255, 0, 0)

for i in range(0,len(tag_coords)):

    
    img = cv2.circle(img, (int(tag_coords[i,0]), int(tag_coords[i,1])), 2, tuple(color), -1)



color = ( 0, 255, 0)

for i in range(0,len(robot_amcl)):

    color = ( 0, 255, 0)
    img = cv2.circle(img, (int(robot_amcl[i,1]), int(robot_amcl[i,2])), 2, tuple(color), -1)

    
    
    # img = cv2.circle(img, (int(antenna_amcl[i,1]), int(antenna_amcl[i,2])), 2, tuple(color), -1)



color = ( 0, 0, 255)
for i in range(0,len(robot_odometry)):
    
    img = cv2.circle(img, (int(robot_odometry[i,1]), int(robot_odometry[i,2])), 2, tuple(color), -1)



img = cv2.resize(img, (0,0), fx=0.6, fy=0.6) 

cv2.imshow('image', img)

# Maintain output window utill
# user presses a key
cv2.waitKey(0)       
 
# Destroying present windows on screen
cv2.destroyAllWindows()


