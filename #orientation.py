#orientation         # Compute the yaw change over dt seconds:

import math 
import cv2 
import numpy as np 
def compute_yaw_change(rpm_fl, rpm_fr, rpm_rl, rpm_rr, wheel_radius, track_width, dt):

    v_fl = 2 * math.pi * wheel_radius * (rpm_fl / 60)
    v_fr = 2 * math.pi * wheel_radius * (rpm_fr / 60)
    v_rl = 2 * math.pi * wheel_radius * (rpm_rl / 60)
    v_rr = 2 * math.pi * wheel_radius * (rpm_rr / 60)
    v_left = (v_fl + v_rl) / 2
    v_right = (v_fr + v_rr) / 2
    omega = (v_right - v_left) / track_width
    delta_yaw = omega * dt
    return delta_yaw


def calculate_initial_compass_bearing(pointA, pointB):
    lat1, lon1 = pointA
    lat2, lon2 = pointB
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    diffLong = math.radians(lon2 - lon1)

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def compute_relative_yaw(img1, img2, focal_length, principal_point, max_matches=100):
    orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        print("Warning: Insufficient features detected")
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:max_matches]
    if len(matches) < 8:
        print(f"Warning: Only {len(matches)} matches found, need at least 8")
        return None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    try:
        E, mask = cv2.findEssentialMat(
            pts1, pts2, 
            focal=focal_length,
            pp=principal_point,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            print("Warning: Essential matrix computation failed")
            return None
            
        _, R, t, mask_pose = cv2.recoverPose(
            E, pts1, pts2,
            focal=focal_length,
            pp=principal_point,
            mask=mask
        )
        
        yaw = math.atan2(R[1, 0], R[0, 0])
        yaw_deg = math.degrees(yaw)
        
        return yaw_deg
        
    except cv2.error as e:
        print(f"Error in pose estimation: {e}")
        return None

