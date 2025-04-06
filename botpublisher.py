import rospy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from cv_bridge import CvBridge
import tf
from tf.transformations import quaternion_from_euler
import math
from helper_func import compute_velocity_from_rpms
import cv2
import numpy as np
class ROSPublisher:
    def __init__(self, start_utm):
        # Initialize the ROS node
        self.start_utm = start_utm
        rospy.init_node('data_publisher', anonymous=True)
        print("ROS node initialized")

        self.image_pub = rospy.Publisher('/camera/image', Image, queue_size=10)
        self.gps_pub = rospy.Publisher('/gps_odom', Odometry, queue_size=10)
        self.imu_pub = rospy.Publisher('/imu/orientation', Imu, queue_size=10)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        # Initialize CvBridge for converting OpenCV images to ROS Image messages
        self.bridge = CvBridge()

        #subsriber part of the code 
        self.latest_state = None
        self.previous_time = 0 
        self.gps_previous_time = 0.0
        self.odom_filtered_sub = rospy.Subscriber('/odometry/filtered/global', Odometry, self.odom_filtered_callback)

    def publish_image(self, image):
        cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.image_pub.publish(ros_image)

    def publish_data(self,v, w, orient, utm_x, utm_y, timestamps= None):
        self.publish_gps(utm_x,utm_y,gps_time_stamp= timestamps[0])
        self.publish_imu(orient,timestamps[2])
        self.publish_odom(v,w, timestamps[1])
    
    def publish_gps(self, utm_x, utm_y, gps_time_stamp= None):
        """Publish GPS data in UTM format to /odom2 topic."""
        #condition to verify the timestamp
        # print(self.gps_previous_time,gps_time_stamp)
        if abs( self.gps_previous_time -float(gps_time_stamp)) >.7:
            self.gps_previous_time = float(gps_time_stamp)
            odom_msg1 = Odometry() 
            #TODO: Initializee in init, justt  update the timestamp and position here
            odom_msg1.header.stamp = rospy.Time.from_sec(float(gps_time_stamp))
            odom_msg1.header.frame_id = 'map'  # Global frame for GPS data
            odom_msg1.child_frame_id = 'base_link'
            odom_msg1.pose.pose.position.x = utm_x - self.start_utm[0]
            odom_msg1.pose.pose.position.y = utm_y -self.start_utm[1]
            odom_msg1.pose.pose.position.z = 0.0
            odom_msg1.pose.covariance[0] =20
            odom_msg1.pose.covariance[7] =20

            # Twist is set to zero since GPS typically provides position only
            self.gps_pub.publish(odom_msg1)

    def publish_imu(self, yaw_degrees, imu_timestamp = None):
        imu_msg = Imu()
        imu_msg.header.stamp =rospy.Time.from_sec(float(imu_timestamp))
        imu_msg.header.frame_id = 'imu_link'
        yaw_degrees = (yaw_degrees + 360) % 360
        yaw_rad = math.radians(yaw_degrees)
        # print(yaw_rad,"yawww")
        quat = quaternion_from_euler(0, 0, yaw_rad)  # Roll=0, Pitch=0, Yaw=yaw_rad
        orientation_quat = Quaternion(*quat)
        # Assign the orientation to the Imu message
        imu_msg.orientation = orientation_quat
        self.imu_pub.publish(imu_msg)

    def publish_odom(self, v, omega, odom_timestamp = None):
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.from_sec(float(odom_timestamp))
        odom_msg.header.frame_id = 'odom'  # Local odometry frame
        odom_msg.child_frame_id = 'base_link'
        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = omega
        self.odom_pub.publish(odom_msg)


        # SUBSCRIBER PART OF THE CODE 
    def odom_filtered_callback(self, msg):
        self.latest_state = msg.pose.pose

    def get_latest_data(self):
        if self.latest_state is not None:
            #TODO check whether we need to add or not the start utm
            self.current_utm = (self.latest_state.position.x+self.start_utm[0], self.latest_state.position.y+self.start_utm[1])
            orientation = self.latest_state.orientation
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]
            # print(orientation, "the print of pose from fusion")
            return self.current_utm, yaw
        else:
            return None,None
        
    def calculate_compass_angle(self, x_lsb, y_lsb, avg_x=-177.2872340425532, max_x = 618.2872340425532, avg_y = 905.5425531914893, max_y = 731.4574468085107):
        # Convert LSB Raw Data to Gauss
# why dont i make this tune in itself, like why not get the averagge value and do it ourselves

        x_gauss = (x_lsb - avg_x) / max_x  # Adjust scaling factor as needed
        y_gauss = (y_lsb - avg_y) / max_y

        east = x_gauss
        north = y_gauss

        # Calculate heading using atan2(north, east)
        heading = math.atan2(north, east) * 180 / math.pi

        # Normalize heading to [0, 360) degrees
        heading_deg = (heading + 360) % 360

        return heading_deg
        
    def parse_and_publish(self,data, tracker):
        if data != None:
            cuurent_time = data["timestamp"]
            if cuurent_time != self.previous_time:
                rpm_data = data['rpms'][-1][0:4]
                gps_lat_avg =data["latitude"]
                gps_lon_avg = data["longitude"]
                mags = data["mags"][-1]
                orientation = self.calculate_compass_angle(mags[1], mags[2])
                # print(orientation, data["orientation"],"this is the orientation")
                linear_velocity,angular_velocity = compute_velocity_from_rpms(rpm_data)
                current_gps = (gps_lat_avg, gps_lon_avg)
                utm_x, utm_y = tracker.gps_to_utm(current_gps[0],current_gps[1])
                self.publish_data(linear_velocity,angular_velocity,orientation, utm_x,utm_y, timestamps = [data["timestamp"],data['rpms'][-1][4], data["mags"][0][3]] )
                return (gps_lat_avg,gps_lon_avg), orientation, linear_velocity, angular_velocity
            else:
                return (None,None),None,None, None
        else:
            return (None,None),None,None, None