from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, PoseStamped
from sensor_msgs.msg import LaserScan
from tf_transformations import quaternion_matrix, quaternion_from_euler, euler_from_quaternion
from rclpy.node import Node

import rclpy
import numpy as np
assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)
        self.particles_pub = self.create_publisher(PoseArray, '/visualize_particles', 1)
        self.estimated_robot_pub = self.create_publisher(PoseStamped, '/estimated_robot', 1)
        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.motion_model.deterministic = False
        self.sensor_model = SensorModel(self)

        self.num_particles = 100
        self.particles = np.zeros((self.num_particles, 3))
        self.particle_probabilities = np.empty((self.num_particles,))

        self.initialized = False
        self.get_logger().info("=============+READY+=============")

        self.prev_time = self.get_clock().now().nanoseconds*1e-9

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
    def create_odom_msg(self, pose):
        t= Odometry()
         # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        # CHANGE WHEN OUT OF SIM
        t.child_frame_id = "/pf/pose/odom"
        t.pose.pose.position.x = pose[0]
        t.pose.pose.position.y = pose[1]
        t.pose.pose.position.z = 0.

        q = quaternion_from_euler(0,0,pose[2])
        t.pose.pose.orientation.x = q[0]
        t.pose.pose.orientation.y = q[1]
        t.pose.pose.orientation.z = q[2]
        t.pose.pose.orientation.w = q[3]

        return t

    def publish_particles(self):
        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for particle in self.particles:
            particle_pose = Pose()
            particle_pose.position.x = particle[0]
            particle_pose.position.y = particle[1]
            particle_pose.position.z = 0.

            q = quaternion_from_euler(0.,0.,particle[2])
            particle_pose.orientation.x = q[0]
            particle_pose.orientation.y = q[1]
            particle_pose.orientation.z = q[2]
            particle_pose.orientation.w = q[3]

            msg.poses.append(particle_pose)
        self.particles_pub.publish(msg)
    def publish_robot_pose(self):
        avg_theta = np.arctan2(np.mean(np.sin(self.motion_model.updated_particles_pose[:,2])), np.mean(np.cos(self.motion_model.updated_particles_pose[:, 2])))%(2*np.pi)
        odom_msg = self.create_odom_msg(np.array([np.mean(self.motion_model.updated_particles_pose[:,0]), np.mean(self.motion_model.updated_particles_pose[:,1]), avg_theta]))
        self.odom_pub.publish(odom_msg)

        est_robot_msg = PoseStamped()
        est_robot_msg.pose = odom_msg.pose.pose
        est_robot_msg.header.stamp = self.get_clock().now().to_msg()
        self.estimated_robot_pub.publish(est_robot_msg)
    
    # def stratified_resample(self, weights):
    #     N = len(weights)
    #     # make N subdivisions, chose a random position within each one
    #     positions = (np.random.uniform(0, 1, N) + range(N)) / N
    #     self.get_logger().info(f"{positions}")

    #     indexes = np.zeros(N, 'i')
    #     cumulative_sum = np.cumsum(weights)
    #     i, j = 0, 0
    #     while i < N:
    #         if positions[i] < cumulative_sum[j]:
    #             indexes[i] = j
    #             i += 1
    #             self.get_logger().info(f"{j}")
    #         else:
    #             j += 1
    #     # self.get_logger().info(f"{indexes}")
    #     return indexes
    # def multinomal_resample(self, weights):
    #     N = len(weights)
    #     avg = np.mean(weights)
    #     indexes = np.zeros(N, 'i')

    #     # take int(N*w) copies of each weight, which ensures particles with the
    #     # same weight are drawn uniformly
    #     num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    #     k = 0
    #     for i in range(N):
    #         for _ in range(num_copies[i]): # make n copies
    #             indexes[k] = i
    #             k += 1

    #     # use multinormal resample on the residual to fill up the rest. This
    #     # maximizes the variance of the samples
    #     residual = weights - num_copies     # get fractional part
    #     residual /= sum(residual)           # normalize
    #     cumulative_sum = np.cumsum(residual) 
    #     indexes[k:N] = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, N-k)/avg)

    #     return indexes 

    def gpt_resample(self, probabilities):
        # I give up
         # Normalize the probabilities to ensure they sum to 1
        norm_probabilities = probabilities / np.sum(probabilities)
        
        # Create a cumulative distribution function (CDF)
        cdf = np.cumsum(norm_probabilities)
        
        # Generate random numbers to stratify resampling
        random_numbers = np.random.rand(self.num_particles)
        
        # Use the CDF to determine which intervals the random numbers fall into
        resampled_indices = np.searchsorted(cdf, random_numbers)
        
        return resampled_indices
    def laser_callback(self, msg):
        # evaluate sensor model here
        # downsampling in the sensor model for now
        if self.initialized and self.sensor_model.map_set:
            self.particle_probabilities = self.sensor_model.evaluate(self.particles, np.array(msg.ranges))
            # self.get_logger().info(f"{self.particle_probabilities}")

            # resample particles
            # NOT WORKING PLEASE HELP AHHHHHHHHHHHHHHH
            resampled_indices = self.gpt_resample(self.particle_probabilities)
            self.particles[:] = self.particles[resampled_indices]

            # update average particle pose (in theory the robot pose)
            self.publish_robot_pose()
            self.publish_particles()

    def pose_callback(self, msg):
        # initialize pose of particles (and therefore the robot pose)
        
        theta= euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[-1]
        
        self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, theta])
        self.init_randomness = np.random.normal(0, 0.2, (self.num_particles, 3))

        self.particles = self.robot_pose + self.init_randomness

        self.get_logger().info(f"{self.particles}")
        self.odom_pub.publish(self.create_odom_msg(self.robot_pose))
        self.initialized = True
    
    def odom_callback(self, msg):
        # odom we want dx, dy, dtheta
        dt = self.get_clock().now().nanoseconds*1e-9 - self.prev_time
        odom = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])*dt

        self.prev_time = self.get_clock().now().nanoseconds*1e-9

        # update motion model
        self.particles=self.motion_model.evaluate(self.particles, odom)

        # update average particle pose (in theory the robot pose)
        # other thoughts on averaging - only take particles with probability higher than threshold??
        self.get_logger().info(f"{self.particles}")

        self.publish_robot_pose()
        self.publish_particles()
        



def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
