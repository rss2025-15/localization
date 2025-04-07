from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, PoseStamped
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import quaternion_matrix, quaternion_from_euler, euler_from_quaternion
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from geometry_msgs.msg import TransformStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray

import rclpy
import numpy as np
assert rclpy
import math


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        self.declare_parameter('init_xy_noise', 0.0)
        self.init_xy_noise = self.get_parameter('init_xy_noise').get_parameter_value().double_value

        self.declare_parameter('init_theta_noise', 0.0)
        self.init_theta_noise = self.get_parameter('init_theta_noise').get_parameter_value().double_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('drive_topic', "/drive")
        self.declare_parameter('in_sim', 1)
        self.declare_parameter('is_rosbag', 0)

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        drive_topic = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.IN_SIM = self.get_parameter("in_sim").get_parameter_value().integer_value
        self.is_rosbag = self.get_parameter("is_rosbag").get_parameter_value().integer_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)
        
        self.drive_sub = self.create_subscription(AckermannDriveStamped, drive_topic,
                                                 self.drive_callback,
                                                 1)
        
        self.drive_msg = None
        self.timer = self.create_timer(0.01, self.drive_timer_callback)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)
        # self.point_sub = self.create_subscription(PointStamped, "/clicked_point", self.point_callback, 1)
        # self.checkpoint_pub = self.create_publisher(MarkerArray, "/checkpoint", 1)
        # self.checkpoints = []
        self.particles_pub = self.create_publisher(PoseArray, '/visualize_particles_rosbag' if self.is_rosbag else '/visualize_particles', 1)
        self.estimated_robot_pub = self.create_publisher(PoseStamped, '/estimated_robot', 1)
        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        self.declare_parameter('num_particles', 100)
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        self.declare_parameter('deterministic', False)
        self.deterministic = self.get_parameter('deterministic').get_parameter_value().bool_value

        self.declare_parameter('car_length', 0.0)
        self.car_length = self.get_parameter('car_length').get_parameter_value().double_value

        self.declare_parameter('use_imu', True)
        self.use_imu = self.get_parameter('use_imu').get_parameter_value().bool_value
        self.declare_parameter('uniform_noise', False)
        self.uniform_noise = self.get_parameter('uniform_noise').get_parameter_value().bool_value
        # imu noise params
        self.declare_parameter('imu_vx_noise', 0.0)
        self.imu_vx_noise = self.get_parameter('imu_vx_noise').get_parameter_value().double_value
        self.declare_parameter('imu_vy_noise', 0.0)
        self.imu_vy_noise = self.get_parameter('imu_vy_noise').get_parameter_value().double_value
        self.declare_parameter('imu_omega_noise', 0.0)
        self.imu_omega_noise = self.get_parameter('imu_omega_noise').get_parameter_value().double_value
        # drive command noise params
        self.declare_parameter('drive_vel_noise', 0.0)
        self.drive_vel_noise = self.get_parameter('drive_vel_noise').get_parameter_value().double_value
        self.declare_parameter('drive_steer_noise', 0.0)
        self.drive_steer_noise = self.get_parameter('drive_steer_noise').get_parameter_value().double_value
        # resulting pose noise params
        self.declare_parameter('x_noise', 0.0)
        self.x_noise = self.get_parameter('x_noise').get_parameter_value().double_value
        self.declare_parameter('y_noise', 0.0)
        self.y_noise = self.get_parameter('y_noise').get_parameter_value().double_value
        self.declare_parameter('theta_noise', 0.0)
        self.theta_noise = self.get_parameter('theta_noise').get_parameter_value().double_value        

        # Initialize the models
        self.motion_model = MotionModel(self, self.num_particles)
        self.motion_model.deterministic = self.deterministic
        self.sensor_model = SensorModel(self)

        self.particles = np.zeros((self.num_particles, 3))
        self.particle_probabilities = np.empty((self.num_particles,))

        self.initialized = False
        self.get_logger().info("=============+READY+=============")

        self.prev_time = self.get_clock().now().nanoseconds*1e-9

        # "latest" ground truth, for sim
        # self.ground_truth_pose = np.empty(3)

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        # map->base_link tf broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.buffer = Buffer()
        self.tf_listener = TransformListener(self.buffer, self)
    
    # def point_callback(self, msg):
    #     self.checkpoints.append((msg.point.x, msg.point.y))
    #     marker_arr = MarkerArray()
    #     for i,x,y in enumerate(self.checkpoints):
    #         marker = Marker()

    #         marker.header.frame_id = "/map"
    #         marker.header.stamp = self.get_clock().now().to_msg()

    #         marker.id = 0

    #         marker.type = Marker.SPHERE

    #         marker.action = Marker.ADD

    #         marker.pose.position.x = x
    #         marker.pose.position.y = y
    #         marker.pose.position.z = 0
    #         marker.pose.orientation.x = 0.0
    #         marker.pose.orientation.y = 0.0
    #         marker.pose.orientation.z = 0.0
    #         marker.pose.orientation.w = 1.0

    #         marker.scale.x = 1.0
    #         marker.scale.y = 1.0
    #         marker.scale.z = 1.0

    #         marker.color.r = 0.0
    #         marker.color.g = 0.0
    #         marker.color.b = 1.0
    #         marker.color.a = 1.0

    #         marker_arr.markers.append(marker)
    #     self.checkpoint_pub.publish(marker_arr)
        
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
        odom_array = np.array([np.mean(self.motion_model.updated_particles_pose[:,0]), np.mean(self.motion_model.updated_particles_pose[:,1]), avg_theta])
        odom_msg = self.create_odom_msg(odom_array)
        self.odom_pub.publish(odom_msg)

        est_robot_msg = PoseStamped()
        est_robot_msg.pose = odom_msg.pose.pose
        est_robot_msg.header.frame_id = 'map'
        est_robot_msg.header.stamp = self.get_clock().now().to_msg()
        self.estimated_robot_pub.publish(est_robot_msg)

        # ground truthing
        if self.IN_SIM and self.initialized:
            truth = self.buffer.lookup_transform('map', 'base_link', rclpy.time.Time(), rclpy.duration.Duration(seconds = 2))
            truth_orientation = euler_from_quaternion([truth.transform.rotation.x, truth.transform.rotation.y, truth.transform.rotation.z, truth.transform.rotation.w])[-1]
            truth_pose = np.array([truth.transform.translation.x, truth.transform.translation.y, truth_orientation])
            error = truth_pose - odom_array
            error[2] = min(abs(error[2]), abs(2*np.pi - abs(error[2]))) # angle error is absolute valued
            self.get_logger().info(f"error {error}")
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = self.particle_filter_frame

        t.transform.translation.x = odom_array[0]
        t.transform.translation.y = odom_array[1]
        q = quaternion_from_euler(0,0,odom_array[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)
        
        

    def multinomial_resample(self, probabilities):
        # sample each particle proportional to its weight
        # high weight = more likely
        # cluster collapses, but no diversity
        norm_probabilities = probabilities / np.sum(probabilities)
        cdf = np.cumsum(norm_probabilities)
        random_numbers = np.random.rand(self.num_particles)
        resampled_indices = np.searchsorted(cdf, random_numbers)
        
        return resampled_indices
    
    def stratified_resample(self, probabilities):
        norm_probabilities = probabilities / np.sum(probabilities)
        positions = (np.random.random(self.num_particles) + range(self.num_particles)) / self.num_particles

        indexes = np.zeros(self.num_particles, 'i')
        cumulative_sum = np.cumsum(norm_probabilities)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes
    
    def residual_resample(self, probabilities):
        norm_probabilities = probabilities / np.sum(probabilities)
        indexes = np.zeros(self.num_particles, 'i')

        # take int(N*w) copies of each weight
        num_copies = (self.num_particles*np.asarray(norm_probabilities)).astype(int)
        k = 0
        for i in range(self.num_particles):
            for _ in range(num_copies[i]): # make n copies
                indexes[k] = i
                k += 1

        # use multinormial resample on the residual to fill up the rest.
        residual = norm_probabilities - num_copies     # get fractional part
        residual /= sum(residual)     # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1. # ensures sum is exactly one
        indexes[k:self.num_particles] = np.searchsorted(cumulative_sum, np.random.random(self.num_particles-k))

        return indexes
    
    def systematic_resample(self, probabilities):
        # make N subdivisions, choose positions 
        # with a consistent random offset
        norm_probabilities = probabilities / np.sum(probabilities)
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles

        indexes = np.zeros(self.num_particles, 'i')
        cumulative_sum = np.cumsum(norm_probabilities)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes
    def laser_callback(self, msg):
        # evaluate sensor model here
        # downsampling in the sensor model for now
        if self.initialized and self.sensor_model.map_set:
            self.particle_probabilities = self.sensor_model.evaluate(self.particles, np.array(msg.ranges))
            # self.get_logger().info(f"{self.particle_probabilities}")

            # resample particles
            # NOT WORKING PLEASE HELP AHHHHHHHHHHHHHHH
            resampled_indices = self.systematic_resample(self.particle_probabilities)
            self.particles[:] = self.particles[resampled_indices]

            # update average particle pose (in theory the robot pose)
            self.publish_robot_pose()
            self.publish_particles()


    def pose_callback(self, msg):
        # initialize pose of particles (and therefore the robot pose)
        
        theta= euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[-1]
        
        self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, theta])
        if not self.uniform_noise:
            self.init_randomness = np.random.normal(0, np.array([self.init_xy_noise, self.init_xy_noise, self.init_theta_noise]), (self.num_particles, 3))
        else:
            self.init_randomness = np.random.uniform(-np.array([self.init_xy_noise, self.init_xy_noise, self.init_theta_noise])/2, np.array([self.init_xy_noise, self.init_xy_noise, self.init_theta_noise])/2, (self.num_particles, 3))
        self.particles = self.robot_pose + self.init_randomness

        self.get_logger().info(f"{self.particles}")
        self.odom_pub.publish(self.create_odom_msg(self.robot_pose))
        self.initialized = True
        self.publish_particles()

    def deterministic_imu(self, vx, vy, omega, dt):
        return np.array([vx, vy, omega])*dt
    
    def deterministic_drive(self, vel, steer, dt):
        # self.get_logger().info(f'vel: {vel}, steer: {steer}')
        stop_dist = vel*dt
        if abs(steer) < 0.001:
            return np.array([stop_dist, 0, 0])
        else:
            turning_radius = self.car_length / math.tan(steer) # positive if turning left, negative if right
            turn_angle = stop_dist / turning_radius # positive if left, negative if right,m -pi/2 to pi/2
            stop_x, stop_y = turning_radius*(math.cos(turn_angle)-1), turning_radius*math.sin(turn_angle)
            return np.array([stop_y, -stop_x, turn_angle])
    
    def odom_callback(self, msg):
        if not self.use_imu:
            return
        # odom we want dx, dy, dtheta
        dt = self.get_clock().now().nanoseconds*1e-9 - self.prev_time
        self.prev_time = self.get_clock().now().nanoseconds*1e-9

        if not self.IN_SIM:
            vx = -msg.twist.twist.linear.x
            vy = -msg.twist.twist.linear.y
            omega = -msg.twist.twist.angular.z
        else:
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            omega = msg.twist.twist.angular.z

        # update motion model
        odom = np.ndarray((self.num_particles,3))
        for i in range(self.num_particles):
            if self.deterministic:
                odom[i,:] = self.deterministic_imu(vx, vy, omega, dt)
            else:
                if not self.uniform_noise:
                    odom[i,:] = self.deterministic_imu(vx + np.random.normal(0, self.imu_vx_noise), vy + np.random.normal(0, self.imu_vy_noise), omega + np.random.normal(0, self.imu_omega_noise), dt) + np.random.normal(0, np.array([self.x_noise, self.y_noise, self.theta_noise]), (3,))
                else:
                    odom[i,:] = self.deterministic_imu(vx + np.random.uniform(-self.imu_vx_noise/2, self.imu_vx_noise/2), vy + np.random.uniform(-self.imu_vy_noise/2, self.imu_vy_noise/2), omega + np.random.uniform(-self.imu_omega_noise/2, self.imu_omega_noise/2), dt) + np.random.uniform(-np.array([self.x_noise, self.y_noise, self.theta_noise])/2, np.array([self.x_noise, self.y_noise, self.theta_noise])/2, (3,))
        self.particles=self.motion_model.evaluate(self.particles, odom)

        # update average particle pose (in theory the robot pose)
        # other thoughts on averaging - only take particles with probability higher than threshold??
        # self.get_logger().info(f"{self.particles}")

        #### ignore ts
        # # update latest ground truth pose
        # theta = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[-1]
        # self.ground_truth_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, theta])

        self.publish_robot_pose()
        self.publish_particles()
        
    def drive_callback(self, msg):
        self.drive_msg = msg
        
    def drive_timer_callback(self):
        # self.get_logger().info('pre odom callback')
        if self.use_imu or (self.drive_msg is None):
            return
        # self.get_logger().info('odom callback')
        # odom we want dx, dy, dtheta
        dt = self.get_clock().now().nanoseconds*1e-9 - self.prev_time
        self.prev_time = self.get_clock().now().nanoseconds*1e-9

        vel = self.drive_msg.drive.speed
        steer = self.drive_msg.drive.steering_angle
        # self.get_logger().info(f'vel: {vel}, steer: {steer}')

        # update motion model
        odom = np.ndarray((self.num_particles,3))
        for i in range(self.num_particles):
            if self.deterministic:
                odom[i,:] = self.deterministic_drive(vel, steer, dt)
            else:
                if not self.uniform_noise:
                    odom[i,:] = self.deterministic_drive(vel + np.random.normal(0, self.drive_vel_noise), steer + np.random.normal(0, self.drive_steer_noise), dt) + np.random.normal(0, np.array([self.x_noise, self.y_noise, self.theta_noise]), (3,))
                else:
                    odom[i,:] = self.deterministic_drive(vel + np.random.uniform(-self.drive_vel_noise/2, self.drive_vel_noise/2), steer + np.random.uniform(-self.drive_steer_noise/2, self.drive_steer_noise/2), dt) + np.random.uniform(-np.array([self.x_noise, self.y_noise, self.theta_noise])/2, np.array([self.x_noise, self.y_noise, self.theta_noise])/2, (3,))
        
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
