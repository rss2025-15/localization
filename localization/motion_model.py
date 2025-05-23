import numpy as np

class MotionModel:

    def __init__(self, node, n):
        ####################################
        # Do any precomputation for the motion
        # model here.

        self.deterministic = node.get_parameter("in_sim").get_parameter_value().integer_value
        self.odom = None
        self.IN_SIM = node.get_parameter("in_sim").get_parameter_value().integer_value
        # self.updated_particles_pose=None
        # self.particles_T = None
        # self.odom_T = None
        # self.randomness = None

        self.updated_particles_pose = np.empty((n, 3))
        self.particles_T = np.empty((n,3, 3))
        self.odom_T = np.empty((n, 3, 3))
        # self.randomness = np.zeros((n,3,3))

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]. is in robot frame?

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        def generate_T_stack(poses, T_stack):
            # poses in nx3 array
            x = poses[:,0]
            y = poses[:, 1]
            theta = poses[:,2]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            np.stack([np.column_stack([cos_theta, -sin_theta, x]), 
                                        np.column_stack([sin_theta, cos_theta, y]),
                                        np.array([[0,0,1]]*n)], axis=1, out=T_stack)
            return T_stack


        n, _ = particles.shape          

        # self.odom = np.array([odometry]*n)
        self.odom = odometry
        # if not self.deterministic:
        #     randomness =np.random.normal(0, np.array([self.x_noise, self.y_noise, self.theta_noise]), (n, 3))
        #     self.odom += randomness
        
        self.odom_T =generate_T_stack(self.odom, self.odom_T)
        self.particles_T = generate_T_stack(particles, self.particles_T)
        self.particles_T = np.matmul(self.particles_T, self.odom_T)
        self.updated_particles_pose[:, 0] = self.particles_T[:, 0,2]
        self.updated_particles_pose[:, 1] = self.particles_T[:, 1,2]
        self.updated_particles_pose[:,2] = np.arctan2(self.particles_T[:,1,0], self.particles_T[:,0,0])

        return self.updated_particles_pose
    

        ####################################
