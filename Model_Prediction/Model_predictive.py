import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
import math
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from tf.transformations import euler_from_quaternion
from cvxpy_mpc import MPC, VehicleModel
import utils
import numpy as np

class Model_Predictive():
    def __init__(self):
        rospy.init_node("Bicycle_Model_Predictive")
        self.target_pose_sub = rospy.Subscriber('/target_pose', Pose, self.target_pose_callback)
        self.target_velocity_sub = rospy.Subscriber('/target_velocity', Twist, self.target_velocity_callback)
        self.target_angular_sub = rospy.Subscriber('/target_angular', Float64, self.target_angular_callback)
        self.model_predicted_num = 5
        self.dt = 0.1
        self.target_x = 0
        self.target_y = 0
        self.target_velocity = 0
        self.target_angular_z = 0
        self.control = np.zeros(2)
        self.state=[0]*4
        rate = rospy.Rate(1000)
        file_path = '~/Desktop/car-racing/local_coordinates_tags.csv'
        df = pd.read_csv(file_path)
        self.cx = list(df['local_x'])
        self.cy = list(df['local_y'])
        self.path=utils.compute_path_from_wp(self.cx,self.cy,0.05)
        print(self.path)
        self.model_prediction_x = []
        self.model_prediction_y = []
        self.Prediction_policy='heuristic'
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'bo-')
        self.ax.set_xlim(min(self.cx) - 1, max(self.cx) + 1)
        self.ax.set_ylim(min(self.cy) - 1, max(self.cy) + 1)
        self.track_line, = self.ax.plot(self.cx, self.cy, 'r--')
        self.animation = FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, blit=True)
        while not rospy.is_shutdown():
            plt.pause(0.001)
        
            rate.sleep()
    def init_plot(self):
        self.line.set_data([], [])
        self.track_line.set_data(self.cx, self.cy)
        return [self.line, self.track_line]
    def update_plot(self, frame):
        self.line.set_data(self.model_prediction_x, self.model_prediction_y)
        return [self.line, self.track_line]

    def target_pose_callback(self, data):
        self.target_x = data.position.x
        self.target_y = data.position.y
        self.target_z = data.position.z
        self.target_orientation_x=data.orientation.x 
        self.target_orientation_y=data.orientation.y 
        self.target_orientation_z=data.orientation.z 
        self.target_orientation_w=data.orientation.w 
        self.target_orientation = [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]
        if self.Prediction_policy=='heuristic':
           self.heuristic_model_predictive()# Signal to update plot
        if self.Prediction_policy=='MPC':
           self.MPC_model_predictive()
           

    def target_velocity_callback(self, data):
        self.target_velocity_x = data.linear.x
        self.target_velocity_y = data.linear.y
        self.target_velocity=((self.target_velocity_x)**2+(self.target_velocity_y)**2)**(1/2)
        self.target_yaw = math.atan2(self.target_velocity_y, self.target_velocity_x)
        print(self.target_yaw)

    def target_angular_callback(self, data):
        self.target_angular_z = data.data

    def get_yaw_from_orientation(self, x, y, z, w):
        euler = euler_from_quaternion([x, y, z, w])
        return euler[2] 

    def heuristic_model_predictive(self):
        model_prediction_x = self.target_x
        target_velocity_x = self.target_velocity_x
        target_yaw = self.target_yaw
        model_prediction_y = self.target_y
        target_velocity_y = self.target_velocity_y
        target_angular_z = self.target_angular_z
        self.model_prediction_x = []
        self.model_prediction_y = []
        for i in range(self.model_predicted_num):
            self.model_prediction_x.append(model_prediction_x)
            model_prediction_x += target_velocity_x
            target_yaw += target_angular_z
            target_velocity_x = self.target_velocity * math.cos(target_yaw)
        for j in range(self.model_predicted_num):
            self.model_prediction_y.append(model_prediction_y)
            model_prediction_y += target_velocity_y
            target_yaw += target_angular_z
            target_velocity_y = self.target_velocity * math.sin(target_yaw)

    def MPC_model_predictive(self):
        T=1 #Prediction Horizon in Time
        L=0.3 #Vehicle_Wheelbase
        Q = [20, 20, 10, 20]  # state error cost [x,y,v,yaw]
        Qf = [30, 30, 30, 30]  # state error cost at final timestep [x,y,v,yaw]
        R = [10, 10]  # input cost [acc ,steer]
        P = [10, 10]  # input rate of change cost [acc ,steer]
        mpc = MPC(VehicleModel(), T, self.dt, Q, Qf, R, P)
        self.model_prediction_x = []
        self.model_prediction_y = []
        self.state[0]=self.target_x
        self.state[1]=self.target_y
        self.state[2]=self.target_velocity
        self.state[3]=self.target_yaw
        target = utils.get_ref_trajectory(self.state, self.path, self.target_velocity, T, self.dt)
        ego_state = np.array([0.0, 0.0, self.state[2], 0.0])
        ego_state[0] = ego_state[0] + ego_state[2] * np.cos(ego_state[3]) * self.dt
        ego_state[1] = ego_state[1] + ego_state[2] * np.sin(ego_state[3]) * self.dt
        ego_state[2] = ego_state[2] + self.control[0] * self.dt
        ego_state[3] = ego_state[3] + self.control[0] * np.tan(self.control[1]) / L * self.dt
        x_mpc, u_mpc = mpc.step(ego_state, target, self.control, verbose=False)
        self.control[0] = u_mpc.value[0, 0]
        self.control[1] = u_mpc.value[1, 0]
        for i in range(self.model_predicted_num):
            self.model_prediction_x.append(x_mpc[0,i])
            self.model_prediction_y.append(x_mpc[1,i])


        







if __name__ == '__main__':
    try:
        Model_Predictive()
    except rospy.ROSInterruptException:
        pass
