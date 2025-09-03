#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Jaeyoung Lim"
__contact__ = "jalim@ethz.ch"

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleRatesSetpoint
from px4_msgs.msg import VehicleAttitudeSetpoint


class OffboardControl(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        # Declare and retrieve the namespace parameter
        self.declare_parameter('namespace', '')  # Default to empty namespace
        self.namespace = self.get_parameter('namespace').value
        self.namespace_prefix = f'/{self.namespace}' if self.namespace else ''
        
                # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile_sub)
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.create_publisher(VehicleRatesSetpoint, 'fmu/in/vehicle_rates_setpoint', qos_profile_pub)
        self.publisher_acceleration = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_profile_pub)
        self.publisher_attitude = self.create_publisher(VehicleAttitudeSetpoint, 'fmu/in/vehicle_attitude_setpoint_v1', qos_profile_pub)
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.dt = timer_period
        self.declare_parameter('radius', 10.0)
        self.declare_parameter('omega', 5.0)
        self.declare_parameter('altitude', 5.0)

        self.kf = 1
        self.km = 1
        self.lin_cof_a = 1.0
        self.lin_int_b = -0.35
        self.mass = 2.06

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        #self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        # Note: no parameter callbacks are used to prevent sudden inflight changes of radii and omega 
        # which would result in large discontinuities in setpoints
        self.theta = 0.0
        self.radius = self.get_parameter('radius').value
        self.omega = self.get_parameter('omega').value
        self.altitude = self.get_parameter('altitude').value
 
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def force_to_throttle_linear(self, force):
        g = 9.81
        T_hover = -0.728
        F_hover = np.sqrt((self.mass * g) / 4)
        a = T_hover / F_hover
        b = 0.0
        return a * force + b
    
    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position=False
        offboard_msg.velocity = False
        offboard_msg.acceleration=False
        offboard_msg.attitude = True
        offboard_msg.body_rate = False

        self.publisher_offboard_mode.publish(offboard_msg)

        # --- 1. Extract ENU angular velocity ---
        ang_vel_enu = np.array([
            0.0,
            0.0,
            0.0])

        # --- 2. Transform ENU -> NED (aircraft body frame) ---
        # This assumes ENU to FRD (body frame) transformation is: [x, y, z] -> [y, x, -z]
        # Or use a proper transformation if px4_ros_com.frame_transforms exists in Python
        ang_vel_ned = np.array([
            ang_vel_enu[1],   # roll
            ang_vel_enu[0],   # pitch
            -ang_vel_enu[2]   # yaw
        ])

        ## --- 3. Compute throttle ---

        ##thrust = (4.686*9.81)
        thrust = np.sqrt((3.2*9.81)/4)

        ## --- 4. Scale and clamp throttle ---
        ## self.lin_cof_a, self.lin_int_b are linear fit parameters
        #
        throttle = self.force_to_throttle_linear(thrust)
        #self.get_logger().info(f"Throttle: {throttle:.3f}")
        ##throttle = np.clip(throttle, 0.0, 1.0)

        thrust_body = [0.0, 0.0, throttle] 
        qd = [1.0, 0.0, 0.0, 0.0]

        ## --- 5. Create and publish VehicleRatesSetpoint ---
        rates_msg = VehicleRatesSetpoint()
        now = self.get_clock().now()
        rates_msg.timestamp = int(now.nanoseconds / 1000)  # microseconds
        rates_msg.roll = float(ang_vel_ned[0])
        rates_msg.pitch = float(ang_vel_ned[1])
        rates_msg.yaw = float(0.1)
        rates_msg.thrust_body = thrust_body

        self.publisher_trajectory.publish(rates_msg)

        
        ## Acceleration set point
        #trajectory_msg = TrajectorySetpoint()
        #trajectory_msg.velocity[0] = 0.0
        #trajectory_msg.velocity[1] = 0.0
        #trajectory_msg.velocity[2] = 0.0

        #trajectory_msg.acceleration[0] = 1.0*9.8
        #trajectory_msg.acceleration[1] = 0.0
        #trajectory_msg.acceleration[2] = -1.65*9.81
        #trajectory_msg.yaw = 0.0
        #self.publisher_acceleration.publish(trajectory_msg)

        
        #attitude_msg = VehicleAttitudeSetpoint()
        #now = self.get_clock().now()
        #attitude_msg.timestamp = int(now.nanoseconds / 1000)  # microseconds
        #attitude_msg.thrust_body = thrust_body
        #attitude_msg.q_d = qd
        #self.publisher_attitude.publish(attitude_msg)


        


def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
