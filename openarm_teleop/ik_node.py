#!/usr/bin/env python3
"""
Quest2ROS2 → pyroki IK → OpenArm JointTrajectory

Relative tracking with safety features:
- Joint velocity clamping
- Workspace radius limit
- Exponential smoothing
"""

import threading
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray, Bool, Float32
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient

from teleop_xr.ik_utils import ensure_ik_dependencies

ensure_ik_dependencies()

import jax.numpy as jnp
import jaxlie

from teleop_xr.ik.loader import load_robot_class
from teleop_xr.ik.solver import PyrokiSolver


def pose_msg_to_se3(msg: PoseStamped) -> jaxlie.SE3:
    p = msg.pose.position
    o = msg.pose.orientation
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=jnp.array([o.w, o.x, o.y, o.z])),
        translation=jnp.array([p.x, p.y, p.z]),
    )


class QuestTeleopIKNode(Node):
    def __init__(self):
        super().__init__("quest_teleop_ik")

        # --- Parameters ---
        self.declare_parameter("bimanual", False)
        self.declare_parameter("use_forward_controller", True)
        self.declare_parameter("smoothing", 0.05)           # 低いほどゆっくり
        self.declare_parameter("speed", 0.5)                 # Quest 移動量のスケール
        self.declare_parameter("max_joint_step", 0.02)       # 1ステップの最大関節変化 [rad] (50Hz → 1.0 rad/s)
        self.declare_parameter("max_workspace_radius", 0.3)  # Quest 移動量の最大半径 [m]

        self.bimanual = self.get_parameter("bimanual").value
        self.use_forward = self.get_parameter("use_forward_controller").value
        self.smoothing = self.get_parameter("smoothing").value
        self.speed = self.get_parameter("speed").value
        self.max_joint_step = self.get_parameter("max_joint_step").value
        self.max_workspace_radius = self.get_parameter("max_workspace_radius").value

        # --- Robot & IK setup ---
        self.get_logger().info("Initializing OpenArmRobot + PyrokiSolver (JIT warmup)...")
        robot_cls = load_robot_class("teleop_xr.ik.robots.openarm:OpenArmRobot")
        self.robot = robot_cls()
        self.solver = PyrokiSolver(self.robot)
        self.get_logger().info("IK solver ready.")

        self.q_current = jnp.array(self.robot.get_default_config())
        self.actuated_names = self.robot.actuated_joint_names
        self.get_logger().info(f"Actuated joints: {self.actuated_names}")

        # Separate arm joint names
        self.right_arm_names = [n for n in self.actuated_names if "right" in n and "finger" not in n]
        self.right_arm_indices = [self.actuated_names.index(n) for n in self.right_arm_names]

        # --- Relative tracking state ---
        self.right_init_pose: jaxlie.SE3 | None = None
        self.right_init_fk: jaxlie.SE3 | None = None
        self.tracking_active = False

        # --- Deadman switch (grip button) ---
        self.grip_pressed = False
        self.grip_was_pressed = False  # previous state for edge detection
        self.grip_ever_received = False  # True after first grip message arrives

        # --- Gripper (trigger button) ---
        self.trigger_value = 0.0  # 0.0 = released, 1.0 = fully pulled
        self.trigger_ever_received = False
        self.gripper_max = 0.044  # max opening [m] from URDF
        self.gripper_pos_cmd = 0.0  # current smoothed gripper position (start closed)
        self.last_gripper_sent = -1.0  # track last sent value to avoid spamming
        self.gripper_smoothing = 0.1  # smoothing rate for gripper

        # --- State ---
        self.left_pose: PoseStamped | None = None
        self.right_pose: PoseStamped | None = None
        self.lock = threading.Lock()
        self.publish_count = 0
        self.cmd_q = None  # smoothed command (numpy array)

        # --- Subscribers ---
        self.create_subscription(PoseStamped, "/q2r_left_hand_pose", self._left_pose_cb, 10)
        self.create_subscription(PoseStamped, "/q2r_right_hand_pose", self._right_pose_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.create_subscription(Bool, "/q2r_right_grip_pressed", self._grip_cb, 10)
        self.create_subscription(Float32, "/q2r_right_trigger_value", self._trigger_cb, 10)

        # --- Gripper action client ---
        self.gripper_client = ActionClient(
            self, GripperCommand, "/gripper_controller/gripper_cmd"
        )

        # --- Publishers ---
        if self.use_forward:
            self.pub_forward = self.create_publisher(
                Float64MultiArray, "/forward_position_controller/commands", 10
            )
            self.get_logger().info("Mode: forward_position_controller")
        else:
            self.pub_single = self.create_publisher(
                JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
            )
            self.get_logger().info("Mode: joint_trajectory_controller")

        # --- Control loop at 50 Hz ---
        self.timer = self.create_timer(0.02, self._control_loop)
        self.get_logger().info(
            f"Started (50Hz, smoothing={self.smoothing}, speed={self.speed}, "
            f"max_joint_step={self.max_joint_step} rad, max_workspace={self.max_workspace_radius} m)"
        )

    # ------------------------------------------------------------------ callbacks
    def _left_pose_cb(self, msg: PoseStamped):
        with self.lock:
            self.left_pose = msg

    def _right_pose_cb(self, msg: PoseStamped):
        with self.lock:
            self.right_pose = msg

    def _trigger_cb(self, msg: Float32):
        with self.lock:
            self.trigger_value = msg.data
            self.trigger_ever_received = True

    def _grip_cb(self, msg: Bool):
        with self.lock:
            if not self.grip_ever_received:
                self.grip_ever_received = True
                self.get_logger().info("Grip topic received - deadman switch active")
            old = self.grip_pressed
            self.grip_pressed = msg.data
            if old != msg.data:
                self.get_logger().info(f"Grip: {'PRESSED' if msg.data else 'RELEASED'}")

    def _joint_state_cb(self, msg: JointState):
        q = np.array(self.q_current)
        for i, name in enumerate(self.actuated_names):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]
            elif not self.bimanual and "right" in name:
                single_name = name.replace("_right", "")
                if single_name in msg.name:
                    idx = msg.name.index(single_name)
                    q[i] = msg.position[idx]
        self.q_current = jnp.array(q)

    # ------------------------------------------------------------------ IK loop
    def _control_loop(self):
        with self.lock:
            rp = self.right_pose
            grip = self.grip_pressed

        if rp is None:
            return

        # Block until grip topic is available (Unity must send grip state)
        if not self.grip_ever_received:
            if self.publish_count == 0:
                self.get_logger().info("Waiting for /q2r_right_grip_pressed topic... (robot will not move until grip is received)")
                self.publish_count = -1  # log once
            return

        # Deadman switch: grip not pressed → hold position, reset origin on next press
        if not grip:
            if self.grip_was_pressed:
                # Grip just released → mark tracking for re-init on next press
                self.tracking_active = False
                self.get_logger().info("Grip released - robot stopped (hold position)")
            self.grip_was_pressed = False
            return

        quest_R = pose_msg_to_se3(rp)

        # Initialize/re-initialize tracking when grip is pressed
        if not self.tracking_active:
            self.right_init_pose = quest_R
            fk = self.robot.forward_kinematics(self.q_current)
            self.right_init_fk = fk.get("right")
            if self.right_init_fk is None:
                return
            if self.cmd_q is None:
                self.cmd_q = np.array(self.q_current)
            self.tracking_active = True
            self.grip_was_pressed = True
            self.get_logger().info("Grip pressed - tracking started (origin reset)")
            return

        self.grip_was_pressed = True

        # Compute relative delta from initial Quest pose
        delta = self.right_init_pose.inverse() @ quest_R
        delta_translation = np.array(delta.translation()) * self.speed
        # Mirror Y axis: operator stands behind the robot, so left/right is flipped
        delta_translation[1] = -delta_translation[1]

        # Safety: clamp workspace radius
        dist = np.linalg.norm(delta_translation)
        if dist > self.max_workspace_radius:
            delta_translation = delta_translation * (self.max_workspace_radius / dist)

        # Mirror rotation to match Y-axis flip (negate qy and qz)
        delta_rot_wxyz = delta.rotation().wxyz
        delta_rotation = jaxlie.SO3(wxyz=jnp.array([
            delta_rot_wxyz[0], delta_rot_wxyz[1],
            -delta_rot_wxyz[2], -delta_rot_wxyz[3]
        ]))
        scaled_delta = jaxlie.SE3.from_rotation_and_translation(
            delta_rotation, jnp.array(delta_translation)
        )
        target_R = self.right_init_fk @ scaled_delta

        # Solve IK
        try:
            new_q = self.solver.solve(None, target_R, None, self.q_current)
        except Exception as e:
            self.get_logger().error(f"IK solve error: {e}")
            return

        # Smooth: blend toward IK target
        new_q_np = np.array(new_q)
        desired = self.cmd_q + self.smoothing * (new_q_np - self.cmd_q)

        # Safety: clamp joint velocity (max change per step)
        delta_q = desired - self.cmd_q
        delta_q = np.clip(delta_q, -self.max_joint_step, self.max_joint_step)
        self.cmd_q = self.cmd_q + delta_q

        # Publish
        if self.use_forward:
            msg = Float64MultiArray()
            msg.data = [float(self.cmd_q[i]) for i in self.right_arm_indices]
            self.pub_forward.publish(msg)
        else:
            stamp = self.get_clock().now().to_msg()
            msg = JointTrajectory()
            msg.header.stamp = stamp
            msg.joint_names = [
                "openarm_joint1", "openarm_joint2", "openarm_joint3",
                "openarm_joint4", "openarm_joint5", "openarm_joint6",
                "openarm_joint7",
            ]
            pt = JointTrajectoryPoint()
            pt.positions = [float(self.cmd_q[i]) for i in self.right_arm_indices]
            pt.time_from_start = Duration(sec=0, nanosec=int(1e7))
            msg.points = [pt]
            self.pub_single.publish(msg)

        self.publish_count += 1
        if self.publish_count % 250 == 1:
            q_list = [f"{float(self.cmd_q[i]):.3f}" for i in self.right_arm_indices]
            self.get_logger().info(f"#{self.publish_count} joints=[{', '.join(q_list)}]")

        # Gripper: trigger pulled = open, released = closed
        with self.lock:
            trigger = self.trigger_value
            trigger_ready = self.trigger_ever_received
        if trigger_ready:
            target_pos = self.gripper_max * trigger  # 0.0 = closed, 1.0 trigger → full open
            # Smooth gripper movement
            self.gripper_pos_cmd += self.gripper_smoothing * (target_pos - self.gripper_pos_cmd)
            # Only send if changed enough to avoid spamming
            if abs(self.gripper_pos_cmd - self.last_gripper_sent) > 0.0005:
                self.last_gripper_sent = self.gripper_pos_cmd
                goal = GripperCommand.Goal()
                goal.command.position = self.gripper_pos_cmd
                goal.command.max_effort = 100.0
                self.gripper_client.send_goal_async(goal)


def main(args=None):
    rclpy.init(args=args)
    node = QuestTeleopIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
