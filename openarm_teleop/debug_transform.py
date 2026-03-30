#!/usr/bin/env python3
"""
座標変換デバッグツール v4 (回転対応)

コントローラーの初期向きから「ユーザーの正面」を推定し、
体に対して相対的な前後・左右・上下 + 回転をロボットにマッピング。
config.yamlを編集すると自動で再読み込みされる。
"""

import os
import threading
import math

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from teleop_xr.ik_utils import ensure_ik_dependencies

ensure_ik_dependencies()

import jax.numpy as jnp
import jaxlie

from teleop_xr.ik.loader import load_robot_class
from teleop_xr.ik.solver import PyrokiSolver
from teleop_xr.ik.weights import TeleopConfig


def quat_to_forward_yaw(w, x, y, z):
    """Extract the horizontal forward direction (yaw) from a quaternion.
    Returns a 2D unit vector [fwd_x, fwd_y] in the XR horizontal plane.
    Controller forward = -Z in local frame (WebXR convention).
    """
    fx = 2.0 * (x * z + w * y)
    fy = 2.0 * (y * z - w * x)
    length = math.sqrt(fx * fx + fy * fy)
    if length < 0.01:
        return np.array([1.0, 0.0])
    return np.array([fx / length, fy / length])


def build_frame_rotation_3x3(body_rot_2x2):
    """Build a 3x3 rotation matrix from XR world to body frame.
    body_rot_2x2: 2x2 matrix mapping XR horizontal [x,y] → [body_fwd, body_right]
    Z (up) passes through unchanged.
    """
    return np.array([
        [body_rot_2x2[0, 0], body_rot_2x2[0, 1], 0.0],
        [body_rot_2x2[1, 0], body_rot_2x2[1, 1], 0.0],
        [0.0,                0.0,                 1.0],
    ])


def transform_rotation(
    init_ori_wxyz: np.ndarray,
    curr_ori_wxyz: np.ndarray,
    frame_R: np.ndarray,
    init_fk_rot: jaxlie.SO3,
) -> jaxlie.SO3:
    """Transform XR rotation delta to robot rotation.

    1. Compute world-frame rotation delta: delta = current @ init^-1
    2. Transform delta from XR frame to robot frame using frame_R
    3. Apply: result = transformed_delta @ init_fk_rotation
    """
    # Quaternion to SO3
    init_so3 = jaxlie.SO3(wxyz=jnp.array(init_ori_wxyz))
    curr_so3 = jaxlie.SO3(wxyz=jnp.array(curr_ori_wxyz))

    # World-frame rotation delta
    delta_so3 = curr_so3 @ init_so3.inverse()

    # Convert to rotation matrix, transform frame, convert back
    delta_mat = np.array(delta_so3.as_matrix())
    R = frame_R
    # Similarity transform: R @ delta @ R^T
    transformed_mat = R @ delta_mat @ R.T

    # Convert back to SO3
    transformed_so3 = jaxlie.SO3.from_matrix(jnp.array(transformed_mat))

    # Apply to initial FK rotation
    return transformed_so3 @ init_fk_rot


class DebugTransformNode(Node):
    def __init__(self):
        super().__init__("debug_transform")

        self.declare_parameter("config_file", "")
        self.declare_parameter("enable_rotation", True)
        self.config_path = self.get_parameter("config_file").value
        self.enable_rotation = self.get_parameter("enable_rotation").value

        if self.config_path and os.path.exists(self.config_path):
            self.config = TeleopConfig.from_yaml(self.config_path)
            self.config_mtime = os.path.getmtime(self.config_path)
        else:
            self.config = TeleopConfig()
            self.config_mtime = 0
        self._print_config()

        self.get_logger().info("Loading robot model + IK solver...")
        robot_cls = load_robot_class("teleop_xr.ik.robots.openarm:OpenArmRobot")
        self.robot = robot_cls(weights=self.config.ik_weights)
        self.solver = PyrokiSolver(self.robot)
        self.actuated_names = self.robot.actuated_joint_names
        self.right_arm_names = [n for n in self.actuated_names if "right" in n and "finger" not in n]
        self.right_arm_indices = [self.actuated_names.index(n) for n in self.right_arm_names]

        self.q_current = jnp.array(self.robot.get_default_config())
        self.hand_pose: PoseStamped | None = None
        self.grip_pressed = False
        self.tracking_active = False
        self.init_pos: np.ndarray | None = None
        self.init_ori_wxyz: np.ndarray | None = None
        self.init_fk: jaxlie.SE3 | None = None
        self.body_rotation_2x2: np.ndarray | None = None
        self.frame_R: np.ndarray | None = None  # 3x3 frame rotation
        self.lock = threading.Lock()
        self.cmd_q = np.array(self.q_current)

        self.create_subscription(PoseStamped, "/q2r_right_hand_pose", self._hand_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(Bool, "/q2r_right_grip_pressed", self._grip_cb, 10)

        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        self.JointTrajectory = JointTrajectory
        self.JointTrajectoryPoint = JointTrajectoryPoint
        self.Duration = Duration
        self.pub = self.create_publisher(
            JointTrajectory, "/right_joint_trajectory_controller/joint_trajectory", 10
        )

        self.timer = self.create_timer(0.02, self._tick)
        self.print_count = 0
        rot_status = "ON" if self.enable_rotation else "OFF"
        self.get_logger().info(f"Ready. Rotation={rot_status}. Grip to start.")

    def _print_config(self):
        tc = self.config.transform
        print("\n" + "=" * 60)
        print("CONFIG:")
        print(f"  speed: {tc.speed}")
        print(f"  axis_map_signs: {tc.axis_map_signs}")
        print(f"  rotation: {'ON' if self.enable_rotation else 'OFF'}")
        print(f"  max_workspace_radius: {tc.max_workspace_radius}")
        print("=" * 60 + "\n")

    def _reload_config(self):
        if not self.config_path or not os.path.exists(self.config_path):
            return
        mtime = os.path.getmtime(self.config_path)
        if mtime > self.config_mtime:
            try:
                self.config = TeleopConfig.from_yaml(self.config_path)
                self.config_mtime = mtime
                print("\n*** CONFIG RELOADED ***")
                self._print_config()
            except Exception as e:
                print(f"\n*** CONFIG RELOAD FAILED: {e} ***")

    def _hand_cb(self, msg):
        with self.lock:
            self.hand_pose = msg

    def _joint_cb(self, msg):
        q = np.array(self.q_current)
        for i, name in enumerate(self.actuated_names):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]
        with self.lock:
            self.q_current = jnp.array(q)

    def _grip_cb(self, msg):
        with self.lock:
            old = self.grip_pressed
            self.grip_pressed = msg.data
            if old != msg.data:
                if msg.data:
                    print("\n>>> GRIP PRESSED")
                else:
                    print(">>> GRIP RELEASED")
                    self.tracking_active = False

    def _tick(self):
        self.print_count += 1
        if self.print_count % 50 == 0:
            self._reload_config()

        with self.lock:
            hp = self.hand_pose
            grip = self.grip_pressed
            q_cur = self.q_current

        if hp is None or not grip:
            return

        xr_pos = np.array([hp.pose.position.x, hp.pose.position.y, hp.pose.position.z])
        o = hp.pose.orientation
        curr_ori_wxyz = np.array([o.w, o.x, o.y, o.z])

        if not self.tracking_active:
            self.init_pos = xr_pos.copy()
            self.init_ori_wxyz = curr_ori_wxyz.copy()

            # Body frame from controller yaw
            fwd_2d = quat_to_forward_yaw(o.w, o.x, o.y, o.z)
            self.body_rotation_2x2 = np.array([
                [fwd_2d[0], fwd_2d[1]],    # forward row
                [-fwd_2d[1], fwd_2d[0]],   # right row (flipped)
            ])
            self.frame_R = build_frame_rotation_3x3(self.body_rotation_2x2)

            fk = self.robot.forward_kinematics(q_cur)
            self.init_fk = fk.get("right")
            if self.init_fk is None:
                return
            # Initialize cmd_q from actual joint state to prevent jump
            self.cmd_q = np.array(q_cur)
            self.tracking_active = True

            init_fk_t = np.array(self.init_fk.translation())
            angle = math.degrees(math.atan2(fwd_2d[1], fwd_2d[0]))
            print(f"  Init XR:   [{xr_pos[0]:.3f}, {xr_pos[1]:.3f}, {xr_pos[2]:.3f}]")
            print(f"  Init FK:   [{init_fk_t[0]:.3f}, {init_fk_t[1]:.3f}, {init_fk_t[2]:.3f}]")
            print(f"  Body fwd:  [{fwd_2d[0]:.3f}, {fwd_2d[1]:.3f}] (yaw={angle:.0f}°)")
            print(f"  Rotation:  {'ON' if self.enable_rotation else 'OFF'}")
            return

        tc = self.config.transform

        # --- Translation ---
        world_delta = xr_pos - self.init_pos
        horizontal_xr = np.array([world_delta[0], world_delta[1]])
        body_horizontal = self.body_rotation_2x2 @ horizontal_xr
        body_fwd = body_horizontal[0]
        body_right = body_horizontal[1]
        body_up = world_delta[2]

        sgn = tc.axis_map_signs
        robot_delta = np.array([
            sgn[0] * body_fwd,
            sgn[1] * body_right,
            sgn[2] * body_up,
        ]) * tc.speed

        dist = np.linalg.norm(robot_delta)
        if dist > tc.max_workspace_radius:
            robot_delta = robot_delta * (tc.max_workspace_radius / dist)

        init_t = np.array(self.init_fk.translation())
        target_pos = init_t + robot_delta

        # --- Rotation ---
        if self.enable_rotation:
            target_rot = transform_rotation(
                self.init_ori_wxyz,
                curr_ori_wxyz,
                self.frame_R,
                self.init_fk.rotation(),
            )
        else:
            target_rot = self.init_fk.rotation()

        target = jaxlie.SE3.from_rotation_and_translation(
            target_rot,
            jnp.array(target_pos),
        )

        # IK
        try:
            new_q = self.solver.solve(None, target, None, q_cur)
        except Exception as e:
            print(f"  IK error: {e}")
            return

        # Smooth + clamp
        new_q_np = np.array(new_q)
        desired = self.cmd_q + 0.05 * (new_q_np - self.cmd_q)
        delta_q = np.clip(desired - self.cmd_q, -0.02, 0.02)
        self.cmd_q = self.cmd_q + delta_q

        # Publish
        msg = self.JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.right_arm_names
        pt = self.JointTrajectoryPoint()
        pt.positions = [float(self.cmd_q[i]) for i in self.right_arm_indices]
        pt.time_from_start = self.Duration(sec=0, nanosec=int(1e7))
        msg.points = [pt]
        self.pub.publish(msg)

        # Print every 0.5s
        if self.print_count % 25 == 0:
            q_arm = [float(self.cmd_q[i]) for i in self.right_arm_indices]
            print(
                f"  body: fwd={body_fwd:+.3f} right={body_right:+.3f} up={body_up:+.3f} | "
                f"robot: [{robot_delta[0]:+.3f}, {robot_delta[1]:+.3f}, {robot_delta[2]:+.3f}]"
            )
            print(
                f"  joints: [{', '.join(f'{v:+.3f}' for v in q_arm)}]"
            )


def main(args=None):
    rclpy.init(args=args)
    node = DebugTransformNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
