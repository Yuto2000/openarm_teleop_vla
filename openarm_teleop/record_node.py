#!/usr/bin/env python3
"""
Teaching by Demonstration: データ記録ノード

ロボットをcompliant modeにした状態で、人間がXRコントローラーを持ちながら
ロボットの腕を直接動かし、両方のデータを同期記録する。

記録データ:
  - 生XRコントローラーポーズ（変換前）
  - XRコントローラーのポーズ → IKターゲットSE3に変換済み
  - 実関節角度（エンコーダ値）

使い方:
  ros2 run openarm_teleop record_node --ros-args \
    -p output_path:=demo1.npz \
    -p arm:=right

Ctrl+C で記録終了、npzファイル保存。
"""

import os
import threading

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


def pose_msg_to_arrays(msg: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
    """Extract position [x,y,z] and orientation [w,x,y,z] from PoseStamped."""
    p = msg.pose.position
    o = msg.pose.orientation
    return (
        np.array([p.x, p.y, p.z]),
        np.array([o.w, o.x, o.y, o.z]),
    )


class DemoRecorderNode(Node):
    """Records synchronized XR controller poses and joint states for IK optimization."""

    def __init__(self):
        super().__init__("demo_recorder")

        # --- Parameters ---
        self.declare_parameter("output_path", "~/demo_recording.npz")
        self.declare_parameter("arm", "right")

        self.output_path = os.path.expanduser(
            self.get_parameter("output_path").value
        )
        self.arm = self.get_parameter("arm").value

        # --- Robot model (for FK and joint names only, no IK) ---
        self.get_logger().info("Loading robot model...")
        robot_cls = load_robot_class("teleop_xr.ik.robots.openarm:OpenArmRobot")
        self.robot = robot_cls()
        self.actuated_names = self.robot.actuated_joint_names
        self.get_logger().info(f"Actuated joints: {self.actuated_names}")

        # --- State ---
        self.q_current = np.array(self.robot.get_default_config())
        self.hand_pose: PoseStamped | None = None
        self.grip_pressed = False
        self.tracking_active = False
        self.init_raw_pos: np.ndarray | None = None
        self.init_raw_ori: np.ndarray | None = None
        self.init_fk_pos: np.ndarray | None = None
        self.init_fk_ori: np.ndarray | None = None
        self.lock = threading.Lock()

        # --- Recording buffers ---
        self.timestamps: list[float] = []
        self.raw_delta_positions: list[np.ndarray] = []
        self.raw_delta_orientations: list[np.ndarray] = []
        self.init_fk_positions: list[np.ndarray] = []
        self.init_fk_orientations: list[np.ndarray] = []
        self.joint_positions: list[np.ndarray] = []

        # --- Subscribers ---
        hand_topic = f"/q2r_{self.arm}_hand_pose"
        self.create_subscription(PoseStamped, hand_topic, self._hand_pose_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.create_subscription(Bool, f"/q2r_{self.arm}_grip_pressed", self._grip_cb, 10)

        # --- Timer at 50Hz ---
        self.timer = self.create_timer(0.02, self._record_tick)

        self.get_logger().info(
            f"Demo recorder ready. arm={self.arm}, output={self.output_path}"
        )
        self.get_logger().info(
            "Squeeze grip to start recording, release to pause. Ctrl+C to save and exit."
        )

    # ------------------------------------------------------------------ callbacks
    def _hand_pose_cb(self, msg: PoseStamped):
        with self.lock:
            self.hand_pose = msg

    def _joint_state_cb(self, msg: JointState):
        q = np.array(self.q_current)
        for i, name in enumerate(self.actuated_names):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]
        with self.lock:
            self.q_current = q

    def _grip_cb(self, msg: Bool):
        with self.lock:
            old = self.grip_pressed
            self.grip_pressed = msg.data
            if old != msg.data:
                self.get_logger().info(
                    f"Grip: {'PRESSED - recording' if msg.data else 'RELEASED - paused'}"
                )

    # ------------------------------------------------------------------ recording
    def _record_tick(self):
        with self.lock:
            hp = self.hand_pose
            grip = self.grip_pressed
            q = np.array(self.q_current)

        if hp is None or not grip:
            if self.tracking_active and not grip:
                self.tracking_active = False
            return

        raw_pos, raw_ori = pose_msg_to_arrays(hp)

        # Initialize tracking on grip press
        if not self.tracking_active:
            self.init_raw_pos = raw_pos
            self.init_raw_ori = raw_ori
            fk = self.robot.forward_kinematics(jnp.array(q))
            fk_se3 = fk.get(self.arm)
            if fk_se3 is None:
                return
            self.init_fk_pos = np.array(fk_se3.translation())
            self.init_fk_ori = np.array(fk_se3.rotation().wxyz)
            self.tracking_active = True
            self.get_logger().info("Tracking started (origin set)")
            return

        # Compute raw delta (before any axis mirroring or speed scaling)
        init_se3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(wxyz=jnp.array(self.init_raw_ori)),
            jnp.array(self.init_raw_pos),
        )
        curr_se3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(wxyz=jnp.array(raw_ori)),
            jnp.array(raw_pos),
        )
        delta = init_se3.inverse() @ curr_se3
        delta_pos = np.array(delta.translation())
        delta_ori = np.array(delta.rotation().wxyz)

        timestamp = self.get_clock().now().nanoseconds / 1e9

        self.timestamps.append(timestamp)
        self.raw_delta_positions.append(delta_pos)
        self.raw_delta_orientations.append(delta_ori)
        self.init_fk_positions.append(self.init_fk_pos)
        self.init_fk_orientations.append(self.init_fk_ori)
        self.joint_positions.append(q)

        if len(self.timestamps) % 250 == 1:
            self.get_logger().info(
                f"Recorded {len(self.timestamps)} samples "
                f"({len(self.timestamps) / 50.0:.1f}s)"
            )

    # ------------------------------------------------------------------ save
    def save_recording(self):
        n = len(self.timestamps)
        if n == 0:
            self.get_logger().warn("No data recorded, skipping save.")
            return

        self.get_logger().info(f"Saving {n} samples to {self.output_path}...")
        np.savez(
            self.output_path,
            timestamps=np.array(self.timestamps),
            raw_delta_positions=np.array(self.raw_delta_positions),
            raw_delta_orientations=np.array(self.raw_delta_orientations),
            init_fk_positions=np.array(self.init_fk_positions),
            init_fk_orientations=np.array(self.init_fk_orientations),
            joint_positions=np.array(self.joint_positions),
            joint_names=np.array(self.actuated_names),
            arm=np.array(self.arm),
        )
        self.get_logger().info(f"Saved to {self.output_path}")


def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_recording()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
