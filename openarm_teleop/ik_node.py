#!/usr/bin/env python3
"""
Quest2ROS2 → pyroki IK → OpenArm bimanual JointTrajectory

Body-relative tracking with safety features:
- World-frame delta with body orientation estimation
- Rotation tracking via frame transform
- Joint velocity clamping
- Workspace radius limit
- Exponential smoothing
"""

import threading
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool, Float32, Float64MultiArray
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient

from teleop_xr.ik_utils import ensure_ik_dependencies

ensure_ik_dependencies()

import jax.numpy as jnp
import jaxlie

from teleop_xr.ik.loader import load_robot_class
from teleop_xr.ik.solver import PyrokiSolver
from teleop_xr.ik.weights import TeleopConfig


def quat_to_forward_yaw(w, x, y, z):
    """Extract horizontal forward direction from controller quaternion."""
    fx = 2.0 * (x * z + w * y)
    fy = 2.0 * (y * z - w * x)
    length = math.sqrt(fx * fx + fy * fy)
    if length < 0.01:
        return np.array([1.0, 0.0])
    return np.array([fx / length, fy / length])


def build_frame_rotation_3x3(body_rot_2x2):
    """Build 3x3 rotation matrix from XR world to body frame."""
    return np.array([
        [body_rot_2x2[0, 0], body_rot_2x2[0, 1], 0.0],
        [body_rot_2x2[1, 0], body_rot_2x2[1, 1], 0.0],
        [0.0,                0.0,                 1.0],
    ])


def transform_rotation(init_ori_wxyz, curr_ori_wxyz, frame_R, init_fk_rot):
    """Transform XR rotation delta to robot rotation via frame_R."""
    init_so3 = jaxlie.SO3(wxyz=jnp.array(init_ori_wxyz))
    curr_so3 = jaxlie.SO3(wxyz=jnp.array(curr_ori_wxyz))
    delta_so3 = curr_so3 @ init_so3.inverse()
    delta_mat = np.array(delta_so3.as_matrix())
    transformed_mat = frame_R @ delta_mat @ frame_R.T
    transformed_so3 = jaxlie.SO3.from_matrix(jnp.array(transformed_mat))
    return transformed_so3 @ init_fk_rot


class ArmTrackingState:
    """Per-arm tracking state."""

    def __init__(self):
        self.init_pos: np.ndarray | None = None
        self.init_ori_wxyz: np.ndarray | None = None
        self.init_fk: jaxlie.SE3 | None = None
        self.body_rotation_2x2: np.ndarray | None = None
        self.frame_R: np.ndarray | None = None
        self.active = False
        self.grip_pressed = False
        self.grip_was_pressed = False
        self.grip_ever_received = False
        self.trigger_value = 0.0
        self.trigger_ever_received = False
        self.gripper_pos_cmd = 0.0
        self.last_gripper_sent = -1.0

    def reset(self):
        self.active = False


class QuestTeleopIKNode(Node):
    def __init__(self):
        # When running two instances (arm:=right and arm:=left), remap the
        # node name via -r __node:=right_ik / left_ik at launch.
        super().__init__("quest_teleop_ik")

        # --- Parameters ---
        self.declare_parameter("use_forward_controller", True)
        self.declare_parameter("smoothing", 0.05)
        self.declare_parameter("max_joint_step", 0.02)
        self.declare_parameter("weights_file", "")
        # arm: "both" (default, legacy bimanual in one process),
        #      "right" or "left" (this node handles only that arm —
        #      run two instances to get independent right/left IK).
        self.declare_parameter("arm", "both")
        # pose_filter_alpha: EMA coefficient applied to Quest pose.
        # 1.0 = no filter (raw pose, original behavior).
        # Smaller = more filtering / smoother = more lag.
        # At 20Hz pose rate, alpha=0.3 gives ~150ms time constant.
        self.declare_parameter("pose_filter_alpha", 0.3)

        self.use_forward = self.get_parameter("use_forward_controller").value
        self.smoothing = self.get_parameter("smoothing").value
        self.max_joint_step = self.get_parameter("max_joint_step").value
        self.pose_filter_alpha = float(self.get_parameter("pose_filter_alpha").value)
        self.arm_side = self.get_parameter("arm").value
        if self.arm_side not in ("both", "right", "left"):
            raise ValueError(f"param 'arm' must be both/right/left, got {self.arm_side}")
        self.handle_right = self.arm_side in ("both", "right")
        self.handle_left = self.arm_side in ("both", "left")
        self.get_logger().info(f"arm mode: {self.arm_side}")

        # --- Robot & IK setup ---
        self.get_logger().info("Initializing OpenArmRobot + PyrokiSolver (JIT warmup)...")
        robot_cls = load_robot_class("teleop_xr.ik.robots.openarm:OpenArmRobot")
        weights_path = self.get_parameter("weights_file").value
        teleop_config = TeleopConfig.from_yaml(weights_path) if weights_path else TeleopConfig()
        if weights_path:
            self.get_logger().info(f"Loaded teleop config from {weights_path}")
        self.teleop_config = teleop_config
        self.robot = robot_cls(weights=teleop_config.ik_weights)
        # Warm up only the IK patterns we actually need for this arm mode
        # → faster startup and smaller JIT cache.
        if self.arm_side == "right":
            warmup_patterns = [(False, True, False)]
        elif self.arm_side == "left":
            warmup_patterns = [(True, False, False)]
        else:
            # both: need L-only, R-only, and (L,R) combinations for the
            # independent-solve codepath
            warmup_patterns = [
                (False, True, False),
                (True, False, False),
                (True, True, False),
            ]
        self.solver = PyrokiSolver(self.robot, warmup_patterns=warmup_patterns)
        self.get_logger().info("IK solver ready.")

        self.q_current = jnp.array(self.robot.get_default_config())
        self.actuated_names = self.robot.actuated_joint_names
        self.get_logger().info(f"Actuated joints: {self.actuated_names}")

        # Arm joint names/indices
        self.right_arm_names = [n for n in self.actuated_names if "right" in n and "finger" not in n]
        self.right_arm_indices = [self.actuated_names.index(n) for n in self.right_arm_names]
        self.left_arm_names = [n for n in self.actuated_names if "left" in n and "finger" not in n]
        self.left_arm_indices = [self.actuated_names.index(n) for n in self.left_arm_names]

        # --- Per-arm tracking state ---
        self.right = ArmTrackingState()
        self.left = ArmTrackingState()

        # --- Gripper ---
        self.gripper_max = 0.044
        self.gripper_smoothing = 0.1

        # --- State ---
        self.right_pose: PoseStamped | None = None
        self.left_pose: PoseStamped | None = None
        self.lock = threading.Lock()
        self.publish_count = 0
        self.cmd_q = None
        # Guard: only start tracking once we've seen real joint states, otherwise
        # cmd_q would be initialized from the hardcoded default_config, causing
        # the arm to snap to that pose on the first publish.
        self.joint_state_received = False

        # --- Subscribers ---
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        if self.handle_right:
            self.create_subscription(PoseStamped, "/q2r_right_hand_pose", self._right_pose_cb, 10)
            self.create_subscription(Bool, "/q2r_right_grip_pressed", self._right_grip_cb, 10)
            self.create_subscription(Float32, "/q2r_right_trigger_value", self._right_trigger_cb, 10)
        if self.handle_left:
            self.create_subscription(PoseStamped, "/q2r_left_hand_pose", self._left_pose_cb, 10)
            self.create_subscription(Bool, "/q2r_left_grip_pressed", self._left_grip_cb, 10)
            self.create_subscription(Float32, "/q2r_left_trigger_value", self._left_trigger_cb, 10)

        # --- Gripper action clients ---
        if self.handle_right:
            self.right_gripper_client = ActionClient(
                self, GripperCommand, "/right_gripper_controller/gripper_cmd"
            )
        if self.handle_left:
            self.left_gripper_client = ActionClient(
                self, GripperCommand, "/left_gripper_controller/gripper_cmd"
            )

        # --- Publishers ---
        if self.use_forward:
            if self.handle_right:
                self.pub_right = self.create_publisher(
                    Float64MultiArray, "/right_forward_position_controller/commands", 10
                )
            if self.handle_left:
                self.pub_left = self.create_publisher(
                    Float64MultiArray, "/left_forward_position_controller/commands", 10
                )
            self.get_logger().info(f"Mode: {self.arm_side} forward_position_controller")
        else:
            if self.handle_right:
                self.pub_right = self.create_publisher(
                    JointTrajectory, "/right_joint_trajectory_controller/joint_trajectory", 10
                )
            if self.handle_left:
                self.pub_left = self.create_publisher(
                    JointTrajectory, "/left_joint_trajectory_controller/joint_trajectory", 10
                )
            self.get_logger().info(f"Mode: {self.arm_side} joint_trajectory_controller")

        # --- Control loop at 50 Hz ---
        self.timer = self.create_timer(0.02, self._control_loop)
        tc = self.teleop_config.transform
        self.get_logger().info(
            f"Started (50Hz, smoothing={self.smoothing}, speed={tc.speed}, "
            f"max_joint_step={self.max_joint_step} rad, max_workspace={tc.max_workspace_radius} m)"
        )

    # ------------------------------------------------------------------ pose filter
    def _filter_pose(self, old: PoseStamped | None, new: PoseStamped) -> PoseStamped:
        """Exponential moving average of a Quest pose.

        Position: scalar EMA on each axis.
        Orientation: EMA on quaternion components (after double-cover sign
        flip) + renormalize. This is a valid approximation of SLERP for small
        per-sample deltas, which is what we expect at ~20Hz from Quest.
        """
        a = self.pose_filter_alpha
        if old is None or a >= 0.999:
            return new
        out = PoseStamped()
        out.header = new.header
        # Position
        out.pose.position.x = a * new.pose.position.x + (1.0 - a) * old.pose.position.x
        out.pose.position.y = a * new.pose.position.y + (1.0 - a) * old.pose.position.y
        out.pose.position.z = a * new.pose.position.z + (1.0 - a) * old.pose.position.z
        # Orientation: resolve double-cover (pick shorter-path sign) then EMA
        ow, ox, oy, oz = (old.pose.orientation.w, old.pose.orientation.x,
                          old.pose.orientation.y, old.pose.orientation.z)
        nw, nx, ny, nz = (new.pose.orientation.w, new.pose.orientation.x,
                          new.pose.orientation.y, new.pose.orientation.z)
        if ow * nw + ox * nx + oy * ny + oz * nz < 0.0:
            nw, nx, ny, nz = -nw, -nx, -ny, -nz
        qw = a * nw + (1.0 - a) * ow
        qx = a * nx + (1.0 - a) * ox
        qy = a * ny + (1.0 - a) * oy
        qz = a * nz + (1.0 - a) * oz
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm > 1e-6:
            qw /= norm; qx /= norm; qy /= norm; qz /= norm
        out.pose.orientation.w = qw
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        return out

    # ------------------------------------------------------------------ callbacks
    def _right_pose_cb(self, msg):
        with self.lock:
            self.right_pose = self._filter_pose(self.right_pose, msg)

    def _left_pose_cb(self, msg):
        with self.lock:
            self.left_pose = self._filter_pose(self.left_pose, msg)

    def _right_grip_cb(self, msg):
        with self.lock:
            arm = self.right
            if not arm.grip_ever_received:
                arm.grip_ever_received = True
                self.get_logger().info("Right grip topic received")
            old = arm.grip_pressed
            arm.grip_pressed = msg.data
            if old != msg.data:
                self.get_logger().info(f"Right grip: {'PRESSED' if msg.data else 'RELEASED'}")

    def _left_grip_cb(self, msg):
        with self.lock:
            arm = self.left
            if not arm.grip_ever_received:
                arm.grip_ever_received = True
                self.get_logger().info("Left grip topic received")
            old = arm.grip_pressed
            arm.grip_pressed = msg.data
            if old != msg.data:
                self.get_logger().info(f"Left grip: {'PRESSED' if msg.data else 'RELEASED'}")

    def _right_trigger_cb(self, msg):
        with self.lock:
            self.right.trigger_value = msg.data
            self.right.trigger_ever_received = True

    def _left_trigger_cb(self, msg):
        with self.lock:
            self.left.trigger_value = msg.data
            self.left.trigger_ever_received = True

    def _joint_state_cb(self, msg):
        q = np.array(self.q_current)
        for i, name in enumerate(self.actuated_names):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]
        self.q_current = jnp.array(q)
        self.joint_state_received = True

    # ------------------------------------------------------------------ arm target
    def _compute_arm_target(
        self, pose: PoseStamped, arm_state: ArmTrackingState, fk_key: str
    ) -> jaxlie.SE3 | None:
        """Compute IK target for one arm. Returns None if not tracking."""
        with self.lock:
            grip = arm_state.grip_pressed

        if not grip:
            if arm_state.grip_was_pressed:
                arm_state.reset()
                self.get_logger().info(f"{fk_key} arm released")
            arm_state.grip_was_pressed = False
            return None

        p = pose.pose.position
        o = pose.pose.orientation
        xr_pos = np.array([p.x, p.y, p.z])
        xr_ori_wxyz = np.array([o.w, o.x, o.y, o.z])

        if not arm_state.active:
            arm_state.init_pos = xr_pos.copy()
            arm_state.init_ori_wxyz = xr_ori_wxyz.copy()

            fwd_2d = quat_to_forward_yaw(o.w, o.x, o.y, o.z)
            arm_state.body_rotation_2x2 = np.array([
                [fwd_2d[0], fwd_2d[1]],
                [-fwd_2d[1], fwd_2d[0]],
            ])
            arm_state.frame_R = build_frame_rotation_3x3(arm_state.body_rotation_2x2)

            fk = self.robot.forward_kinematics(self.q_current)
            arm_state.init_fk = fk.get(fk_key)
            if arm_state.init_fk is None:
                return None
            arm_state.active = True
            arm_state.grip_was_pressed = True
            self.get_logger().info(f"{fk_key} arm tracking started")
            return None  # skip first frame

        arm_state.grip_was_pressed = True
        tc = self.teleop_config.transform

        # Translation: world delta → body frame → robot frame
        world_delta = xr_pos - arm_state.init_pos
        horizontal_xr = np.array([world_delta[0], world_delta[1]])
        body_horizontal = arm_state.body_rotation_2x2 @ horizontal_xr
        sgn = tc.axis_map_signs
        robot_delta = np.array([
            sgn[0] * body_horizontal[0],
            sgn[1] * body_horizontal[1],
            sgn[2] * world_delta[2],
        ]) * tc.speed

        dist = np.linalg.norm(robot_delta)
        if dist > tc.max_workspace_radius:
            robot_delta = robot_delta * (tc.max_workspace_radius / dist)

        init_t = np.array(arm_state.init_fk.translation())
        target_pos = init_t + robot_delta

        # Rotation
        target_rot = transform_rotation(
            arm_state.init_ori_wxyz, xr_ori_wxyz,
            arm_state.frame_R, arm_state.init_fk.rotation(),
        )

        return jaxlie.SE3.from_rotation_and_translation(
            target_rot, jnp.array(target_pos),
        )

    # ------------------------------------------------------------------ gripper
    def _handle_gripper(self, arm_state: ArmTrackingState, client: ActionClient):
        with self.lock:
            trigger = arm_state.trigger_value
            ready = arm_state.trigger_ever_received
        if not ready:
            return
        target = self.gripper_max * trigger
        arm_state.gripper_pos_cmd += self.gripper_smoothing * (target - arm_state.gripper_pos_cmd)
        if abs(arm_state.gripper_pos_cmd - arm_state.last_gripper_sent) > 0.0005:
            arm_state.last_gripper_sent = arm_state.gripper_pos_cmd
            goal = GripperCommand.Goal()
            goal.command.position = arm_state.gripper_pos_cmd
            goal.command.max_effort = 100.0
            client.send_goal_async(goal)

    # ------------------------------------------------------------------ control loop
    def _control_loop(self):
        # Wait until we've actually seen joint states from the robot —
        # otherwise cmd_q would snapshot the hardcoded default_config and the
        # arm would snap to that pose on the first publish.
        if not self.joint_state_received:
            return

        with self.lock:
            rp = self.right_pose
            lp = self.left_pose

        if rp is None and lp is None:
            return

        # Initialize cmd_q from current joint state
        if self.cmd_q is None:
            self.cmd_q = np.array(self.q_current)

        # Compute targets for each arm
        target_R = self._compute_arm_target(rp, self.right, "right") if rp else None
        target_L = self._compute_arm_target(lp, self.left, "left") if lp else None

        # Skip if neither arm is tracking
        if target_R is None and target_L is None:
            return

        # Solve IK
        try:
            new_q = self.solver.solve(target_L, target_R, None, self.q_current)
        except Exception as e:
            self.get_logger().error(f"IK solve error: {e}")
            return

        # Smooth + clamp (only for actively tracked arms)
        new_q_np = np.array(new_q)
        desired = self.cmd_q + self.smoothing * (new_q_np - self.cmd_q)
        delta_q = np.clip(desired - self.cmd_q, -self.max_joint_step, self.max_joint_step)

        # Mask: only update joints of tracked arms, keep others at current state
        if not self.right.active:
            for i in self.right_arm_indices:
                delta_q[i] = 0.0
        if not self.left.active:
            for i in self.left_arm_indices:
                delta_q[i] = 0.0

        self.cmd_q = self.cmd_q + delta_q

        # Keep untracked arms synced with actual joint state
        q_actual = np.array(self.q_current)
        if not self.right.active:
            for i in self.right_arm_indices:
                self.cmd_q[i] = q_actual[i]
        if not self.left.active:
            for i in self.left_arm_indices:
                self.cmd_q[i] = q_actual[i]

        # Publish right arm
        if self.right.active:
            if self.use_forward:
                msg = Float64MultiArray()
                msg.data = [float(self.cmd_q[i]) for i in self.right_arm_indices]
                self.pub_right.publish(msg)
            else:
                stamp = self.get_clock().now().to_msg()
                msg = JointTrajectory()
                msg.header.stamp = stamp
                msg.joint_names = self.right_arm_names
                pt = JointTrajectoryPoint()
                pt.positions = [float(self.cmd_q[i]) for i in self.right_arm_indices]
                pt.time_from_start = Duration(sec=0, nanosec=int(1e7))
                msg.points = [pt]
                self.pub_right.publish(msg)

        # Publish left arm
        if self.left.active:
            if self.use_forward:
                msg = Float64MultiArray()
                msg.data = [float(self.cmd_q[i]) for i in self.left_arm_indices]
                self.pub_left.publish(msg)
            else:
                stamp = self.get_clock().now().to_msg()
                msg = JointTrajectory()
                msg.header.stamp = stamp
                msg.joint_names = self.left_arm_names
                pt = JointTrajectoryPoint()
                pt.positions = [float(self.cmd_q[i]) for i in self.left_arm_indices]
                pt.time_from_start = Duration(sec=0, nanosec=int(1e7))
                msg.points = [pt]
                self.pub_left.publish(msg)

        self.publish_count += 1
        if self.publish_count % 250 == 1:
            active = []
            if self.right.active:
                active.append("R")
            if self.left.active:
                active.append("L")
            self.get_logger().info(f"#{self.publish_count} active=[{','.join(active)}]")

        # Grippers
        if self.handle_right:
            self._handle_gripper(self.right, self.right_gripper_client)
        if self.handle_left:
            self._handle_gripper(self.left, self.left_gripper_client)


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
