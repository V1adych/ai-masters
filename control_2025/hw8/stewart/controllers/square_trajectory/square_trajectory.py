from itertools import cycle
from controller import Supervisor
import numpy as np

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
base_piston_length = 2.82525225
kp = 0.5
kd = 0.4
base_piston_lengths = np.array([base_piston_length] * 6, dtype=np.float64)


pistons = []
for i in range(6):
    motor = robot.getDevice(f"piston{i}")
    motor.setVelocity(10)
    pistons.append(motor)


ball_node = robot.getFromDef("BALL")
upper_platform_node = robot.getFromDef("UPPER_PLATFORM")
lower_hinge_nodes = [robot.getFromDef(f"LOWER_HINGE2_{i}") for i in range(6)]


def get_geometry_constants() -> tuple[np.ndarray, np.ndarray]:
    base_anchors = []
    platform_anchors = []

    plat_node = robot.getFromDef("UPPER_PLATFORM")
    plat_pos = np.array(plat_node.getPosition(), dtype=np.float64)
    plat_rot = np.array(plat_node.getOrientation(), dtype=np.float64).reshape(3, 3)

    for i in range(6):
        joint_node = robot.getFromDef(f"LOWER_HINGE2_{i}")
        joint_params = joint_node.getField("jointParameters").getSFNode()
        b_vec = joint_params.getField("anchor").getSFVec3f()
        base_anchors.append(b_vec)
        piston_node = robot.getFromDef(f"UPPER_PISTON_{i}")
        p_pos = np.array(piston_node.getPosition(), dtype=np.float64)
        p_rot = np.array(piston_node.getOrientation(), dtype=np.float64).reshape(3, 3)
        anchor_local = np.array([0.2, 0.0, 0.0], dtype=np.float64)
        anchor_global = p_pos + p_rot @ anchor_local
        anchor_platform_local = plat_rot.T @ (anchor_global - plat_pos)
        platform_anchors.append(anchor_platform_local)
    return np.array(base_anchors, dtype=np.float64), np.array(platform_anchors, dtype=np.float64)


def get_ball_position() -> np.ndarray:
    pos = ball_node.getPosition()
    return np.array([pos[0], pos[1], pos[2]], dtype=np.float64)


def get_ball_velocity() -> np.ndarray:
    vel = ball_node.getVelocity()
    return np.array([vel[0], vel[1], vel[2]], dtype=np.float64)


def get_platform_position() -> np.ndarray:
    pos = upper_platform_node.getPosition()
    return np.array([pos[0], pos[1], pos[2]], dtype=np.float64)


def get_platform_rot_matrix() -> np.ndarray:
    return np.array(upper_platform_node.getOrientation(), dtype=np.float64).reshape(3, 3)


def passive_wait(sec: float):
    start_time = robot.getTime()
    while robot.step(timestep) != -1:
        if robot.getTime() - start_time >= sec:
            break


def get_rot_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rot_roll = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]], dtype=np.float64
    )
    rot_pitch = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float64
    )
    rot_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], dtype=np.float64)

    return rot_roll @ rot_pitch @ rot_yaw


def get_piston_vectors(t: np.ndarray, r: np.ndarray, p: np.ndarray, b: np.ndarray) -> np.ndarray:
    return t[None] + (r @ p.T).T - b


def set_piston_lengths(lengths: np.ndarray):
    adjusted_lengths = lengths - base_piston_lengths
    for i in range(6):
        pistons[i].setPosition(adjusted_lengths[i])
    

square_side_length = 1.0
a = square_side_length / 2

target_positions = cycle(
    list(np.array([[a, a], [a, -a], [-a, -a], [-a, a]], dtype=np.float64))
)

target_position = next(target_positions)
    
while robot.step(timestep) != -1:
    ball_pos = get_ball_position()
    ball_velocity = get_ball_velocity()
    platform_pos = get_platform_position()
    platform_rot = get_platform_rot_matrix()
    base_anchors, platform_anchors = get_geometry_constants()
    target_pos_world = platform_pos[:2] + target_position
    ball_displacement = ball_pos[:2] - target_pos_world
    pos_err = 0 - ball_displacement
    vel_err = 0 - ball_velocity
    if np.linalg.norm(pos_err) < 0.1 and np.linalg.norm(vel_err) < 0.1:
        target_position = next(target_positions)
        continue
    roll_cmd = -(kp * pos_err[0] + kd * vel_err[0])
    pitch_cmd = kp * pos_err[1] + kd * vel_err[1]
    rot_mat = get_rot_mat(pitch_cmd, roll_cmd, 0)
    vectors = get_piston_vectors(platform_pos, rot_mat, platform_anchors, base_anchors)

    vector_lengths = np.linalg.norm(vectors, axis=1)

    set_piston_lengths(vector_lengths)


