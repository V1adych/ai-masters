# Copyright 1996-2024 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ned_Controller controller in python.

# Webots controller for the Niryo Ned robot.
# With this controller, you can see the 6 different axis of the robot moving
# You can also control the robots with your keyboard and launch a Pick and Pack

import math
from controller import Keyboard, Supervisor

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
ned_box = robot.getFromDef("NedBox")
self_node = robot.getSelf()


box_pos_world = ned_box.getField("translation").getSFVec3f()
print("Box position (world):", box_pos_world)  # [x, y, z] in world coordinates
base_translation = self_node.getField("translation").getSFVec3f()
base_rotation = self_node.getField("rotation").getSFRotation()  # [ax, ay, az, theta]
theta_base = base_rotation[3]


def world_to_base(p_world):
    dx = p_world[0] - base_translation[0]
    dy = p_world[1] - base_translation[1]
    dz = p_world[2] - base_translation[2]
    # rotate around z by -theta_base
    c = math.cos(-theta_base)
    s = math.sin(-theta_base)
    x = c * dx - s * dy
    y = s * dx + c * dy
    z = dz
    return [x, y, z]


box_pos = world_to_base(box_pos_world)
print("Box position (base):", box_pos)

# Init the motors - the Ned robot is a 6-axis robot arm
# You can find the name of the rotationalMotors is the device parameters of each HingeJoints
m1 = robot.getDevice("joint_1")
m2 = robot.getDevice("joint_2")
m3 = robot.getDevice("joint_3")
m4 = robot.getDevice("joint_4")
m5 = robot.getDevice("joint_5")
m6 = robot.getDevice("joint_6")
m7 = robot.getDevice("gripper::left")

# Set the motor velocity
# First we make sure that every joints are at their initial positions
m1.setPosition(0)
m2.setPosition(0)
m3.setPosition(0)
m4.setPosition(0)
m5.setPosition(0)
m6.setPosition(0)
m7.setPosition(0)

# Set the motors speed. Here we set it to 1 radian/second
m1.setVelocity(1)
m2.setVelocity(1)
m3.setVelocity(1)
m4.setVelocity(1)
m5.setVelocity(1)
m6.setVelocity(1)
m7.setVelocity(1)

TIME_STEP = int(robot.getBasicTimeStep())

def wrap_angle(angle: float):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def passive_wait(sec: float):
    start_time = robot.getTime()
    while robot.step(TIME_STEP) != -1:
        if robot.getTime() - start_time >= sec:
            break


def to_zero():
    for m in [m1, m2, m3, m4, m5, m6, m7]:
        m.setPosition(0)


def pick_up_box(x: float, y: float, z: float):
    # link lengths
    a = 0.221  # shoulder–elbow
    b = 0.35  # elbow–wrist
    l = 0.1715  # shoulder height


    for m in [m1, m2, m3, m4, m5, m6, m7]:
        m.setVelocity(1)

    box_pos_world = [x, y, z]
    box_pos = world_to_base(box_pos_world)
    print("Box position (base):", box_pos)

    q1 = math.atan2(box_pos[1], box_pos[0]) - math.pi / 2
    q1 = wrap_angle(q1)
    m1.setPosition(q1)
    passive_wait(2.0)

    r = math.hypot(box_pos[0], box_pos[1])
    print(f"r: {r}")
    z_rel = l# - box_pos[2]
    c = math.hypot(r, z_rel)
    print(f"triangle sides: {a}, {b}, {c}")

    # cos_beta = (a * a + b * b - c * c) / (2 * a * b)
    # beta = math.acos(cos_beta)
    # print(f"elbow angle: {beta}")
    # adjust_beta = math.pi / 2 - beta
    # m3.setPosition(adjust_beta)
    # passive_wait(1.0)

    gamma = math.acos(z_rel / c)
    print(f"gamma: {gamma}")
    cos_alpha = (a * a + c * c - b * b) / (2 * a * c)
    alpha = math.acos(cos_alpha)
    print(f"alpha: {alpha}")
    adjust_alpha = math.pi - alpha - gamma
    m2.setPosition(adjust_alpha)
    passive_wait(2.0)

    m3.setPosition(math.pi / 4)
    passive_wait(1.0)
    m3.setPosition(0)

    to_zero()


while True:
    print("------------COMMANDS--------------")
    print("Move joint_1 --> A or Z")
    print("Move joint_2 --> Q or S")
    print("Move joint_3 --> W or X")
    print("Move joint_4 --> Y or U")
    print("Move joint_5 --> H or J")
    print("Move joint_6 --> B or N")
    print("Open/Close Gripper --> L or M")
    print("Pick Up Box --> P")
    print("Reset to Zero --> R")
    print("----------------------------------")

    timestep = int(robot.getBasicTimeStep())
    keyboard = Keyboard()
    keyboard.enable(timestep)

    while robot.step(timestep) != -1:
        key = keyboard.getKey()

        if key == ord("A"):
            print("Move --> joint_1 left")
            m1.setPosition(-1.5)

        elif key == ord("Z"):
            print("Move --> joint_1 right")
            m1.setPosition(1.5)

        elif key == ord("Q"):
            print("Move --> joint_2 left")
            m2.setPosition(0.5)

        elif key == ord("S"):
            print("Move --> joint_2 right")
            m2.setPosition(-0.5)

        elif key == ord("W"):
            print("Move --> joint_3 left")
            m3.setPosition(0.5)

        elif key == ord("X"):
            print("Move --> joint_3 right")
            m3.setPosition(-0.5)

        elif key == ord("Y"):
            print("Move --> joint_4 left")
            m4.setPosition(1)

        elif key == ord("U"):
            print("Move --> joint_4 right")
            m4.setPosition(-1)

        elif key == ord("H"):
            print("Move --> joint_5 left")
            m5.setPosition(1.4)

        elif key == ord("J"):
            print("Move --> joint_5 right")
            m5.setPosition(-1.4)

        elif key == ord("B"):
            print("Move --> joint_6 left")
            m6.setPosition(1.5)

        elif key == ord("N"):
            print("Move --> joint_6 right")
            m6.setPosition(-1.5)

        elif key == ord("L"):
            print("Move --> Open Gripper")
            m7.setPosition(0.01)

        elif key == ord("M"):
            print("Move --> Close Gripper")
            m7.setPosition(0)

        elif key == ord("P"):
            print("Pick Up Box")
            pick_up_box(box_pos_world[0], box_pos_world[1], box_pos_world[2])

        elif key == ord("R"):
            print("Reset to Zero")
            to_zero()
