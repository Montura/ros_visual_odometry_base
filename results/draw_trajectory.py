#!/usr/bin/python
# -*- coding: utf8 -*-

from pylab import *
import os


class Pose:
    def __init__(self, timestamp, coord_x, coord_y, coord_z, rot_x, rot_y, rot_z, rot_w):
        self.t = int(timestamp)
        self.t = [float(coord_x), float(coord_y), float(coord_z)]
        self.q = [float(rot_x), float(rot_y), float(rot_z), float(rot_w)]


def create_dict(filename):
    pose = []
    with open(filename, 'r') as file:
        content = file.readlines()
        for line in content:
            split_line = line.split()
            pose.append([split_line[0], Pose(split_line[0],
                                             split_line[1],
                                             split_line[2],
                                             split_line[3],
                                             split_line[4],
                                             split_line[5],
                                             split_line[6],
                                             split_line[7])])
    return pose


def read_data(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
        line = [x.split() for x in content]
    return line


def get_pose(container):
    return [sublist[1:4] for sublist in container], np.asarray([sublist[4:8] for sublist in odometry], dtype=float)


def get_x_y(translation):
    return np.asarray([sublist[0:1][0] for sublist in translation], dtype=float),\
           np.asarray([sublist[1:2][0] for sublist in translation], dtype=float)


def get_x_z(translation):
    return np.asarray([sublist[0:1][0] for sublist in translation], dtype=float),\
           np.asarray([sublist[2:3][0] for sublist in translation], dtype=float)


def change_bad_values(coord_x, coord_y, ideal_coord):
    x_nan_positions = [i for i, x in enumerate(coord_x) if (str(x) == 'nan') or (np.abs(x) > 2)]
    y_nan_positions = [i for i, x in enumerate(coord_y) if (str(x) == 'nan') or (np.abs(x) > 2)]
    ideal_tr_values = [sublist[1].t for sublist in ideal_coord]
    ideal_x = [x[0] for x in ideal_tr_values]
    ideal_y = [x[1] for x in ideal_tr_values]
    # print coord_x
    for i in range(0, len(coord_x)):
        if i in x_nan_positions:
            coord_x[i] = ideal_x[i]
    for i in range(0, len(coord_y)):
        if i in y_nan_positions:
            coord_y[i] = ideal_y[i]
    # print ideal_x
    # print coord_x
    return coord_x, coord_y


#path_to_write = "/home/montura/ROS_Workaspace/catkin_ws/src/hello_world/src/"
#for directory in sorted(os.listdir(path_to_write)):
#    if "sift" in directory or "surf" in directory or "fast" in directory:
#        path_to_files = path_to_write + directory + '/'
#        method = directory

# Draw trajectory from data from visual odometry algorithm

path_to_files = "/home/montura/ROS_Workaspace/catkin_ws/src/hello_world/src/"
file_names = []
for file_name in sorted(os.listdir(path_to_files)):
    if ".txt" in file_name:
        file_names += [path_to_files + file_name]
    odometry = read_data(file_names[0])
    robot = read_data(file_names[1])
    visual_0 = read_data(file_names[2])
    visual_1 = read_data(file_names[3])

    odometry_tr, odometry_rot = get_pose(odometry)
    robot_tr, robot_rot = get_pose(robot)
    visual_0_tr, visual_0_rot = get_pose(visual_0)
    visual_1_tr, visual_1_rot = get_pose(visual_1)

    odometry_x, odometry_y = get_x_y(odometry_tr)
    robot_x, robot_y = get_x_y(robot_tr)
    visual_0_x, visual_0_y = get_x_z(visual_0_tr)
    visual_1_x, visual_1_y = get_x_y(visual_1_tr)

    for i in range(1, len(odometry_x)):
        odometry_x[i] += odometry_x[i - 1]
        odometry_y[i] += odometry_y[i - 1]
        robot_x[i] += robot_x[i - 1]
        robot_y[i] += robot_y[i - 1]

    fig, ax = plt.subplots()
    # ax.plot(odometry_x, odometry_y, 'r', label='Odometry')
    ax.plot(robot_x, robot_y, 'r', label='groundtruth')
    ax.plot(visual_0_x, visual_0_y, 'g', label='estimated')
    # ax.plot(visual_1_x, visual_1_y, 'b', label='estimated_K')
    xlabel('Coordinate X')
    ylabel('Coordinate Y')
    title('Robot trajectory')
    # if "26" in method:
    #     title('Robot trajectory for 2011-01-25-06-29-26.bag with ' + method.split('_')[0])
    # else:
    #     title('Robot trajectory for 2011-01-24-06-18-27.bag with ' + method.split('_')[0])
    legend = ax.legend(loc='upper right', shadow=True)
    # savefig(path_to_write + "/img/" + method + "_trajectory.png")
    plt.show()


# # Plot per coordinates X and Y
# fig, ax = plt.subplots()
# ax.plot(odometry_x, 'r', label='Odometry')
# ax.plot(robot_x, 'b', label='Robot_EKF')
# xlabel('time(sec)')
# ylabel('Coordinate X')
# title('X(time) for 2011-01-24-06-18-27.bag')
# legend = ax.legend(loc='upper right', shadow=True)
# plt.show()


# fig, ax = plt.subplots()
# ax.plot(odometry_y, 'r', label='Odometry')
# ax.plot(robot_y, 'b', label='Robot_EKF')
# xlabel('time(sec)')
# ylabel('Coordinate Y')
# title('Y(time) for 2011-01-24-06-18-27.bag')
# legend = ax.legend(loc='upper right', shadow=True)
# plt.show()
