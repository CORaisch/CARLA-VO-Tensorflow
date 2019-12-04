#!/usr/bin/env python

# sample call: python kitti_to_carla_euler.py --poses PATH/TO/KITTI_POSES_FILE

import argparse, sys, math
import numpy as np


def invert_pose(T):
    T_inv = T.copy()
    R = T[:3, :3]; t = T[:3, 3];
    T_inv[:3, :3] = R.T
    T_inv[:3,  3] = -R.T * t
    return T_inv

def extract_euler_from_rotation(R):
    # NOTE code taken from: https://www.learnopencv.com/rotation-matrix-to-euler-angles
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def pose_to_string(euler_pose):
    # NOTE expecting 6d array
    ret  = str(euler_pose[0]) + " " + str(euler_pose[1]) + " " + str(euler_pose[2]) + " "
    ret += str(euler_pose[3]) + " " + str(euler_pose[4]) + " " + str(euler_pose[5]) + "\n"
    return ret

def main():
    # setup argparser
    argparser = argparse.ArgumentParser(description="Converts absolute KITTI poses to relative Euler poses, ready for mixed training with CARLA sequences using 'train_sequence.py'.")
    argparser.add_argument('--poses', '-p', type=str, help="path to absolute KITTI poses")
    args = argparser.parse_args()

    # read in absolute poses as numpy matrices
    filename = args.poses
    f = open(filename, 'r'); lines = f.readlines(); f.close();
    poses_np = []
    for line in lines:
        l = [float(x.replace('\n', '')) for x in line.split(' ')]
        poses_np.append(np.matrix([[l[0], l[1], l[2], l[3]], [l[4], l[5], l[6], l[7]], [l[8], l[9], l[10], l[11]], [0.0, 0.0, 0.0, 1.0]]))

    # give status information
    print("{0} absolute KITTI poses have been parsed".format(len(poses_np)))

    # iterate absolute poses, concatenate relative poses for each frame and extract euler angles
    T_last = poses_np.pop(0).copy()
    poses_str = pose_to_string([T_last[2,3], -T_last[0,3], -T_last[1,3], 0.0, 0.0, 0.0]) # 1st pose should always be identity
    for pose in poses_np:
        # invert last pose
        T_last_inv = invert_pose(T_last)
        T_rel = T_last_inv * pose
        # extract euler angles from relative transform
        eulers = extract_euler_from_rotation(T_rel[:3,:3])
        # compose euler pose in carla coordinate system
        euler_pose = [T_rel[2,3], -T_rel[0,3], -T_rel[1,3], eulers[2], -eulers[0], -eulers[1]]
        # write relative euler pose to file
        poses_str += pose_to_string(euler_pose)
        # update last transform
        T_last = pose.copy()

    # write relative poses to file
    filename_out = filename.replace('.txt', '_relative_euler_carla.txt')
    with open(filename_out, 'w') as abs_poses_file:
        abs_poses_file.write(poses_str)

    # give status report
    print("converted absoulute poses written to '{0}'".format(filename_out))

if __name__ == "__main__":
    main()
