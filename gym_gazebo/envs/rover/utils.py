import numpy as np
from geometry_msgs.msg import Quaternion


def quaternion_to_euler(q, only_yaw = False):
    """
    convert from quaternion (x,y,z,w) to euler angles (roll,pitch,yaw) between [-pi,pi]
    """
    if not only_yaw: 
        roll = np.arctan2(2*(q.w*q.x + q.y*q.z), 1 - 2*(q.x**2 + q.y**2))
        pitch = np.arcsin(2*(q.w*q.y + q.z*q.x))
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
        return [roll,pitch,yaw]
    else:
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
        return yaw
