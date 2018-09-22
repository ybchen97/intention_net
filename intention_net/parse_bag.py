"""
Generate data from huawei rosbag
"""
import cv2
import fire
import rosbag
import glob
import shutil
import os
import csv
import os.path as osp
from tqdm import tqdm
from munch import Munch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32
from toolz import partition_all
from itertools import chain
# import local
from threadedgenerator import ThreadedGenerator

# Topics for register
CAMERA_IMG = '/image'
CAMERA_FRONT_96 = '/front_96_image'
CAMERA_LEFT_96 = '/fl_96_image'
CAMERA_RIGHT_96 = '/fr_96_image'
INTENTION_DLM = '/intention_dlm'
INTENTION_LPE = '/intention_lpe'
SPEED = '/speed'
CONTROL = '/labeled_control'
TOPICS = [CAMERA_IMG,
          CAMERA_FRONT_96,
          CAMERA_LEFT_96,
          CAMERA_RIGHT_96,
          INTENTION_DLM,
          INTENTION_LPE,
          SPEED,
          CONTROL]

# CHUNK_SIZE for parallel parsing
CHUNK_SIZE = 128

def imgmsg_to_cv2(msg):
    return CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')

def parse_bag(bagfn):
    bag = rosbag.Bag(bagfn)
    image = None
    left_96 = None
    right_96 = None
    front_96 = None
    intention_dlm = None
    intention_lpe = None
    speed = None
    acc = None
    steer = None
    start = False

    def gen(t):
        img = imgmsg_to_cv2(image)
        left_96_img = imgmsg_to_cv2(left_96)
        right_96_img = imgmsg_to_cv2(right_96)
        front_96_img = imgmsg_to_cv2(front_96)
        dlm = intention_dlm
        lpe = imgmsg_to_cv2(intention_lpe)
        return Munch(t=t, img=img, left_96_img=left_96_img,
                     right_96_img=right_96_img,
                     front_96_img=front_96_img,
                     dlm=dlm, lpe=lpe, speed=speed,
                     acc=acc, steer=steer)

    for topic, msg, t in bag.read_messages(topics=TOPICS):
        if 'fr_96' in topic:
            right_96 = msg
        elif 'fl_96' in topic:
            left_96 = msg
        elif 'front_96' in topic:
            front_96 = msg
        elif 'speed' in topic:
            speed = msg.data
        elif 'dlm' in topic:
            intention_dlm = msg.data
        elif 'lpe' in topic:
            intention_lpe = msg
        elif 'control' in topic:
            acc = msg.linear.x
            steer = msg.angular.z
        elif '/image' == topic:
            image = msg
            # publish at the same rate of image
            if start:
                yield gen(t)

        if image is not None and left_96 is not None and right_96 is not None and front_96 is not None and speed is not None and acc is not None and intention_dlm is not None and intention_lpe is not None:
            start = True

    bag.close()

def main_wrapper(data_dir):
    bagfns = glob.glob(data_dir + '/*.bag') 
    print ('bags:', bagfns)
    bags = chain(*[parse_bag(bagfn) for bagfn in bagfns])
    it = ThreadedGenerator(bags, queue_maxsize=6500)
    # make dirs for images
    shutil.rmtree(osp.join(data_dir, 'gendata'))
    os.mkdir(osp.join(data_dir, 'gendata'))
    os.mkdir(osp.join(data_dir, 'gendata', 'camera_img'))
    os.mkdir(osp.join(data_dir, 'gendata', 'camera_img', 'front_60'))
    os.mkdir(osp.join(data_dir, 'gendata', 'camera_img', 'front_96_left'))
    os.mkdir(osp.join(data_dir, 'gendata', 'camera_img', 'side_96_left'))
    os.mkdir(osp.join(data_dir, 'gendata', 'camera_img', 'side_96_right'))
    os.mkdir(osp.join(data_dir, 'gendata', 'intention_img'))
    f = open(osp.join(data_dir, 'gendata', 'LabelData_VehicleData_PRT.txt'), 'w')
    labelwriter = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    labelwriter.writerow(['intention_type', 'current_velocity', 'steering_wheel_angle'])
    for chunk in partition_all(CHUNK_SIZE, tqdm(it)):
        for c in chunk:
            cv2.imwrite(osp.join(data_dir, 'gendata', 'camera_img', 'front_60', '{}.jpg'.format(c.t)), c.img)
            cv2.imwrite(osp.join(data_dir, 'gendata', 'camera_img', 'front_96_left', '{}.jpg'.format(c.t)), c.img)
            cv2.imwrite(osp.join(data_dir, 'gendata', 'camera_img', 'side_96_left', '{}.jpg'.format(c.t)), c.img)
            cv2.imwrite(osp.join(data_dir, 'gendata', 'camera_img', 'side_96_right', '{}.jpg'.format(c.t)), c.img)
            cv2.imwrite(osp.join(data_dir, 'gendata', 'intention_img', '{}.jpg'.format(c.t)), c.img)
            print (c.acc, c.steer)

if __name__ == '__main__':
    fire.Fire(main_wrapper)