
"""
Run the learned model to connect to client with ros messages
"""
import sys
import string 
import random
import numpy as np 
import argparse

from absl import app as absl_app
from absl import flags
from control.policy import Policy
from config import *

from keras.preprocessing.image import load_img, img_to_array

import cv2

import sys
sys.path.append('/mnt/intention_net')
sys.path.append('../utils')

# SCREEN SCALE IS FOR high dpi screen, i.e. 4K screen
SCREEN_SCALE = 1
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

flags_obj = None

def define_intention_net_flags():
    flags.DEFINE_enum(
            name='intention_mode', short_name='mode', default="DLM",
            enum_values=['DLM', 'LPE_SIAMESE', 'LPE_NO_SIAMESE'],
            help=help_wrap("Intention Net mode to run"))

    flags.DEFINE_enum(
            name='input_frame', short_name='input_frame', default="NORMAL",
            enum_values=['NORMAL', 'WIDE', 'MULTI'],
            help=help_wrap("Intention Net mode to run"))

def main(_):
    global flags_obj
    flags_obj = flags.FLAGS

    num_intentions = 4
    num_control = 2

    # Model inputs
    img = img_to_array(load_img(f"{flags_obj.data_dir}/images/X_0.png", target_size=(224, 224)))

    intention = img_to_array(load_img(f"{flags_obj.data_dir}/intentions/I_0.png", target_size=(224, 224)))
    speed = np.array([0])  # unimportant

    policy = Policy(flags_obj.mode, flags_obj.input_frame, num_control, flags_obj.model_dir, num_intentions)
    pred_control = policy.predict_control(img, intention, speed)
    
    print("Predicted control: ", pred_control)

if __name__ == '__main__':
    try:
        define_intention_net_flags()
        absl_app.run(main)
    except KeyboardInterrupt:
        print ('\nCancelled by user! Bye Bye!')
