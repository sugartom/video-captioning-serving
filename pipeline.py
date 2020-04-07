import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import sys
import os
import numpy as np
from modules_video_cap.utils_cap import *

sys.path.append(os.path.abspath('./'))

if (sys.argv[1] == "original"):
  # original version
  from modules_video_cap.data_reader import DataReader
  from modules_video_cap.video_cap_vgg16 import VGG16
  from modules_video_cap.video_cap_s2vt import S2VT

elif (sys.argv[1] == "serving"):
  # serving version 
  from modules_video_cap.data_reader import DataReader
  from modules_video_cap.video_cap_vgg16_serving import VGG16
  from modules_video_cap.video_cap_alexnet_serving import AlexNet
  from modules_video_cap.video_cap_s2vt_serving import S2VT

# ============ Video Input Module ============
video_path = os.path.abspath("/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/inputs/vid264.mp4")
reader = DataReader()
reader.Setup(video_path)

# # ============ VGG16 Embedding Module ===========
vgg16 = VGG16()
vgg16.Setup()

alexnet = AlexNet()
alexnet.Setup()

first = vgg16

# ============ S2VT Caption Module ===========
s2vt = S2VT()
s2vt.Setup()

while(True):

  # Read input
  frame_data = reader.PostProcess()
  if not frame_data:  # end of video 
    break

  first.PreProcess(frame_data)
  first.Apply()
  features_data = first.PostProcess()

  s2vt.PreProcess(features_data)
  s2vt.Apply()
  s2vt.PostProcess()

# ============ Play Video Module ============
play_video = raw_input('Play Video? ')
if play_video.lower() == 'y':
  playVideo(video_path)

  

    
