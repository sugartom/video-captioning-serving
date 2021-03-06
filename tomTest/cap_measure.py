import os
import time
import pickle
import cv2
import threading
import sys

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import threading
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/')

from modules_video_cap.video_cap_vgg16_rim import CapVGG16
from modules_video_cap.video_cap_s2vt_rim import CapS2VT

vgg = CapVGG16()
vgg.Setup()

s2vt =CapS2VT()
s2vt.Setup()

ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

simple_route_table = "cap_vgg-cap_s2vt"
route_table = simple_route_table

measure_module = "cap_vgg"

file_name = sys.argv[1]
video_path = "/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/inputs/%s.mp4" % file_name
reader = cv2.VideoCapture(video_path)

frame_id = 1

features_fc7 = []
my_lock = threading.Lock()

while (frame_id < 250):
  _, image = reader.read()

  request = dict()

  request["client_input"] = image

  vgg.PreProcess(request = request, istub = istub, features_fc7 = features_fc7, my_lock = my_lock, grpc_flag = False)
  vgg.Apply()
  next_request = vgg.PostProcess(grpc_flag = False)

  # print(next_request["vgg_output"])

  if (frame_id == 80 or frame_id == 160 or frame_id == 240):
    pickle_output = "/home/yitao/Downloads/tmp/docker-share/pickle_tmp_combined/video-captioning-serving/pickle_tmp/cap_vgg/%s/%s" % (file_name, str(frame_id).zfill(3))
    with open(pickle_output, 'w') as f:
      pickle.dump(next_request, f)

  # s2vt.PreProcess(request = next_request, istub = istub, grpc_flag = False)
  # s2vt.Apply()
  # next_request = s2vt.PostProcess(grpc_flag = False)

  # if (next_request["FINAL"] != "None"):
  #   print(next_request["FINAL"])

  frame_id += 1
