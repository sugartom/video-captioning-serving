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

measure_module = "cap_s2vt"

file_name = sys.argv[1]
video_path = "/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/inputs/%s.mp4" % file_name
reader = cv2.VideoCapture(video_path)

frame_id = 1

features_fc7 = []
my_lock = threading.Lock()

frame_id = sys.argv[2]

run_count = 0
duration_sum = 0

for i in range(10):
  pickle_input = "/home/yitao/Downloads/tmp/docker-share/pickle_tmp_combined/video-captioning-serving/pickle_tmp/cap_vgg/%s/%s" % (file_name, str(frame_id).zfill(3))
  f = open(pickle_input)
  next_request = pickle.load(f)

  start = time.time()

  s2vt.PreProcess(request = next_request, istub = istub, grpc_flag = False)
  s2vt.Apply()
  next_request = s2vt.PostProcess(grpc_flag = False)

  end = time.time()
  duration = end - start
  print("duration = %s" % duration)

  if (i > 5):
    duration_sum += duration
    run_count += 1

print("average duration = %f" % (duration_sum / run_count))
