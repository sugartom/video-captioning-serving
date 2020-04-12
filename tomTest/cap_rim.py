import os
import time
import pickle
import cv2
import threading

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import threading
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/')

from modules_video_cap.video_cap_vgg16_rim import CapVGG16
from modules_video_cap.video_cap_alexnet_rim import CapAlexnet
from modules_video_cap.video_cap_s2vt_rim import CapS2VT

vgg = CapVGG16()
vgg.Setup()

alexnet = CapAlexnet()
alexnet.Setup()

# first = vgg
first = alexnet

s2vt =CapS2VT()
s2vt.Setup()

ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

video_path = "/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/inputs/vid264.mp4"
reader = cv2.VideoCapture(video_path)

frame_id = 1

features_fc7 = []
my_lock = threading.Lock()

total = 0.0
count = 0

while (frame_id < 250):
  start = time.time()

  _, image = reader.read()

  request = dict()

  request["client_input"] = image

  first.PreProcess(request = request, istub = istub, features_fc7 = features_fc7, my_lock = my_lock, grpc_flag = False)
  first.Apply()
  next_request = first.PostProcess(grpc_flag = False)

  # print(next_request["vgg_output"])

  s2vt.PreProcess(request = next_request, istub = istub, grpc_flag = False)
  s2vt.Apply()
  next_request = s2vt.PostProcess(grpc_flag = False)

  # if (next_request["FINAL"] != "None"):
  #   print(next_request["FINAL"])

  end = time.time()

  duration = end - start
  print("duration = %f" % duration)
  if (frame_id > 5):
    count += 1
    total += duration

  frame_id += 1

print("on average, it takes %f sec per frame" % (total / count))