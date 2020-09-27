import cv2
import time

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/')

from modules_video_cap.video_cap_alexnet_d2 import CapAlexnet

alexnet = CapAlexnet()
alexnet.Setup()

ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

video_path = "/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/inputs/vid264.mp4"
reader = cv2.VideoCapture(video_path)

frame_id = 1
batch_size = 16

while (frame_id < 256):

  start = time.time()

  data_array = []

  for i in range(batch_size):
    _, image = reader.read()

    request = predict_pb2.PredictRequest()
    request.inputs['client_input'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image, shape = image.shape))

    data_dict = alexnet.GetDataDict(request)
    data_array.append(data_dict)

    frame_id += 1

  batched_data_dict = alexnet.GetBatchedDataDict(data_array, batch_size)

  batched_result_dict = alexnet.Apply(batched_data_dict, batch_size, istub)

  batched_result_array = alexnet.GetBatchedResultArray(batched_result_dict, batch_size)

  for i in range(len(batched_result_array)):
    # deal with the outputs of the ith input in the batch
    result_dict = batched_result_array[i]

    result_list = alexnet.GetResultList(result_dict)

    for result in result_list:
      next_request = alexnet.GetNextRequest(result)

  end = time.time()
  print("duration for batch %d is %.6f" % (batch_size, (end - start)))
