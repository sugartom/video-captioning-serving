import pickle
import cv2

import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util

class CapAlexnet:

  @staticmethod
  def Setup():
    CapAlexnet.image_size = 227
    CapAlexnet.keep_prob = np.float32(1.0)
    CapAlexnet.imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

  def GetDataDict(self, request):
    data_dict = dict()

    # client_input
    image = tensor_util.MakeNdarray(request.inputs["client_input"])

    image = cv2.resize(image.astype(np.float32), (CapAlexnet.image_size, CapAlexnet.image_size))
    image -= CapAlexnet.imagenet_mean
    image = image.reshape((1, CapAlexnet.image_size, CapAlexnet.image_size, 3))

    data_dict["client_input"] = image

    return data_dict

  def GetBatchedDataDict(self, data_array, batch_size):
    if (len(data_array) != batch_size):
      print("[Error] batch size not matched...")
      return None

    else:
      batched_data_dict = dict()

      # client_input
      batched_data_dict["client_input"] = data_array[0]["client_input"]
      for data in data_array[1:]:
        batched_data_dict["client_input"] = np.append(batched_data_dict["client_input"], data["client_input"], axis = 0)

      return batched_data_dict

  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict["client_input"])):
      print("[Error] batch size not matched...")
      return None
    else:
      client_input = batched_data_dict["client_input"]

      internal_request = predict_pb2.PredictRequest()
      internal_request.model_spec.name = 'cap_alexnet'
      internal_request.model_spec.signature_name = 'predict_images'
      internal_request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(client_input, shape = client_input.shape))
      internal_request.inputs['input_keep_prob'].CopyFrom(
        tf.contrib.util.make_tensor_proto(CapAlexnet.keep_prob, dtype=np.float32))

      internal_result = istub.Predict(internal_request, 10.0)

      feature = tensor_util.MakeNdarray(internal_result.outputs['output'])

      batched_result_dict = dict()
      batched_result_dict["feature"] = feature

      return batched_result_dict

  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict["feature"])):
      print("[Error] batch size not matched...")
      return None
    else:
      batched_result_array = []
      for i in range(batch_size):
        my_dict = dict()
        my_dict["feature_fc7"] = [batched_result_dict["feature"][i].reshape(-1, batched_result_dict["feature"][i].shape[-1])]
        batched_result_array.append(my_dict)
      return batched_result_array

  def GetResultList(self, result_dict):
    result_list = []
    for feature_fc7 in result_dict["feature_fc7"]:
      result_list.append({"feature_fc7": feature_fc7})
    return result_list

  def GetNextRequest(self, result):
    next_request = predict_pb2.PredictRequest()
    next_request.inputs["feature_fc7"].CopyFrom(
        tf.make_tensor_proto(result["feature_fc7"]))
    return next_request
