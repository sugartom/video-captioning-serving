# import pickle
import cv2
import threading

import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util

class CapSequencer:

  @staticmethod
  def Setup():
    CapSequencer.out_dims = 80

    CapSequencer.my_lock = threading.Lock()
    CapSequencer.features_fc7 = []

  def GetDataDict(self, request):
    data_dict = dict()

    feature_fc7 = tensor_util.MakeNdarray(request.inputs["feature_fc7"])

    data_dict["feature_fc7"] = feature_fc7

    return data_dict

  def GetBatchedDataDict(self, data_array, batch_size):
    if (len(data_array) != batch_size):
      print("[Error] batch size not matched...")
      return None

    else:
      batched_data_dict = dict()

      # feature_fc7
      batched_data_dict["feature_fc7"] = data_array[0]["feature_fc7"]
      for data in data_array[1:]:
        batched_data_dict["feature_fc7"] = np.append(batched_data_dict["feature_fc7"], data["feature_fc7"], axis = 0)

      return batched_data_dict

  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict["feature_fc7"])):
      print("[Error] batch size not matched...")
      return None
    else:
      feature_fc7 = batched_data_dict["feature_fc7"]

      batched_result_dict = dict()

      CapSequencer.my_lock.acquire()
      CapSequencer.features_fc7.extend(feature_fc7)

      if (len(CapSequencer.features_fc7) >= CapSequencer.out_dims):
        curr_feats = np.array(CapSequencer.features_fc7[:CapSequencer.out_dims])
        batched_result_dict["features"] = [curr_feats]
        batched_result_dict["num_features"] = [curr_feats.shape[0]]
        batched_result_dict["meta"] = ["vid264"]
        del CapSequencer.features_fc7[:CapSequencer.out_dims]
      else:
        batched_result_dict["features"] = [None]
        batched_result_dict["num_features"] = [0]
        batched_result_dict["meta"] = [None]

      CapSequencer.my_lock.release()

      return batched_result_dict

  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict["features"])):
      print("[Error] batch size not matched...")
      return None
    else:
      batched_result_array = []
      for i in range(batch_size):
        my_dict = dict()
        my_dict["features"] = [batched_result_dict["features"][i]]
        my_dict["num_features"] = [batched_result_dict["num_features"][i]]
        my_dict["meta"] = [batched_result_dict["meta"][i]]
        batched_result_array.append(my_dict)
      return batched_result_array

  def GetResultList(self, result_dict):
    result_list = []
    for i in range(len(result_dict["features"])):
      features = result_dict["features"][i]
      num_features = result_dict["num_features"][i]
      meta = result_dict["meta"][i]
      if (features is not None):
        result_list.append({"features": features, "num_features": num_features, "meta": meta})
    return result_list

  def GetNextRequest(self, result):
    next_request = predict_pb2.PredictRequest()
    next_request.inputs["features"].CopyFrom(
        tf.make_tensor_proto(result["features"]))
    next_request.inputs["num_features"].CopyFrom(
        tf.make_tensor_proto(result["num_features"]))
    next_request.inputs["meta"].CopyFrom(
        tf.make_tensor_proto(result["meta"]))
    return next_request
