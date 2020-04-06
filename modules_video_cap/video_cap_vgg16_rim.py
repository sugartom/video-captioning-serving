import pickle

import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util

class CapVGG16:

  @staticmethod
  def Setup():
    CapVGG16.image_size = 224
    CapVGG16.out_dims = 80
    # CapVGG16.features_fc7 = []

  def PreProcess(self, request, istub, features_fc7, my_lock, grpc_flag):
    if (grpc_flag):
      self.image = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      self.image = request["client_input"]

    self.istub = istub
    self.features_fc7 = features_fc7
    self.my_lock = my_lock

  def Apply(self):
    internal_request = predict_pb2.PredictRequest()
    internal_request.model_spec.name = 'cap_vgg'
    internal_request.model_spec.signature_name = 'predict_images'
    internal_request.inputs['input'].CopyFrom(
      tf.contrib.util.make_tensor_proto(self.image, shape=self.image.shape))

    internal_result = self.istub.Predict(internal_request, 10.0)

    feature = tensor_util.MakeNdarray(internal_result.outputs['output'])
    feature_fc7 = feature[0].reshape(-1, feature[0].shape[-1])

    self.my_lock.acquire()
    self.features_fc7.extend(feature_fc7)

    if (len(self.features_fc7) >= CapVGG16.out_dims):
      curr_feats = np.array(self.features_fc7)
      self.output = {'features': curr_feats, 'num_features': curr_feats.shape[0], 'meta': "vid264"}
      del self.features_fc7[:]
    else:
      self.output = {'features': None, 'num_features': 0, 'meta': None}

    self.my_lock.release()

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["vgg_output"].CopyFrom(
        tf.make_tensor_proto(pickle.dumps(self.output)))
    else:
      next_request = dict()
      next_request["vgg_output"] = self.output
    return next_request
