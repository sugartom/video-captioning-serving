import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath('./modules_video_cap/models/research/slim/'))
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg
from preprocessing import vgg_preprocessing

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util

IMAGE_SIZE = 224
N_DIMS = 80

class VGG16:

  def Setup(self):

    # input
    self.input = None

    # -- hyper settings
    self.image_size = IMAGE_SIZE
    self.out_dims = N_DIMS
    
    # output
    self.features_fc7 = []

  def PreProcess(self, input):
    self.input = input

  def Apply(self):
    ichannel = grpc.insecure_channel("localhost:8500")
    self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

    self.internal_request = predict_pb2.PredictRequest()
    self.internal_request.model_spec.name = 'vgg16'
    self.internal_request.model_spec.signature_name = 'predict_images'

    self.internal_request.inputs['input'].CopyFrom(
    tf.contrib.util.make_tensor_proto(self.input['img'], shape=self.input['img'].shape))

    self.internal_result = self.istub.Predict(self.internal_request, 10.0) # 10 sec timeout

    feature = tensor_util.MakeNdarray(self.internal_result.outputs['output'])
    self.feature_fc7 = feature[0].reshape(-1, feature[0].shape[-1])


  def PostProcess(self):

    self.features_fc7.extend(self.feature_fc7)
    
    if len(self.features_fc7) >= N_DIMS:
      curr_feats = np.array(self.features_fc7)
      output = {'features': curr_feats, 'num_features': curr_feats.shape[0], 'meta':self.input['meta']}
      self.features_fc7 = []
      return output
    else:
      return {'features': None, 'num_features': 0, 'meta': None}
      
  def log(self, s):
    print('[VGG16] %s' % s)
  