import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath('./modules_video_cap/VGG_tensorflow/research/slim/'))
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg
from preprocessing import vgg_preprocessing

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util

# # Path to actual model
# VGG16_CKPT = os.path.abspath('./modules_video_cap/vggnet/model/vgg_16.ckpt')

IMAGE_SIZE = 224
N_DIMS = 80

class VGG16:
  def __init__(self):

    self.input = None

    # -- hyper settings
    self.image_size = IMAGE_SIZE
    self.out_dims = N_DIMS

    # # net input
    # self.image = tf.placeholder(tf.uint8, shape=(None, None, 3))

  def Setup(self):

    # output
    self.features_fc7 = []

    # image_pre = vgg_preprocessing.preprocess_image(self.image, self.image_size, self.image_size, is_training=False)
    # self.image_4d = tf.expand_dims(image_pre, 0)

    # # net forward
    # with slim.arg_scope(vgg.vgg_arg_scope()):
    # #     1000 classes instead of 1001.
    #    _, _ = vgg.vgg_16(self.image_4d, num_classes=1000, is_training=False)
    
    # self.init_fn = slim.assign_from_checkpoint_fn(
    #     VGG16_CKPT,
    #     slim.get_model_variables('vgg_16'))

    # self.sess = tf.Session()

    # self.vgg16_graph = tf.get_default_graph()

    # # variables need to be initialized before any sess.run() calls
    # self.sess.run(tf.global_variables_initializer())
    # self.init_fn(self.sess)

    # # net output
    # self.fc7 = tf.get_default_graph().get_tensor_by_name("vgg_16/fc7/Relu:0")

    # self.log('init done ')

  def PreProcess(self, input):
    self.input = input

  def Apply(self):
    # if not self.input:
    #   self.log('Input is empty')
    #   return 


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
  