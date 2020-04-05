import os
import sys
import numpy as np
import tensorflow as tf
from utils_cap import *

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util

N_STEPS = 80
HIDDEN_DIM = 500
FRAME_DIM = 4096
BATCH_SIZE = 1
LEARNING_RATE = 0.00001
DROPOUT = 1.0

class S2VT:

  def Setup(self):

    # VARIABLE INITIALIZATIONS TO BUILD MODEL
    self.n_steps = N_STEPS
    self.hidden_dim = HIDDEN_DIM
    self.frame_dim = FRAME_DIM
    self.batch_size = BATCH_SIZE
    self.vocab_size = len(word2id)
    self.learning_rate = LEARNING_RATE
    self.dropout = np.float32(DROPOUT)
    self.bias_init_vector = get_bias_vector()

  def PreProcess(self, input):
    self.input = input

    if self.input['features'] is not None:

      self.vid,self.caption_GT,_,video_urls = fetch_data_batch_inference(self.input['features'], self.input['meta']['vid_name'])
      self.caps,self.caps_mask = convert_caption(['<BOS>'],word2id, self.n_steps)

  def Apply(self):

    self.output_captions = None
    if self.input['features'] is not None:

      for i in range(self.n_steps):

        ichannel = grpc.insecure_channel("localhost:8500")
        self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

        self.internal_request = predict_pb2.PredictRequest()
        self.internal_request.model_spec.name = 's2vt'
        self.internal_request.model_spec.signature_name = 'predict_images'
        self.internal_request.inputs['input_video'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.vid, shape=self.vid.shape))
        self.internal_request.inputs['input_caption'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.caps, shape=self.caps.shape, dtype=np.int32))
        self.internal_request.inputs['input_caption_mask'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.caps_mask, shape=self.caps_mask.shape, dtype=np.float32))
        self.internal_request.inputs['input_dropout_prob'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.dropout, dtype=np.float32))

        self.internal_result = self.istub.Predict(self.internal_request, 10.0)
        o_l = tensor_util.MakeNdarray(self.internal_result.outputs['output_logits'])
        out_logits = o_l.reshape([self.batch_size,self.n_steps-1,self.vocab_size])
        output_captions = np.argmax(out_logits,2)
        self.caps[0][i+1] = output_captions[0][i]
        if id2word[output_captions[0][i]] == '<EOS>':
          break

  def PostProcess(self):
    if self.input['features'] is not None:
      self.log('S2VT Caption:')
      print(print_in_english(self.caps))
      self.log('GT Caption:')
      print(print_in_english(self.caption_GT))

  def log(self, s):
    print('[S2VT] %s' % s)


