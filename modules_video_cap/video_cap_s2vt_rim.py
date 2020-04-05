import pickle

from utils_cap import *

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util

class CapS2VT:

  @staticmethod
  def Setup():
    CapS2VT.n_steps = 80
    CapS2VT.batch_size = 1
    CapS2VT.vocab_size = len(word2id)
    CapS2VT.dropout = np.float32(1.0)

  def PreProcess(self, request, istub, grpc_flag):
    if (grpc_flag):
      self.input = pickle.loads(str(tensor_util.MakeNdarray(request.inputs["vgg_output"])))
    else:
      self.input = request["vgg_output"]

    self.istub = istub

  def Apply(self):
    if self.input['features'] is not None:
      self.vid, self.caption_GT, _ , _ = fetch_data_batch_inference(self.input['features'], self.input['meta'])
      self.caps, self.caps_mask = convert_caption(['<BOS>'], word2id, CapS2VT.n_steps)

      for i in range(CapS2VT.n_steps):
        internal_request = predict_pb2.PredictRequest()
        internal_request.model_spec.name = 's2vt'
        internal_request.model_spec.signature_name = 'predict_images'
        internal_request.inputs['input_video'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.vid, shape=self.vid.shape))
        internal_request.inputs['input_caption'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.caps, shape=self.caps.shape, dtype=np.int32))
        internal_request.inputs['input_caption_mask'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.caps_mask, shape=self.caps_mask.shape, dtype=np.float32))
        internal_request.inputs['input_dropout_prob'].CopyFrom(
            tf.contrib.util.make_tensor_proto(CapS2VT.dropout, dtype=np.float32))

        internal_result = self.istub.Predict(internal_request, 10.0)

        o_l = tensor_util.MakeNdarray(internal_result.outputs['output_logits'])
        out_logits = o_l.reshape([CapS2VT.batch_size, CapS2VT.n_steps - 1, CapS2VT.vocab_size])
        output_captions = np.argmax(out_logits, 2)
        self.caps[0][i + 1] = output_captions[0][i]

        if id2word[output_captions[0][i]] == '<EOS>':
          break

      self.caption = ' '.join(print_in_english(self.caps))

    else:
      self.caption = "None"

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["FINAL"].CopyFrom(
        tf.make_tensor_proto(self.caption))
    else:
      next_request = dict()
      next_request["FINAL"] = self.caption
    return next_request
