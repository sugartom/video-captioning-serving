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

  def GetDataDict(self, request):
    data_dict = dict()

    features = tensor_util.MakeNdarray(request.inputs["features"])
    # num_features = tensor_util.MakeNdarray(request.inputs["num_features"])
    meta = tensor_util.MakeNdarray(request.inputs["meta"])

    data_dict["features"] = features
    # data_dict["num_features"] = num_features
    data_dict["meta"] = meta

    print(features)
    print(meta)

    return data_dict

  def GetBatchedDataDict(self, data_array, batch_size):
    if (len(data_array) != batch_size):
      print("[Error] GetBatchedDataDict() batch size not matched...")
      return None

    else:
      batched_data_dict = dict()

      # features
      batched_data_dict["features"] = [data_array[0]["features"]]
      # for data in data_array[1:]:
      #   batched_data_dict["features"] = np.append(batched_data_dict["features"], data["features"], axis = 0)

      # meta
      batched_data_dict["meta"] = [data_array[0]["meta"]]
      # for data in data_array[1:]:
      #   batched_data_dict["meta"] = np.append(batched_data_dict["meta"], data["meta"], axis = 0)

      return batched_data_dict

  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict["features"])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      features = batched_data_dict["features"][0]
      meta = batched_data_dict["meta"][0]

      vid, _, _, _ = fetch_data_batch_inference(features, meta)
      caps, caps_mask = convert_caption(['<BOS>'], word2id, CapS2VT.n_steps)

      for i in range(CapS2VT.n_steps):
        internal_request = predict_pb2.PredictRequest()
        internal_request.model_spec.name = 'cap_s2vt'
        internal_request.model_spec.signature_name = 'predict_images'
        internal_request.inputs['input_video'].CopyFrom(
            tf.contrib.util.make_tensor_proto(vid, shape = vid.shape))
        internal_request.inputs['input_caption'].CopyFrom(
            tf.contrib.util.make_tensor_proto(caps, shape = caps.shape, dtype=np.int32))
        internal_request.inputs['input_caption_mask'].CopyFrom(
            tf.contrib.util.make_tensor_proto(caps_mask, shape = caps_mask.shape, dtype=np.float32))
        internal_request.inputs['input_dropout_prob'].CopyFrom(
            tf.contrib.util.make_tensor_proto(CapS2VT.dropout, dtype=np.float32))

        internal_result = istub.Predict(internal_request, 10.0)

        o_l = tensor_util.MakeNdarray(internal_result.outputs['output_logits'])
        out_logits = o_l.reshape([CapS2VT.batch_size, CapS2VT.n_steps - 1, CapS2VT.vocab_size])
        output_captions = np.argmax(out_logits, 2)
        caps[0][i + 1] = output_captions[0][i]

        if id2word[output_captions[0][i]] == '<EOS>':
          break

      caption = ' '.join(print_in_english(caps))

      batched_result_dict["caption"] = [caption]

      return batched_result_dict

  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict["caption"])):
      print("[Error] GetBatchedResultArray() batch size not matched...")
      return None
    else:
      batched_result_array = []
      for i in range(batch_size):
        my_dict = dict()
        my_dict["caption"] = [batched_result_dict["caption"][i]]
        batched_result_array.append(my_dict)
      return batched_result_array

  def GetResultList(self, result_dict):
    result_list = []
    for i in range(len(result_dict["caption"])):
      caption = result_dict["caption"][i]
      result_list.append({"caption": caption})
    return result_list

  def GetNextRequest(self, result):
    next_request = predict_pb2.PredictRequest()
    next_request.inputs["caption"].CopyFrom(
        tf.make_tensor_proto(result["caption"]))
    return next_request
