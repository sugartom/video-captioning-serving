import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath('./modules_video_cap/VGG_tensorflow/research/slim/'))
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg
from preprocessing import vgg_preprocessing

# Path to actual model
VGG16_CKPT = os.path.abspath('./modules_video_cap/vggnet/model/vgg_16.ckpt')

IMAGE_SIZE = 224
N_DIMS = 80

class VGG16:
  def __init__(self):

    self.input = None

    # -- hyper settings
    self.image_size = IMAGE_SIZE
    self.out_dims = N_DIMS

    # net input
    self.image = tf.placeholder(tf.uint8, shape=(None, None, 3))

    # output
    self.features_fc7 = []

  def Setup(self):
    
    print(self.image.shape)
    image_pre = vgg_preprocessing.preprocess_image(self.image, self.image_size, self.image_size, is_training=False)
    print(image_pre.shape)
    self.image_4d = tf.expand_dims(image_pre, 0)
    print(self.image_4d.shape)

    # net forward
    with slim.arg_scope(vgg.vgg_arg_scope()):
    #     1000 classes instead of 1001.
       _, _ = vgg.vgg_16(self.image_4d, num_classes=1000, is_training=False)
    
    self.init_fn = slim.assign_from_checkpoint_fn(
        VGG16_CKPT,
        slim.get_model_variables('vgg_16'))

    # self.vgg16_graph = tf.get_default_graph()

    # net output
    self.fc7 = tf.get_default_graph().get_tensor_by_name("vgg_16/fc7/Relu:0")

    self.sess = tf.Session()

    # variables need to be initialized before any sess.run() calls
    # self.sess.run(tf.global_variables_initializer())
    self.init_fn(self.sess)

    self.log('init done ')

  def PreProcess(self, input):
    self.input = input

  def Apply(self):
    if not self.input:
      self.log('Input is empty')
      return 
    self.log("Image type {}, shape {}".format(type(self.input['img']), self.input['img'].shape))
    feature = self.sess.run([self.fc7], feed_dict={self.image: self.input['img']})
    # self.log("FC 7 type {}, shape {}".format(type(feature[0]), len(feature)))
    # self.log("FC 7 numpy shape {}".format(np.array(feature).shape))
    # np.save("fc7_pipeline" + '.npy',feature[0])
    if len(feature) != 0:
      self.feature_fc7 = feature[0].reshape(-1, feature[0].shape[-1])
    else:
      self.log('Error while extracting FC7 embedding')
      return 


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
  