import os
import sys
import tensorflow as tf
import numpy as np
import cv2

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

sys.path.append(os.path.abspath('./modules_video_cap/VGG_tensorflow/research/slim/'))
print(os.path.abspath('./modules_video_cap/VGG_tensorflow/research/slim/'))
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

  def extract_feats(self, input_image):
    image_size = vgg.vgg_16.default_image_size
    # checkpoints_dir="/home/rajrupgh/Project/video-captioning"
    with tf.Graph().as_default():
        image = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # url = 'https://upload.wikimedia.org/wikipedia/commons/d/d9/First_Student_IC_school_bus_202076.jpg'
        # image_string = urllib.urlopen(url).read()
        # image = tf.image.decode_jpeg(image_string, channels=3)
        # print(image.shape)
        # print(image.dtype)
        processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        print(processed_image.shape)
        processed_images  = tf.expand_dims(processed_image, 0)
        print(processed_images.shape)
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(vgg.vgg_arg_scope()):
            # 1000 classes instead of 1001.
            logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
        probabilities = tf.nn.softmax(logits)
        
        init_fn = slim.assign_from_checkpoint_fn(
            VGG16_CKPT,
            slim.get_model_variables('vgg_16'))
        
        tf_fc7 = tf.get_default_graph().get_tensor_by_name("vgg_16/fc7/Relu:0")
        
        with tf.Session() as sess:
            init_fn(sess)
            print(input_image.shape)
            print(input_image.dtype)
            probabilities, feature = sess.run([probabilities, tf_fc7], feed_dict={image: input_image})
            np.save("fc7_general" + '.npy',feature)
            print(type(feature))
            print(feature.shape)
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

            
        # plt.figure()
        # plt.imshow(np_image.astype(np.uint8))
        # plt.axis('off')
        # plt.show()
        
        # names = imagenet.create_readable_names_for_imagenet_labels()
        # for i in range(5):
        #     index = sorted_inds[i]
        #     # Shift the index of a class name by one. 
        #     print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index+1]))

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

  def Preprocess(self, input):
    self.input = input

  def Apply(self):
    if not self.input:
      self.log('Input is empty')
      return 
    self.log("Image type {}, shape {}".format(type(self.input['img']), self.input['img'].shape))
    feature = self.sess.run([self.fc7], feed_dict={self.image: self.input['img']})
    self.log("FC 7 type {}, shape {}".format(type(feature), len(feature)))
    self.log("FC 7 numpy shape {}".format(np.array(feature).shape))
    # self.feature_fc7 = feature.reshape(-1, feature.shape[-1])


  def Postprocess(self):

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

if __name__ == '__main__':
  vgg16 = VGG16()
  path = "/home/rajrupgh/Project/video-captioning/Data/YouTubeClips/"
  filenames = [os.path.join(path, l) for l in os.listdir(path)]
  # vgg16.extract_feats(None, filenames, 16)

  cap = cv2.VideoCapture(os.path.abspath("./modules_video_cap/Data/YoutubeClips/vid264.mp4"))

  ret, frame = cap.read()

  if ret:
    vgg16.extract_feats(frame)