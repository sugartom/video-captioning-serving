import os
import sys
import numpy as np
import tensorflow as tf
from utils import *

# Path to actual model
S2VT_CKPT = '/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/modules_video_cap/s2vt/model/S2VT_Dyn_10_0.0001_300_46000.ckpt'
S2VT_CKPT_META = '/home/yitao/Documents/fun-project/tensorflow-related/video-captioning-serving/modules_video_cap/s2vt/model/S2VT_Dyn_10_0.0001_300_46000.ckpt.meta'

N_STEPS = 80
HIDDEN_DIM = 500
FRAME_DIM = 4096
BATCH_SIZE = 1
LEARNING_RATE = 0.00001
DROPOUT=1.0

class S2VT:
  def __init__(self):
    # VARIABLE INITIALIZATIONS TO BUILD MODEL
    self.n_steps = N_STEPS
    self.hidden_dim = HIDDEN_DIM
    self.frame_dim = FRAME_DIM
    self.batch_size = BATCH_SIZE
    self.vocab_size = len(word2id)
    self.bias_init_vector = get_bias_vector()
    self.learning_rate = LEARNING_RATE
    self.dropout = np.float32(DROPOUT)

  def build_model(self):
    """This function creates weight matrices that transform:
        * frames to caption dimension
        * hidden state to vocabulary dimension
        * creates word embedding matrix """

    self.log("Network config: \nN_Steps: {}\nHidden_dim:{}\nFrame_dim:{}\nBatch_size:{}\nVocab_size:{}\n".format(self.n_steps,
                                                                                                    self.hidden_dim,
                                                                                                    self.frame_dim,
                                                                                                    self.batch_size,
                                                                                                    self.vocab_size))

    #Create placeholders for holding a batch of videos, captions and caption masks
    video = tf.placeholder(tf.float32,shape=[self.batch_size,self.n_steps,self.frame_dim],name='Input_Video')
    caption = tf.placeholder(tf.int32,shape=[self.batch_size,self.n_steps],name='GT_Caption')
    caption_mask = tf.placeholder(tf.float32,shape=[self.batch_size,self.n_steps],name='Caption_Mask')
    dropout_prob = tf.placeholder(tf.float32,name='Dropout_Keep_Probability')

    with tf.variable_scope('Im2Cap') as scope:
        W_im2cap = tf.get_variable(name='W_im2cap',shape=[self.frame_dim,
                                                    self.hidden_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
        b_im2cap = tf.get_variable(name='b_im2cap',shape=[self.hidden_dim],
                                                    initializer=tf.constant_initializer(0.0))
    with tf.variable_scope('Hid2Vocab') as scope:
        W_H2vocab = tf.get_variable(name='W_H2vocab',shape=[self.hidden_dim,self.vocab_size],
                                                        initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
        b_H2vocab = tf.Variable(name='b_H2vocab',initial_value=self.bias_init_vector.astype(np.float32))

    with tf.variable_scope('Word_Vectors') as scope:
        word_emb = tf.get_variable(name='Word_embedding',shape=[self.vocab_size,self.hidden_dim],
                                                                initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))

    #Build two LSTMs, one for processing the video and another for generating the caption
    with tf.variable_scope('LSTM_Video',reuse=None) as scope:
        lstm_vid = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_vid = tf.nn.rnn_cell.DropoutWrapper(lstm_vid,output_keep_prob=dropout_prob)
    with tf.variable_scope('LSTM_Caption',reuse=None) as scope:
        lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cap = tf.nn.rnn_cell.DropoutWrapper(lstm_cap,output_keep_prob=dropout_prob)

    #Prepare input for lstm_video
    video_rshp = tf.reshape(video,[-1,self.frame_dim])
    video_rshp = tf.nn.dropout(video_rshp,keep_prob=dropout_prob)
    video_emb = tf.nn.xw_plus_b(video_rshp,W_im2cap,b_im2cap)
    video_emb = tf.reshape(video_emb,[self.batch_size,self.n_steps,self.hidden_dim])
    padding = tf.zeros([self.batch_size,self.n_steps-1,self.hidden_dim])
    video_input = tf.concat([video_emb,padding],1)
    print "Video_input: {}".format(video_input.get_shape())
    #Run lstm_vid for 2*self.n_steps-1 timesteps
    with tf.variable_scope('LSTM_Video') as scope:
        out_vid,state_vid = tf.nn.dynamic_rnn(lstm_vid,video_input,dtype=tf.float32)
    print "Video_output: {}".format(out_vid.get_shape())

    #Prepare input for lstm_cap
    padding = tf.zeros([self.batch_size,self.n_steps,self.hidden_dim])
    caption_vectors = tf.nn.embedding_lookup(word_emb,caption[:,0:self.n_steps-1])
    caption_vectors = tf.nn.dropout(caption_vectors,keep_prob=dropout_prob)
    caption_2n = tf.concat([padding,caption_vectors],1)
    caption_input = tf.concat([caption_2n,out_vid],2)
    print "Caption_input: {}".format(caption_input.get_shape())
    #Run lstm_cap for 2*self.n_steps-1 timesteps
    with tf.variable_scope('LSTM_Caption') as scope:
        out_cap,state_cap = tf.nn.dynamic_rnn(lstm_cap,caption_input,dtype=tf.float32)
    print "Caption_output: {}".format(out_cap.get_shape())

    #Compute masked loss
    output_captions = out_cap[:,self.n_steps:,:]
    output_logits = tf.reshape(output_captions,[-1,self.hidden_dim])
    output_logits = tf.nn.dropout(output_logits,keep_prob=dropout_prob)
    output_logits = tf.nn.xw_plus_b(output_logits,W_H2vocab,b_H2vocab)
    output_labels = tf.reshape(caption[:,1:],[-1])
    caption_mask_out = tf.reshape(caption_mask[:,1:],[-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_logits,labels=output_labels)
    masked_loss = loss*caption_mask_out
    loss = tf.reduce_sum(masked_loss)/tf.reduce_sum(caption_mask_out)
    return video,caption,caption_mask,output_logits,loss,dropout_prob


  def Setup(self):
    
    self.s2vt_graph = tf.Graph()
    
    with self.s2vt_graph.as_default() as sg:
      self.video,self.caption,self.caption_mask,self.output_logits,self.loss,self.dropout_prob = self.build_model()
      self.optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

      self.log("Model loading...")
      self.saver = tf.train.Saver()
      self.saver = tf.train.import_meta_graph(S2VT_CKPT_META)
    
    self.sess = tf.Session(graph = self.s2vt_graph)
    self.saver.restore(self.sess, S2VT_CKPT)
    self.log("Restored model")

  def PreProcess(self, input):
    self.input = input
    if self.input['features'] is not None:

      self.vid,self.caption_GT,_,video_urls = fetch_data_batch_inference(self.input['features'], self.input['meta']['vid_name'])
      self.caps,self.caps_mask = convert_caption(['<BOS>'],word2id, self.n_steps)

  def Apply(self):

    self.output_captions = None
    if self.input['features'] is not None:

      for i in range(self.n_steps):
        o_l = self.sess.run(self.output_logits,feed_dict={self.video:self.vid,
                                                self.caption:self.caps,
                                                self.caption_mask:self.caps_mask,
                                                self.dropout_prob:self.dropout})
        out_logits = o_l.reshape([self.batch_size,self.n_steps-1,self.vocab_size])
        output_captions = np.argmax(out_logits,2)
        self.caps[0][i+1] = output_captions[0][i]
        if id2word[output_captions[0][i]] == '<EOS>':
          break

  def PostProcess(self):
    if self.input['features'] is not None:
      self.log('S2VT Caption:')
      print_in_english(self.caps)
      self.log('GT Caption:')
      print_in_english(self.caption_GT)

  def log(self, s):
    print('[S2VT] %s' % s)


