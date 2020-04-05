#!/usr/bin/python
import os
import numpy as np
import tensorflow as tf
from preprocess import *
import cv2

"""Utilities for training the video captioning system"""

#Global initializations
n_lstm_steps = 80
DATA_DIR = os.path.abspath('./modules_video_cap/Data/')
# VIDEO_DIR = DATA_DIR + 'Features_VGG/'
# VIDEO_DIR = DATA_DIR + 'Features_Alexnet_tf/'
YOUTUBE_CLIPS_DIR = DATA_DIR + '/YouTubeClips/'
TEXT_DIR = os.path.abspath('./modules_video_cap/text_files')
Vid2Url = eval(open(TEXT_DIR + '/Vid2Url_Full.txt').read())
Vid2Cap_train = eval(open(TEXT_DIR + '/Vid2Cap_train.txt').read())
Vid2Cap_val = eval(open(TEXT_DIR + '/Video2Caption_mytest.txt').read())
word_counts,unk_required = build_vocab(0)
word2id,id2word = word_to_word_ids(word_counts,unk_required)
video_files = Vid2Cap_train.keys()
val_files = Vid2Cap_val.keys()

# log("{0} files processed".format(len(video_files)))

def get_bias_vector():
    """Function to return the initialization for the bias vector
       for mapping from hidden_dim to vocab_size.
       Borrowed from neuraltalk by Andrej Karpathy"""
    bias_init_vector = np.array([1.0*word_counts[id2word[i]] for i in id2word])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return bias_init_vector


def fetch_data_batch_inference(vid_features, vid_name):
    """Function to fetch a batch of video features from the validation set and its captions.
        Input:
                batch_size: Size of batch to load
        Output:
                curr_vids: Features of the randomly selected batch of video_files
                curr_caps: Ground truth (padded) captions for the selected videos"""

    curr_batch_vids = [str(vid_name)]
    curr_vids = np.expand_dims(vid_features, axis = 0)
    video_urls = [vid for vid in curr_batch_vids]
    ind_50 = map(int,np.linspace(0,79,n_lstm_steps))
    curr_vids = curr_vids[:,ind_50,:]
    captions = [np.random.choice(Vid2Cap_val[vid],1)[0] for vid in curr_batch_vids]
    curr_caps,curr_masks = convert_caption(captions,word2id,n_lstm_steps)
    return curr_vids,curr_caps,curr_masks,video_urls

def print_in_english(caption_idx):
    """Function to take a list of captions with words mapped to ids and
        print the captions after mapping word indices back to words."""
    captions_english = [[id2word[word] for word in caption] for caption in caption_idx]
    for i,caption in enumerate(captions_english):
        if '<EOS>' in caption:
            caption = caption[1:caption.index('<EOS>')] # Skipping the starting word "<BOS>"
        log(' '.join(caption))
        log('..................................................')

def playVideo(video_path):
    print("Press 'q' to exit")
    cap = cv2.VideoCapture(video_path)
    while(True):
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame',frame)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()

def log(s):
    print('[S2VT] %s' % s)
