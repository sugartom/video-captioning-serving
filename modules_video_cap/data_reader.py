import os
import cv2

class DataReader:
  def __init__(self):

    self.cap = None
    self.end_of_video = False
    self.frame_id = 0

  def Setup(self, video_path = ''):
    
    if not os.path.exists(video_path):
      self.log('Cannot load video!')
      self.end_of_video = True
      return 
    
    self.vid_name = video_path.rsplit('/',1)[-1].rsplit('.',-1)[0]
    self.cap = cv2.VideoCapture(video_path)

  def PostProcess(self):
    if self.end_of_video:
      return {}
    
    ret, frame = self.cap.read()

    if not ret:
      self.log('Num of frames %d' % self.frame_id)
      self.log('End of video')
      self.end_of_video = True 
      return {}
    
    
    output = {'img': frame, 'meta': {'frame_id':self.frame_id, 'vid_name': self.vid_name}}
    self.frame_id += 1

    return output

  def log(self, s):
    print('[DataReader] %s' % s)
