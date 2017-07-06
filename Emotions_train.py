import os
import cv2
import numpy as np
import glob
from collections import defaultdict
import sys
import face_utils
import random

def main():
    """
    Testing methods
    """
    '''
    The dataset used for emotion is the KDEF data set which can be downloaded
    http://www.emotionlab.se/resources/kdef
    
    #test_frame = cv2.imread('IMG_20170623_F214651.jpg')
    impath = 'KDEF'+os.sep+'AF01/AF01ANS.JPG'
    test_frame = cv2.imread('KDEF'+os.sep+'AF01/AF01ANS.JPG')
    fdet = FaceDetector('.')
    frame_list = fdet.detect(test_frame)
    fdet.display_images(frame_list)
    '''
    fdet = FaceDetector('face_detection_models')
    ke = KdefEmotions('KDEF')
    all_files,all_labels = ke.get_files_labels()

    all_frames = [cv2.imread(f) for f in all_files]
    frame_pairs = fdet.extract_faces_from_frames_labels(all_frames,all_labels)

    #print frame_pairs[0]
    print len(frame_pairs)


    #randomize training label pairs

    random.shuffle(frame_pairs)


    all_faces,all_labels = zip(*frame_pairs)

    #Convert labels from string to numeric

    numeric_labels = face_utils.str_label_to_numeric(KdefEmotions.emotions,all_labels)

    num_records = len(all_faces)
    train_frac = 0.99
    test_frac = 1-train_frac
    pi = int(np.floor(train_frac * num_records))
    train_faces = all_faces[0:pi]
    test_faces = all_faces[pi:]
    train_labels = np.asarray(numeric_labels[0:pi])
    test_labels = np.asarray(numeric_labels[pi:])



    MODEL_FILE="emotions_identifier.model"
    fishface = cv2.face.createLBPHFaceRecognizer()

    if os.path.exists(MODEL_FILE):
        Key_stroke = raw_input('There is already a existing emotions model, should i delete it (Y/N)')
        if Key_stroke == 'Y' or Key_stroke=='y': os.remove(MODEL_FILE)
        print ("createing model")
        fishface.train(train_faces, train_labels)
        fishface.save(MODEL_FILE)
    else:
        print ("createing model")
        fishface.train(train_faces, train_labels)
        fishface.save(MODEL_FILE)





#class EmotionClassifier:
#    def __init__(self,training_frames,training_labels):


class KdefEmotions:
    """
    Construct and train a model using the Karolinksa Directed Emotion
    Face Dataset (KDEF)

    Files in the KDEF dataset are coded in the following way:
    Letter 1: Session (A or B)
    Letter 2: Gender (F or M)
    Letter 3 & 4: ID number (01-35)
    Letter 5 & 6: Expression
    Letter 7 & 8: Angle
    """

    #Emotion list
    #emotions =  ["neutral", "anger", "contempt", "disgust", "fear",
    #             "happy", "sadness", "surprise"]
    emotions = ['afraid','angry','disgusted','happy','neutral','sad','surprised']
    kdef_emo_dict = { 'AF':'afraid',
                      'AN':'angry',
                      'DI':'disgusted',
                      'HA':'happy',
                      'NE':'neutral',
                      'SA':'sad',
                      'SU':'surprised'}

    kdef_angle_dict = { 'FL': 'full_left',
                        'HL': 'half_left',
                        'S': 'straight',
                        'HR': 'half_right',
                        'FR': 'full_right'}


    def __init__(self,folder='KDEF'):
        """
        Given a picture, classify its emotion class
        """
        self.folder = folder

        #dict with emotion as key and list of files as value
        self.emotion_dict = defaultdict(list)
        #Populate emotion_dict with pictures
        self._get_emotion_files('S')


    def get_files_labels(self):
        """ Get a list of (image_file,emotion_label) from
        self.emotion_dict"""
        imfiles = []
        imlabels = []
        for emotion,imlist in self.emotion_dict.viewitems():
            for img in imlist:
                imfiles.append(img)
                imlabels.append(emotion)
        return (imfiles,imlabels)

    def _get_emotion_files(self,angle='S'):
        ''' Populate self.emotion_dict'''
        print "Extracting all images with angle {}".format(KdefEmotions.kdef_angle_dict[angle])
        sub_folders = glob.glob(self.folder+os.sep+'*')
        for folder in sub_folders:
            #print folder
            folder_images = glob.glob(folder+os.sep+'*'+angle+'.JPG')
            #print folder_images
            for image_file in folder_images:
                cur_emotion = self.extract_emotion(image_file)
                self.emotion_dict[cur_emotion].append(image_file)
        #self.emotion_dict.viewkeys(),len(self.emotion_dict.viewvalues())
        print "Number of images for each emotion"
        for key,val in self.emotion_dict.viewitems():
            print '{}:\t{}'.format(key,len(val))


    def extract_emotion(self,fname):
        '''
        Extract letters 5 & 6 in filename to parse emotion of
        image
        '''
        bname = os.path.basename(fname)
        return KdefEmotions.kdef_emo_dict[bname[4:6]]








class FaceDetector:
    def __init__(self,models_dir='facedetect'):
        self.width = 128
        self.height = 128
        self.models_dir = models_dir
        #print self.models_dir
        self.model_file = os.path.abspath(self.models_dir+os.sep+'haarcascade_frontalface_default.xml')


        '''
        [ cv2.CascadeClassifier(
                                      self.dir+os.sep+'haarcascade_frontalface_alt2.xml'),
                          cv2.CascadeClassifier(
                                       self.dir+os.sep+'haarcascade_frontalface_alt.xml'),
                          cv2.CascadeClassifier(
                                       self.dir+os.sep+'haarcascade_frontalface_alt_tree.xml')
                          ]
        '''
        self.face_detector = cv2.CascadeClassifier(self.model_file)

    def detect(self,frame):
        ''' Returns a list of self.height by self.width numpy arrays
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
                                             gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(30, 30),
                                             flags = cv2.CASCADE_SCALE_IMAGE)


        ret_list = []
        for i,(x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+h, y+h), (0, 255, 0), 2)
            im = 'im'+str(i)
            img = gray[y:y+h,x:x+h]
            img = cv2.resize(img,(self.width,self.height))
            #cv2.imshow(im,img)
            ret_list.append(img)

        return ret_list

    def detect_draw(self,frame):
        ''' Returns a list of self.height by self.width numpy arrays
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
                                             gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(30, 30),
                                             flags = cv2.CASCADE_SCALE_IMAGE)


        ret_list = []

        for i,(x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+h, y+h), (0, 255, 0), 2)
            im = 'im'+str(i)
            img = gray[y:y+h,x:x+h]
            img = cv2.resize(img,(self.width,self.height))
            #cv2.imshow(im,img)
            ret_list.append(img)


        return ret_list


    def display_images(self,im_list):
        while True:
            for i,img in enumerate(im_list):
                imname = 'img'+str(i)
                cv2.imshow(imname,img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def identify_faces_in_frames_list(self,frames_list):
        """
        Given a list of image files, detect faces and return frames in each file
        Useful for segmenting out faces from a set of images
        """
        face_frames = []

        for i,frame in enumerate(frames_list):
            #print frame.shape
            face_list = self.detect(frame)
            face_frames.extend(face_list)
        return face_frames

    def extract_faces_from_frames_labels(self,frames,labels):
        '''Given a list of frames and labels,
        return frames containing faces from each frame'''

        ret_tup_list = []
        for i,frame in enumerate(frames):
            label = labels[i]
            seg_list = self.detect(frame)
            seg_len = len(seg_list)
            for subframe in seg_list:
                ret_tup_list.append((subframe,label))
        return ret_tup_list




if __name__ == "__main__":
    main()
