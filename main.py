# -- coding: utf-8 --
"""
Created on Thu Aug 15 13:28:47 2024

@author: visikom2023
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
import time 
from keras.models import load_model
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
#%matplotlib inline 
import matplotlib.pyplot as plt
import copy
#from keras.utils.vis_utils import plot_model
    
def DrawText(img,sText,x,y):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    posf = (x,y)
    fontScale              = 5
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    print("Masuk")
    cv2.putText(img,sText, 
        posf, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    return copy.deepcopy(img)
    

def LoadCitraTraining(sDir,LabelKelas):
  
  JumlahKelas=len(LabelKelas)
  TargetKelas = np.eye(JumlahKelas)
  
  # Menyiapkan variabel list untuk data menampung citra dan data target
  X=[]#Menampung Data Citra
  T=[]#Menampung Target
  for i in range(len(LabelKelas)):    
    #Membaca file citra di setiap direktori data set  
    DirKelas = os.path.join(sDir, LabelKelas[i])
    print(f"Loading images from: {DirKelas}")
    
    files = os.listdir(DirKelas)
    print(f"Found {len(files)} files in {LabelKelas[i]}")
    
    for f in files:
      ff=f.lower()  
      print(f)
      #memilih citra dengan extensi jpg,jpeg,dan png
      if (ff.endswith('.jpg')|ff.endswith('.jpeg')|ff.endswith('.png')):
         NmFile = os.path.join(DirKelas,f)
         print(f"Reading file: {NmFile}")
         
         img = cv2.imread(NmFile, cv2.IMREAD_COLOR)
         if img is None:
                    print(f"Warning: Failed to read {NmFile}")
                    continue  # Skip this file
                    
         #membaca citra berwarna sebagai data bertipe double 
         img= np.double(cv2.imread(NmFile,1))
         if img is None:
                    print(f"Warning: Failed to read {NmFile}")
                    continue  # Skip this file
                    
         if len(img.shape) == 2 or img.shape[2] == 1:  # If the image is grayscale
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                    
         img=cv2.resize(img,(128,128));
         #Normalisasi data citra menjadi sehingga maksimum menjadi 1
         img= np.asarray(img)/255;
         img=img.astype('float32')
         #Menambahkan citra dan target ke daftar
         
         if img.shape != (128, 128, 3):
                    print(f"Warning: Unexpected image shape: {img.shape} for file {NmFile}")
                    continue  # Skip this file if shape is incorrect
                    
         X.append(img)
         T.append(TargetKelas[i])
     #--------akhir loop :Pfor f in files-----------------
  #-----akhir  loop :for i in range(len(LabelKelas))----
  
  #Mengubah List Menjadi numppy array
  X=np.array(X)
  T=np.array(T)
  
  print(f"Final shape of X: {X.shape}")
  print(f"Final shape of T: {T.shape}")
   
  X=X.astype('float32')
  T=T.astype('float32')
  return X,T

def ModelDeepLearningCNN(JumlahKelas):
    input_img = Input(shape=(128, 128, 3)) 
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  
    x = MaxPooling2D((2, 2), padding='same')(x)   
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)   
    x = MaxPooling2D((2, 2), padding='same')(x)   
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(100,activation='relu')(x)
    x = Dense(100,activation='relu')(x)
    x=Dense(JumlahKelas,activation='softmax')(x)
    ModelCNN = Model(input_img, x)  
    ModelCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #ModelCNN.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    #plot_model(ModelCNN, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return ModelCNN


def TrainingCNN(JumlahEpoh,DirektoriDataSet,LabelKelas,NamaFileBobot ='weight.h5' ):
    #Membaca Data training dan label Kelas 
    X,D=LoadCitraTraining(DirektoriDataSet,LabelKelas)
    
    print("Shape of X:", X.shape)  # Should be (num_samples, 128, 128, 3)
    print("Shape of D:", D.shape)  # Should be (num_samples, num_classes)

    print(X)
    JumlahKelas = len(LabelKelas)
    #Membuat Model CNN
    ModelCNN =ModelDeepLearningCNN(JumlahKelas)
    #Trainng
    history=ModelCNN.fit(X, D,epochs=JumlahEpoh,shuffle=True)
    #Menyimpan hasil learning
    ModelCNN.save(NamaFileBobot)
    #Mengembalikan output 
    return ModelCNN,history


##########################################################
def Klasifikasi(Image,ModelCNN):

  X=[]
  ls = [];

  img= copy.deepcopy(Image)
  img=cv2.resize(img,(128,128))
  img= np.asarray(img)/255
  img=img.astype('float32')
  X.append(img)  
  X=np.array(X)
  X=X.astype('float32')
  hs=ModelCNN.predict(X,verbose=0)

  if hs.max()>0.5:
      idx = np.max(np.where( hs == hs.max()))
  else:
    idx=-1
      
 
  return idx
def GetFileName():
        x = datetime.datetime.now()
        s = x.strftime('%Y-%m-%d-%H%M%S%f')
        return s
def CreateDir(path):
    ls = [];
    head_tail = os.path.split(path)
    ls.append(path)
    while len(head_tail[1])>0:
        head_tail = os.path.split(path)
        path = head_tail[0]
        ls.append(path)
        head_tail = os.path.split(path)   
    for i in range(len(ls)-2,-1,-1):
        sf =ls[i]
        isExist = os.path.exists(sf)
        if not isExist:
            os.makedirs(sf)
#NamaDataSet = "TanganSamping"


#########################################################
# Membuat data set pose 
#########################################################
def CreateDataSet(NoKamera,NamaDataSet,DirektoriDataSet ="c:\\temp\\dataimage" ):
    DirektoriData =DirektoriDataSet +"\\"+NamaDataSet+"\\"+GetFileName()    
    CreateDir(DirektoriData)        
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    imsize=(640, 480)
    height = imsize[1]
    width = imsize[0]
    TimeStart = time.time() 
    TimeNow = time.time() +10
    FrameRate = 5
    # For webcam input:
    cap = cv2.VideoCapture(NoKamera,cv2.CAP_DSHOW)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
    
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        ori = copy.deepcopy(image)
       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, imsize)
     
       
        results = pose.process(image)
        if not results.pose_landmarks:
          continue
        lm = []
        
    
        for lmark in results.pose_landmarks.landmark:
            print(lmark)
            if (lmark.x>0.01)and(lmark.x<1-0.01)and(lmark.y>0.01)and(lmark.y<1-0.01):
                m = [lmark.x*width  , lmark.y*height]
                lm.append(m)
        if len(lm) ==0:
            continue
        lm = np.array(lm)
        x = lm[:,0]
        y = lm[:,1]
        ymin = np.min(y)
        ymax = np.max(y)
        xmin =np.min(x)
        xmax =np.max(x)
            
                
            
        ymin = np.int32(np.min(y))-3
        ymax = np.int32(np.max(y))-3
        xmin = np.int32(np.min(x))+3
        xmax = np.int32(np.max(x))+3
    
        
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bimage = np.zeros((height,width,3), np.uint8)
        cv2.rectangle(bimage,(xmin,ymin),(xmax,ymax),(0,255,0),2)
    
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        
        mp_drawing.draw_landmarks(
            bimage,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
 
        image = cv2.rectangle(image, (xmin,ymin),(xmax,ymax), (255,0,0),2)
                          
        cropped_image = bimage[ymin:ymax, xmin:xmax,:]
        dy =ymax -ymin
        dx = xmax -xmin
        print(dy,dx)
        print(cropped_image.shape)
        TimeNow = time.time() 
        if TimeNow-TimeStart>1/FrameRate:
            print(cropped_image.shape)
            TimeStart = TimeNow
            sFile = DirektoriData+"\\"+GetFileName()
            imsize2=(128,128)
            cropped_image = cv2.resize(cropped_image, imsize2)
            cv2.imwrite(sFile+'.jpg', cropped_image)
            cv2.imwrite(sFile+'.png', image)
            #cv2.imwrite(sFile+'.bmp', ori)
            
            
    
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    cv2.destroyAllWindows()

    
##########################################
# Menguji Pose data set yang telah dibuat 
##########################################
def TesPosePrediction(DirDataSet,DirKlasifikasi,LabelKelas,ModelCNN=[]):
#Apabila parameter input ModelCNN tidak di isi maka
#   akan menggunakan bobot pada file 'weight.h5
  if not(ModelCNN):
      ModelCNN = load_model('weight.h5') 
      
#Menyiapkan Data input Yang akan di kasifikasikan
  X=[]
  ls = [];
  DirKelas = DirDataSet+"\\"+DirKlasifikasi
  print(DirKelas)
  files = os.listdir(DirKelas)
  n=0;
  for f in files:
      ff=f.lower()  
      print(f)
      if (ff.endswith('.jpg')|ff.endswith('.jpeg')|ff.endswith('.png')):
         ls.append(ff) 
         NmFile = os.path.join(DirKelas,f)
         img= cv2.imread(NmFile,1)
         img=cv2.resize(img,(128,128))
         img= np.asarray(img)/255
         img=img.astype('float32')
         X.append(img)
     #----Akhir if-------------
  #---Akhir For 
  X=np.array(X)
  X=X.astype('float32')
  #Melakukan prediksi Klasifikasi
  hs=ModelCNN.predict(X,verbose=0)
  
  LKlasifikasi=[];
  LKelasCitra =[];
  n = X.shape[0]
  for i in range(n):
      v=hs[i,:]
      if v.max()>0.5:
          idx = np.max(np.where( v == v.max()))
          LKelasCitra.append(LabelKelas[idx])
      else:
          idx=-1
          LKelasCitra.append("-")
      #------akhir if
      LKlasifikasi.append(idx);
  #----akhir for
  LKlasifikasi = np.array(LKlasifikasi)
  return ls, hs, LKelasCitra

#########################################
#Memprediksi Pose 
#########################################

def PredictPose(NoKamera,LabelKelas):
    ModelCNN = load_model('weight.h5') 

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    imsize=(640, 480)
    height = imsize[1]
    width = imsize[0]

    # For webcam input:
    cap = cv2.VideoCapture(NoKamera,cv2.CAP_DSHOW)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
    
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, imsize)
     
        results = pose.process(image)
        if not results.pose_landmarks:
         continue
        lm = []
        
    
        for lmark in results.pose_landmarks.landmark:
            
            if (lmark.x>0.01)and(lmark.x<1-0.01)and(lmark.y>0.01)and(lmark.y<1-0.01):
                m = [lmark.x*width  , lmark.y*height]
                lm.append(m)
        if len(lm) ==0:
            continue
        lm = np.array(lm)
        x = lm[:,0]
        y = lm[:,1]
        ymin = np.min(y)
        ymax = np.max(y)
        xmin =np.min(x)
        xmax =np.max(x)
            
                
            
        ymin = np.int32(np.min(y))-3
        ymax = np.int32(np.max(y))-3
        xmin = np.int32(np.min(x))+3
        xmax = np.int32(np.max(x))+3
    
        
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bimage = np.zeros((height,width,3), np.uint8)
        cv2.rectangle(bimage,(xmin,ymin),(xmax,ymax),(0,255,0),2)
    
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        
        mp_drawing.draw_landmarks(
            bimage,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cropped_image = bimage[ymin:ymax, xmin:xmax,:]
        idx = Klasifikasi(cropped_image, ModelCNN)
        x=60
        y=60
        image= cv2.flip(image, 1)
    
        if idx>=0:
            cv2.putText(image,LabelKelas[idx], (x,y), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)
        

        cv2.imshow('Prediksi Pose', image)
                
       
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    cv2.destroyAllWindows()
    
print("OK")

DirektoriDataSet="C:\\Temp\\datasetvaru"
#CreateDataSet(2,"TanganKanan",DirektoriDataSet)
#CreateDataSet(2,"TanganKiri",DirektoriDataSet)
#CreateDataSet(2,"Berhenti",DirektoriDataSet)
#CreateDataSet(2,"Maju",DirektoriDataSet)
#CreateDataSet(2,"Mundur",DirektoriDataSet)
#Training Data Set

#   Data Set disimpan dalam direktori yang sama dengan nama kelas    

#b. Label Data Set 
LabelKelas=("TanganKiri",
             "TanganKanan", "Berhenti", "Maju", "Mundur")

X,D=LoadCitraTraining(DirektoriDataSet,LabelKelas)

JumlahEpoh = 10;

#d. training
ModelCNN,history = TrainingCNN(JumlahEpoh,DirektoriDataSet,LabelKelas )
ModelCNN.summary()

#c. Menampilkan Grafik Loss dan accuracy
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss/accuracy')
plt.xlabel('epoch')
plt.show()

#PredictPose(2,LabelKelas)
