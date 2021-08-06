# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 06:49:00 2021

@author: My Pc
"""
import sys
import warnings
import os
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from skimage.io import imread, imshow 
from skimage.transform import resize 
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tkinter import *
from PIL import Image, ImageTk
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os, shutil

def auto_mail_warning(filem):
     #Give from mail address and to mail address
     fromaddr = "@gmail.com"
     toaddr = "@gmail.com"


     msg = MIMEMultipart()


     msg['From'] = fromaddr


     msg['To'] = toaddr
     


     msg['Subject'] = "Your Skin Disease detection Report"

     body = "You're likely to have "+str(filem)


     msg.attach(MIMEText(body, 'plain'))



     s = smtplib.SMTP('smtp.gmail.com', 587)


     s.starttls()

     s.login(fromaddr, "**Your Password here**")


     text = msg.as_string()


     s.sendmail(fromaddr, toaddr, text)
     print("mail warning has been forwarded")


     s.quit()
#the more the limit, the more accurate the results
limit=75

def test(di):
    #where to store the image from client
    image=imread("E:/SDdetection/test/"+tes[0])
    gray=rgb2gray(image)
    gray=resize(gray,(512,512))
    gray = np.ndarray.flatten(gray).reshape(262144,1)
    gray=np.dstack(gray)
    gray = np.rollaxis(gray,axis=2,start=0)
    gray = gray.reshape(1,262144)
    data = pd.DataFrame(gray)
    data["label"] = "Test"
    data = shuffle(data).reset_index()
    data=data.drop(['index'],axis=1)
    #data=data.drop(['level_0'],axis=1)
    x_t= data.values[:,:-1]

    y_t= data.values[:,-1] 
    return x_t,y_t
    
    
def dframe(dty,d1,d2):
    global limit
    images=[None]*limit
    j=0
    for i in dty:
        if(j<limit):
            images[j]=imread(d1+i)
            j+=1
        else:break
    gray=[None]*limit

    k=0

    for i in images:
        if(k<limit):
            gray[k]=rgb2gray(images[k])
            k+=1
        else:
         break


#resize
    for l in range(limit): 
        ac = gray[l]
        gray[l] = resize(ac, (512,512))

    len_of_images = len(gray)

    imgsize = gray[0].shape 

    flatten_size = imgsize[0]*imgsize[1]

    for i in range(len_of_images):
        gray[i] = np.ndarray.flatten(gray[i]).reshape(flatten_size,1) 
# Depth wise Stacking

    gray=np.dstack(gray) 

    gray = np.rollaxis(gray,axis=2,start=0)

    gray = gray.reshape(len_of_images,flatten_size) 

    data = pd.DataFrame(gray) 
 
    #print(ac_data) 

    data["label"] = d2
    return data
def train(ac_data,de_data,me_data,x_te,y_te):
    disease_1 = pd.concat([ac_data,me_data])
    disease = pd.concat([disease_1,de_data])
    disease = shuffle(disease).reset_index()
    disease=disease.drop(['index'],axis=1)
    x= disease.values[:,:-1]
    y= disease.values[:,-1]
    #x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)

#Principal Component Analysis
    pca = decomposition.PCA(n_components =50, whiten=True,random_state=1)

    pca.fit(x)
    x_train_pca=pca.transform(x)

    x_test_pca=pca.transform(x_te)
    
    
#Support Vector Machines


#Support Vector Classifier

    clf = svm.SVC(C=2, gamma=0.006, kernel='rbf')
    clf.fit(x_train_pca,y)
    y_pred = clf.predict(x_test_pca)

   


#Accuracy

    #accuracy = metrics.accuracy_score(y_test,y_pred)

    
    #print(confusion_matrix(y_test, y_pred),accuracy)
    return y_pred
class Root(Tk):
    def __init__(self):
        super(Root,self).__init__()
        self.title("Skin Disease Detection System")
        self.minsize(640,400)

        self.labelFrame = ttk.LabelFrame(self,text="Select the image of the affected part")
        self.labelFrame.grid(column=0,row=1,padx= 20, pady= 20)
        self.btton()

    def btton(self):
        self.button = ttk.Button(self.labelFrame, text="Upload image", command=self.fileDailog)
        self.button.grid(column=1,row=1)
    def fileDailog(self):
        self.fileName = filedialog.askopenfilename(initialdir = "/", title="Select A File",filetype=(("jpeg","*.jpg"),("png","*.png")))
        self.label = ttk.Label(self.labelFrame, text="")
        self.label.grid(column =1,row = 2)
        self.label.configure(text = self.fileName)
        #os.chdir('e:\\')
        #os.system('mkdir BACKUP')
        shutil.move(self.fileName,'E:/SDdetection/test')
if __name__ == '__main__':
    
    root = Root()
    root.mainloop()
    #Directories of datasets
    act = os.listdir("E:/acti")
    der = os.listdir("E:/derma")
    mel = os.listdir("E:/mela")
    tes=os.listdir("E:/SDdetection/test/")
    ac_data=dframe(act,"E:/acti/","Actinic Keratosis")
    de_data=dframe(der,"E:/derma/","Dermatofibroma")
    me_data=dframe(mel,"E:/mela/","Melanoma")
    #print(de_data)
    x_te,y_te=test(tes)
    pred_img=train(ac_data,de_data,me_data,x_te,y_te)
    print(pred_img[0])
    auto_mail_warning(pred_img[0])
    


    
