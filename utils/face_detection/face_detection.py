import cv2 as cv
import numpy as np
import dlib
from mtcnn.mtcnn import MTCNN


face_cascade = cv.CascadeClassifier('../utils/face_detection/haarcascade_frontalface_default.xml')
def cascadeHaar(img):
    img = np.array(img, dtype=np.uint8)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for face in faces:
        img = img[max(0,face[1]):max(0,face[1])+max(0,face[3]), max(0,face[0]):max(0,face[0])+max(0,face[2])]
        return img, 1
    return img, 0



hog_face_detector = dlib.get_frontal_face_detector()
def hog_face_detection(img):
    faces_hog = hog_face_detector(img, 1)
    for face in faces_hog:
        img = img[max(0,face.top()):max(0,face.bottom()), max(0,face.left()):max(0,face.right())]
        return img, 1
    return img, 0


cnn_face_detector = dlib.cnn_face_detection_model_v1('../utils/face_detection/mmod_human_face_detector.dat')
def cnn_face_detection(img):
    faces_cnn = cnn_face_detector(img, 1)
    for face in faces_cnn:
        img = img[max(0,face.rect.top()):max(0,face.rect.bottom()), max(0,face.rect.left()):max(0,face.rect.right())]
        return img, 1
    return img, 0


detector = MTCNN()
def mtcnn_detect(img):
    faces = detector.detect_faces(img)
    for face in faces:
        img = img[max(0,face['box'][1]):max(0,face['box'][1]) + max(0,face['box'][3]),
                  max(0,face['box'][0]): max(0,face['box'][0]) + max(0,face['box'][2])]
        return img, 1
    return img, 0


def get_face(img):
    img, label = cascadeHaar(img)
    if label==0:
        img, label = hog_face_detection(img)
        if label==0:
            img, label = mtcnn_detect(img)
            if label==0:
                img, label = cnn_face_detection(img)
    return img