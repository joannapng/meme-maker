import cv2
import ipywidgets as widgets

def draw_bounding_boxes(faces_frontal, faces_profile, cat_faces, img):
    i = 1
    new_image = img
    for face_frontal in faces_frontal:
        (x, y, w, h) = face_frontal
        new_image = cv2.rectangle(new_image,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.putText(new_image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        i += 1

    for face_profile in faces_profile:
        (x, y, w, h) = face_profile
        new_image = cv2.rectangle(new_image,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(new_image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        i += 1

    for cat_face in cat_faces:
        (x, y, w, h) = cat_face
        new_image = cv2.rectangle(new_image,(x,y),(x+w,y+h),(0,0,255),3)
        cv2.putText(new_image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        i += 1

    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image


def face_detection(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY) # convert to gray scale

    face_frontal_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') # detect frontal faces
    face_profile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml') # detect profiles
    cat_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface.xml') # detect cats
    
    faces_frontal = face_frontal_cascade.detectMultiScale(gray, 1.3, 4)
    faces_profile = face_profile_cascade.detectMultiScale(gray, 1.3, 4)
    cat_faces = cat_cascade.detectMultiScale(gray, 2, 1)

    num_faces = len(faces_frontal) + len(faces_profile)
    num_cat_faces = len(cat_faces)
    
    print(f'Detected {num_faces} faces')
    print(f'Detected {num_cat_faces} cat faces')

    if num_faces + num_cat_faces == 0:
        print('No faces detected. Please choose another image')
        return img, []

    new_img = draw_bounding_boxes(faces_frontal, faces_profile, cat_faces, bgr)
    detections = (faces_frontal, faces_profile, cat_faces)

    return new_img, get_detections(detections)

def get_detections(detections):
    faces_frontal, faces_profile, cat_faces = detections

    faces = []
    if faces_frontal != ():
        for face_frontal in faces_frontal:
            faces.append(face_frontal) 
    if faces_profile != ():
        for face_profile in faces_profile:
            faces.append(face_profile)
    if cat_faces != ():
        for cat_face in cat_faces:
            faces.append(cat_face)

    return faces