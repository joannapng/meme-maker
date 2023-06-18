"""
Functions to detect eyes used for adding glasses
"""

import cv2

def draw_bounding_boxes(eyes, detection, img, i):
    """
    Draw bounding boxes around detected sets of eyes
    """
    new_image = img

    # dx: horizontal offset from the origin of the roi
    # dy: vertical offset from the origin of the roi
    # w: width of eye
    # h: height of eye
    (dx1, dy1, w1, h1) = eyes[0]
    (dx2, dy2, w2, h2) = eyes[1]

    # (x, y): face offset from the image origin
    x, y, _, _ = detection

    # the box will start from the left eye (closer to the origin)
    dx = min(dx1, dx2)
    dy = min(dy1, dy2)

    # the width of the eye set will be the rightmost/lowermost point minus the origin 
    # of the left eye (dx, dy) from the roi
    w = max(dx1 + w1, dx2 + w2) - dx
    h = max(dy1 + h1, dy2 + h2) - dy

    # draw the bounding box around the set of eyes
    new_image = cv2.rectangle(new_image,(x+dx,y+dy),(x+dx+w,y+dy+h),(0,0,255),3)
    cv2.putText(new_image, "eyes" + str(i), (x+dx, y+dy-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # return the coordinates of the eye set wrt the original image, not the roi
    return new_image, x+dx, y+dy, w, h

def get_detections(detections):
    """
    Construct a list of all detected faces (frontal, profile, cats)
    """
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

def eyes_detection(img):
    """
    Given an img, find a set of eyes for each face using Haar Cascades
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)      # convert to bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)    # convert to gray 
                                                    # (could be done in one step, but we need bgr 
                                                    # for drawing bounding boxes)

    face_frontal_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')    # detect frontal faces
    face_profile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')            # detect profiles
    cat_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface.xml')                  # detect cats

    # detectMultiScale parameters: img, scaleFactor, minNeighbors
    # scaleFactor: parameter specifying how much the image size is reduced at 
    # each image scale. The smaller the scaleFactor, the smaller the step for 
    # resizing -> + more accuracy, - slower
    # minNeighbors: parameter specifying how many neighbors each candidate 
    # rectange should have to be called a face. Higher values give less matches
    # of higher quality
    faces_frontal = face_frontal_cascade.detectMultiScale(gray, 1.1, 6)
    faces_profile = face_profile_cascade.detectMultiScale(gray, 1.1, 6)
    cat_faces = cat_cascade.detectMultiScale(gray, 1.1, 6)

    num_faces = len(faces_frontal) + len(faces_profile)
    num_cat_faces = len(cat_faces)
    
    print(f'Detected {num_faces} faces')
    print(f'Detected {num_cat_faces} cat faces')

    if num_faces + num_cat_faces == 0:
        print('No faces detected. Please choose another image')
        return img, []

    detections = get_detections((faces_frontal, faces_profile, cat_faces))

    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml') # detect eyes
    new_img = bgr
    eye_sets = []

    i = 1
    for detection in detections:    # for each face detected
        (x, y, w, h) = detection
        roi_gray = gray[y:y+h, x:x+w]   # roi: region of interest

        eyes = eye_cascade.detectMultiScale(roi_gray)   # leave default parameters per opencv example
        tmp_eyes = []   # will hold a set of eyes if two eyes are found

        for eye in eyes:
            dx, dy, w, h = eye
            tmp_eyes.append(eye)

        if len(tmp_eyes) == 2:  # if two eyes are found, consider them a valid set of eyes
            new_img, x, y, w, h = draw_bounding_boxes(tmp_eyes, detection, new_img, i)
            eye_sets.append((x, y, w, h))
            i+=1
    
    if len(eye_sets) == 0:
        print("No eyes detected. Please choose another image.")
    else:
        print(f'Detected {len(eye_sets)} eyes')

    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    
    return new_img, eye_sets