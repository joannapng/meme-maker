import cv2

def draw_bounding_boxes(eyes, detection, img, i):

    new_image = img

    (dx1, dy1, w1, h1) = eyes[0]
    (dx2, dy2, w2, h2) = eyes[1]

    x, y, _, _ = detection
    dx = min(dx1, dx2)
    dy = min(dy1, dy2)
    w = max(dx1 + w1, dx2 + w2) - dx
    h = max(dy1 + h1, dy2 + h2) - dy
    new_image = cv2.rectangle(new_image,(x+dx,y+dy),(x+dx+w,y+dy+h),(0,0,255),3)
    cv2.putText(new_image, "eyes" + str(i), (x+dx, y+dy-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    return new_image, x+dx, y+dy, w, h

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

def eyes_detection(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)

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

    detections = get_detections((faces_frontal, faces_profile, cat_faces))

    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    
    new_img = bgr
    eye_sets = []

    i = 1
    for detection in detections:
        (x, y, w, h) = detection
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        tmp_eyes = []

        for eye in eyes:
            dx, dy, w, h = eye
            tmp_eyes.append(eye)

        if len(tmp_eyes) == 2:
            new_img, x, y, w, h = draw_bounding_boxes(tmp_eyes, detection, new_img, i)
            eye_sets.append((x, y, w, h))
            i+=1
    
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    
    return new_img, eye_sets