import cv2

"""
def draw_contours(img, contours):
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
        cv2.drawContours(new_img, [approx], 0, (0, 0, 0), 1)

        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        if len(approx) == 3:
            cv2.putText(new_img, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        elif len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                cv2.putText(new_img, "Square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                cv2.putText(new_img, "Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        elif len(approx) == 5:
            cv2.putText(new_img, "Pentagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        elif len(approx) == 10:
            cv2.putText(new_img, "Star", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.putText(new_img, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    return new_img


def geometry_detection(img):
    k = 10

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    (T, threshold) = cv2.threshold(gray, 240, 255,  
                    cv2.THRESH_BINARY_INV)    
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > k:
        contours = sorted(contours, key=cv2.contourArea)[-k:]

    new_img = draw_contours(img, contours)

    min_x, max_x = 0, 0
    min_y, max_y = 0, 0
    w = 0
    h = 0

    contour_boxes = []
    for contour in contours:
        for point in contour:
            if point[0][0] < min_x:
                min_x = point[0][0]

            if point[0][0] > max_x:
                max_x = point[0][0]

            if point[0][1] < min_y:
                min_y = point[0][1]

            if point[0][1] > max_y:
                max_y = point[0][1]

        w = max_x - min_x
        h = max_y - min_y
        contour_boxes.append((min_x, min_y, w, h))

    return new_img, contour_boxes
"""

def draw_bounding_boxes(eye, detection, img, i):

    new_image = img

    (dx, dy, w, h) = eye
    x, y, _, _ = detection
    new_image = cv2.rectangle(new_image,(x+dx,y+dy),(x+dx+w,y+dy+h),(0,0,255),3)
    cv2.putText(new_image, "eye" + str(i), (x+dx, y+dy-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    return new_image

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

def geometry_detection(img):
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
        return img, None

    detections = get_detections((faces_frontal, faces_profile, cat_faces))

    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    
    new_img = bgr
    all_eyes = []

    i = 1
    for detection in detections:
        (x, y, w, h) = detection
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for eye in eyes:
            dx, dy, w, h = eye
            all_eyes.append((x+dx, y+dy, w, h))
            new_img = draw_bounding_boxes(eye, detection, new_img, i)
            i+=1
    
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    
    return new_img, all_eyes