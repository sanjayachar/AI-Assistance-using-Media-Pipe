import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
mp_drawing= mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

def calculate_angle(a,b,c):
    a=np.array(a)#First
    b=np.array(b)#Mid
    c=np.array(c)#End
    
    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle=360-angle
        
    return angle

def recognise_curl(detection):
    counter=0
    global state
    global feedback
    global range_flag
    global left_angle
    global right_angle
    try:
        landmarks=detection.pose_landmarks.landmark
        
        # left arm
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] 

        # right arm
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
       # left_angle.append(int(left_elbow_angle))
        #right_angle.append(int(right_elbow_angle))

        # down state
        if left_elbow_angle > 160 and right_elbow_angle > 160:
            if not range_flag:
                feedback = 'Did not curl completely.'
            else:
                feedback = 'Good rep!'
            state = 'Down'
            
        # not fully curled
        elif (left_elbow_angle > 50 and right_elbow_angle > 50) and state == 'Down':
            range_flag = False
            feedback = ''
            
        # up state
        elif (left_elbow_angle < 30 and right_elbow_angle < 30) and state == 'Down':
            state = 'Up'
            feedback = ''
            range_flag = True
            counter += 1
    
    except:
        left_angle.append(180)
        right_angle.append(180) 

cap=cv2.VideoCapture(0)
counter=0
stage=None
range_flag = True
halfway = False
feedback = ''
frame_count = 0
# Plotting variables
frames = []
left_angle = []
right_angle = []
body_angles = []


# Get user's maximum resolution
with mp_pose.Pose(min_detection_confidence=50,min_tracking_confidence=50) as pose:
    while cap.isOpened():
        ret, frame=cap.read()

        #Recolor image to RGB
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False

        #make detection
        results=pose.process(image)

        #recoloring back to BGR
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        # Render detections
        
        recognise_curl(results)

        cv2.rectangle(image,(0,0),(225,73),(245,117,16),-1)
        
        #Rep data vizualize
        cv2.putText(image,'REPS   ',(15,12),
                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(10,60),
                   cv2.FONT_HERSHEY_SIMPLEX,1,(300,300,300),2,cv2.LINE_AA)
        
        #stage data vizualize
        cv2.putText(image,'STAGE',(65,12),
                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,stage,(60,60),
                   cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        
        #Render detections body node connectios point drawing

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed',image)
        #this is for closeing a screen
        if cv2.waitKey(10) &0xFF == ord('q'):
            break
    #these two lines of code is for releasing the web cam after entering key
    #and destroying the window
    cap.release()
    cv2.destroyAllWindows()