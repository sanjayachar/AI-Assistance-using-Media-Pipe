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

def plot_viz(frames,angle,counter):
    global left_angle
    # Set figure size
    plt.rcParams["figure.figsize"] = (20,5)

    fig, ax = plt.subplots()
    ax.plot(frames,angle, '-', color = 'red', label = 'Arm Angle')
    #ax.plot(frames, right_angle, '-', color = 'blue', label = 'Right Arm Angle')
    ax.axhline(y=30, color='g', linestyle='--')
    ax.legend(loc = 'center left')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Angle')
    print(f'Congratulations! You managed {counter} curls!')
#Accessing a webcam
cap=cv2.VideoCapture(0)

#curl counter variables
counter=0
stage=None
frame_count=0
frames=[]
left_angle=[]
right_angle=[]
halfway=range_flag=None
#Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame=cap.read()
        frame_count+=1
        frames.append(frame_count)
        #Recolor image to RGB
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False

        #make detection
        results=pose.process(image)

        #recoloring back to BGR
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Extract landmarks
        try:
            landmarks=results.pose_landmarks.landmark
            
            #Get coordinates
            left_hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_heel=[landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            #right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            #right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            #right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            #right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]        
        
            #calculate angle
            angle_knee = calculate_angle(left_hip, left_knee, left_heel)
            angle_body = calculate_angle(left_shoulder, left_hip, left_knee)        
            #left_angle.append(int(left))
            #right_angle.append(int(right))
            
            cv2.putText(image,str(angle_knee),
                       tuple(np.multiply(left_knee,[640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image,str(angle_body),
                       tuple(np.multiply(left_hip,[640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)        
        #POSE CHECKING 1: Knees bending inwards    
           
            #curl counter logic
            if angle_body > 90 and angle_knee > 90: #Resting position;to check if situp was done properly
                stage = "down"

            if angle_body < 80 and angle_body < 80 and stage == "down": #Half-way there (Used for checking bad situps)
                stage = "up"
                counter+=1
                print(counter)
    
        except:
            pass
        
        #Render curl counter
        #Setup status box
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
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(200,100,60),thickness=3,circle_radius=3),
                                 mp_drawing.DrawingSpec(color=(200,60,200),thickness=3,circle_radius=3))
        
        cv2.imshow('Mediapipe Feed',image)
        #this is for closeing a screen
        if cv2.waitKey(10) &0xFF == ord('q'):
            break
    #these two lines of code is for releasing the web cam after entering key
    #and destroying the window
    #cap.release()
    #cv2.destroyAllWindows()
    #plot_viz(frames,angle,counter)
    #print('\nThe red/blue lines show the angle of your targetted body part throughout your exercise,') 
    #print('whereas the green dotted line is the minimum angle required for the exercise to be recorded as one repetition.') 


