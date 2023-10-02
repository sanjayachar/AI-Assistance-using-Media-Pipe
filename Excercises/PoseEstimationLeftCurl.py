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
            shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            #calculate angle
            angle=calculate_angle(shoulder,elbow,wrist)
            
            #Visualize angle(printing the vizualized angle into the screen)
            cv2.putText(image,str(angle),
                       tuple(np.multiply(elbow,[640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
            
            #curl counter logic
            if angle>160:
                stage="down"
            if angle<30 and stage=="down":
                stage="up"
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


