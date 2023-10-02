import subprocess
print("Enter the particular number to do excercise\n1.Curl\n2.Left Curl\n3.Right Curl\n4.Situp\n5.Squat\n6.Push Up\n7.Pull Up")
num=int(input())
if num==1:
    subprocess.call(['python','PoseEstimationCurl.py'])
elif num==2:
    subprocess.call(['python','PoseEstimationLeftCurl.py'])
elif num==3:
    subprocess.call(['python','PoseEstimationRightCurl.py'])
elif num==4:
    subprocess.call(['python','PoseEstimationSitup.py'])
elif num==5:
    subprocess.call(['python','PoseEstimationSquat.py'])
elif num==6:
    subprocess.call(['python','PoseEstimationPushups.py'])
elif num==7:
    subprocess.call(['python','PoseEstimationPullups.py'])