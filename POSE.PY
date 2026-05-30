
import mediapipe as mp
import cv2

INPUT = "data/sample_rubble.jpg"

def run():
    mp_pose = mp.solutions.pose
    img = cv2.imread(INPUT)
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            print("No pose landmarks detected.")
            return
        kp = res.pose_landmarks.landmark
        visible = sum(1 for p in kp if p.visibility > 0.5)
        print("Visible keypoints:", visible)

if __name__ == '__main__':
    run()
