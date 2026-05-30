import cv2, numpy as np

INPUT = "data/sample_clip.mp4"
OUT = "data/motion_trace.npy"

def run():
    cap = cv2.VideoCapture(INPUT)
    ret, prev = cap.read()
    if not ret:
        print("Cannot read video.")
        return

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mags = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5,3,15,3,5,1.2,0)
        mag,_ = cv2.cartToPolar(flow[...,0], flow[...,1])
        mags.append(np.mean(mag))
        prev_gray = gray

    mags=np.array(mags)
    np.save(OUT, mags)
    print("Avg motion:", mags.mean())

if __name__ == '__main__':
    run()
