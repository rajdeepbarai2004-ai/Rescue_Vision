import numpy as np

def run():
    body = 15
    try:
        mags = np.load("data/motion_trace.npy")
        motion = mags.mean()
    except:
        motion = 0

    motion_score = min(40, int(motion*40))
    pose = 10
    reach = 7

    total = body + motion_score + pose + reach
    print("Priority Score:", total)

if __name__ == '__main__':
    run()

