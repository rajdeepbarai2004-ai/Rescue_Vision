import torch, cv2, numpy as np

INPUT = "data/sample_rubble.jpg"
OUT = "data/depth.png"

def run():
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform

    img = cv2.imread(INPUT)
    inp = transform(img).unsqueeze(0)

    with torch.no_grad():
        depth = model(inp)

    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min())/(depth.max()-depth.min()+1e-8)
    depth_img = (depth*255).astype('uint8')
    cv2.imwrite(OUT, depth_img)
    print("Depth saved:", OUT)

if __name__ == '__main__':
    run()
