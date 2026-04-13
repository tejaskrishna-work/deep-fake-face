import cv2

img = cv2.imread("assets/overlays/mask.png", cv2.IMREAD_UNCHANGED)

print(img.shape)