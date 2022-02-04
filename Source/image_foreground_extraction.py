import numpy as np
import time
import cv2

image = cv2.imread("../InputData/image_f.png")
rect = cv2.selectROI(image)
image = image[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
width, height = image.shape[:2]
print(width, height)
rect = (0, 0, width, height)

iterCount = 5

mask = np.zeros(image.shape[:2], dtype="uint8")

fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")

start = time.time()
(mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
                                       fgModel, iterCount, mode=cv2.GC_INIT_WITH_RECT)
end = time.time()
print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

values = (
    ("Definite Background", cv2.GC_BGD),
    ("Probable Background", cv2.GC_PR_BGD),
    ("Definite Foreground", cv2.GC_FGD),
    ("Probable Foreground", cv2.GC_PR_FGD),
)

# loop over the possible GrabCut mask values
for (name, value) in values:
    print("[INFO] showing mask for '{}'".format(name))
    valueMask = (mask == value).astype("uint8") * 255
    #cv2.imshow(name, valueMask)
    #cv2.waitKey(0)

outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                      0, 1)
outputMask = (outputMask * 255).astype("uint8")
output = cv2.bitwise_and(image, image, mask=outputMask)

cv2.imshow("Input", image)
cv2.imshow("GrabCut Mask", outputMask)
cv2.imshow("GrabCut Output", output)
cv2.waitKey(0)
