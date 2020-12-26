import cv2
import numpy as np

video = cv2.VideoCapture("../InputData/cc2fCut.avi")
template = cv2.imread('../InputData/template_image_2.png', 0)

i = 0
while (video.isOpened()):
    ret, img_rgb = video.read()
    if ret == False:
        break
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('kang' + str(i) + '.jpg', frame)
    i += 1

    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    print(loc)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    cv2.imshow('Detected', img_rgb)
    cv2.waitKey(30000)
    if i == 1:
        break

video.release()
cv2.destroyAllWindows()

img_rgb = cv2.imread('../InputData/original_image.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('../InputData/template_image.png', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

print(loc)
cv2.imshow('Detected', img_rgb)
cv2.waitKey(30000)
