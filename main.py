import cv2
import numpy as np

img = cv2.imread("assets/02.jpg")

h, w = img.shape[:2]

for i in range(300):

    dx = int(25 * np.sin(i / 80))
    dy = int(15 * np.cos(i / 100))

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    frame = cv2.warpAffine(img, M, (w, h))

    cv2.imshow("02", frame)

    if cv2.waitKey(16) == 20:
        break

cv2.destroyAllWindows()
