# ==========================================================================================
# PARCIAL FINAL CRISTHIAN ALEJANDRO ROJAS
# ==========================================================================================

import cv2
import os
import sys
from matplotlib import pyplot as plt
import numpy as np

# PUNTO NUMERO 1
# Estimar el porcentaje de pixeles que corresponden al cesped

if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image3 = image.copy()

    # Hue histogram
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    # plt.plot(hist, color='green')
    # plt.ylabel('cantidad de pixeles')
    # plt.show()
    # cv2.destroyAllWindows()

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])

    # Hue histogram max and location of max
    max_val = hist_hue.max()
    max_pos = int(hist_hue.argmax())

    # Peak mask
    lim_inf = (max_pos - 10, 0, 0)
    lim_sup = (max_pos + 10, 255, 255)
    mask = cv2.inRange(image_hsv, lim_inf, lim_sup)


    no_white_pix = np.sum(mask == 0)
    white_pix = np.sum(mask == 255)
    percentage = white_pix / (no_white_pix + white_pix)
    print('El porcentaje de pixeles de cesped es de la imagen: ', percentage )
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1280, 720)
    cv2.imshow("Image", mask)
    cv2.waitKey(0)


#PUNTO NUMERO 2
    image_draw = image.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def auto_canny(image, s=0.033):
        # calcular la mediana de las intensidades de píxeles de un solo canal
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        # aplicar la detección automática de bordes Canny utilizando la mediana calculada
        inferior = int(max(0, (1.0 - s) * v))
        superior = int(min(255, (1.0 + s) * v))
        edged = cv2.Canny(image, inferior, superior)
        # devolver la imagen con bordes
        return edged


    canny = auto_canny(mask)

    ret, Ibw_shapes = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(Ibw_shapes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contador = 0
    for idx, i in enumerate(contours):
        color = (0, 0, 255)

        area = cv2.contourArea(contours[idx])

        if area > 9.0:
            x, y, w, h = cv2.boundingRect(contours[idx])
            cv2.rectangle(image_draw, (x, y), (x + w, y + h), (0, 0, 255), 3)
        contador += 1
    print('El total de jugadores en el campo ',contador)

    cv2.imshow("Image", image_draw)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




# PUNTO NO3.




points = []


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

if __name__ == '__main__':


    points1 = []
    points2 = []

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1280, 720)
    cv2.setMouseCallback("Image", click)

    point_counter = 0
    while True:

        cv2.imshow("Image", image3)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points1 = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)
            cv2.circle(image3, (points[-1][0], points[-1][1]), 3, [0, 0, 255], -1)

    point_counter = 0
    while True:
        cv2.imshow("Image", image3)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points2 = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)
            cv2.circle(image3, (points[-1][0], points[-1][1]), 3, [255, 0, 0], -1)

    N = min(len(points1), len(points2))
    assert N >= 1, 'At least 3 points are required'

    pts1 = np.array(points1[:N])
    pts2 = np.array(points2[:N])
    # Green color in BGR
    color = (255, 0, 255)
    # Line thickness
    thickness = 10
    print(points1)
    image3 = cv2.line(image3, points1[0], points1[1], color, thickness)

    cv2.imshow("Image", image3)
    cv2.waitKey(0)

    from _7_lines_detection.hough import Hough

    hough = Hough(bw_edges)
    if method == Methods.Standard:
        accumulator = hough.standard_transform()
    elif method == Methods.Direct:
        image_gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
        theta, _ = gradient_map(image_gray)
        accumulator = hough.direct_transform(theta)
    else:
        sys.exit()

    acc_thresh = 50
    N_peaks = 11
    nhood = [25, 9]
    peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

    _, cols = image3.shape[:2]
    image_draw = np.copy(image3)

    for peak in peaks:
        rho = peak[0]
        theta_ = hough.theta[peak[1]]

        theta_pi = np.pi * theta_ / 180
        theta_ = theta_ - 180
        a = np.cos(theta_pi)
        b = np.sin(theta_pi)
        x0 = a * rho + hough.center_x
        y0 = b * rho + hough.center_y
        c = -rho
        x1 = int(round(x0 + cols * (-b)))
        y1 = int(round(y0 + cols * a))
        x2 = int(round(x0 - cols * (-b)))
        y2 = int(round(y0 - cols * a))

        if np.abs(theta_) < 80:
            image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
        elif np.abs(theta_) > 100:
            image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
        else:
            if theta_ > 0:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
            else:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)
    cv2.imshow("lines", image_draw)
    cv2.waitKey(0)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
