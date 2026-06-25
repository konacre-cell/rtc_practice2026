import numpy as np
import cv2
import glob


CHESSBOARD_SIZE = (9, 6)          # внутренние углы (ширина, высота) — доска 10x7 клеток
SQUARE_SIZE = 1.0                # размер одной клетки в метрах (юнитах Unity)
IMAGE_FOLDER = "*.png"


# 3D точки идеальной доски
pattern_points = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
pattern_points[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
pattern_points *= SQUARE_SIZE

obj_points = []
img_points = []

images = glob.glob(IMAGE_FOLDER)
if not images:
    print(" Нет изображений в папке", IMAGE_FOLDER)
    exit()

print(f" Найдено файлов: {len(images)}")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        obj_points.append(pattern_points)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        img_points.append(corners2)
        print(f" Углы найдены: {fname}")
    else:
        print(f" Углы НЕ найдены: {fname}")

if len(obj_points) < 5:
    print(" Слишком мало кадров с углами, Нужно минимум 5.")
    exit()


# калибровка
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points,
    gray.shape[::-1],
    None, None
)


# вывод параметров
print("\n" + "="*50)
print("КАЛИБРОВКА ЗАВЕРШЕНА")
print("="*50)
print("\n МАТРИЦА КАМЕРЫ (3x3):")
print(mtx)
print("\n КОЭФФИЦИЕНТЫ ДИСТОРСИИ [k1, k2, p1, p2, k3]:")
print(dist.ravel())
print("\n Средняя ошибка репроекции (пиксели):")
mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print(f"{mean_error/len(obj_points):.3f}")
