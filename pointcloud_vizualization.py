import open3d as o3d
import numpy as np

# Загрузка облака точек 
def load_point_cloud(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                x, y, z = map(float, parts[:3])
                points.append([x, y, z])
    return np.array(points)

# Загрузка траектории (TUM) 
def load_trajectory(filename):
    positions = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                # Формат: timestamp tx ty tz qx qy qz qw
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                positions.append([tx, ty, tz])
    return np.array(positions)

# Основной код
point_cloud_file = r"C:\Users\user\Desktop\9НИЦ\файлы с убунту\траектории и облака точек\путь1\point_cloud.txt"
trajectory_file = r"C:\Users\user\Desktop\9НИЦ\файлы с убунту\траектории и облака точек\путь1\CameraTrajectory.txt"

# Загружаем данные
points = load_point_cloud(point_cloud_file)
traj_points = load_trajectory(trajectory_file)
print(f"Загружено облако: {len(points)} точек")
print(f"Загружена траектория: {len(traj_points)} позиций")

# Создаём объекты Open3D 
# 1. Облако точек
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 2. Траектория как линия (LineSet)
# Соединяем последовательные точки траектории
if len(traj_points) >= 2:
    lines = [[i, i+1] for i in range(len(traj_points)-1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(traj_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Цвет линии (красный)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])

# 3. Точки траектории 
traj_pcd = o3d.geometry.PointCloud()
traj_pcd.points = o3d.utility.Vector3dVector(traj_points)
traj_pcd.paint_uniform_color([1, 0, 0])  # красные точки

#  Визуализация 
# Собираем все геометрии
geometries = [pcd, line_set, traj_pcd]
# Создаём окно визуализатора с настройками
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Проезд №1", width=1024, height=768)

for geo in geometries:
    if geo is not None:
        vis.add_geometry(geo)

# Настраиваем опции рендеринга (размер точек)
opt = vis.get_render_option()
opt.point_size = 1.5
opt.line_width = 1.0

vis.run()
vis.destroy_window()

all_points = points
all_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points) * 0.7

# Добавляем точки траектории
if len(traj_points) > 0:
    all_points = np.vstack([all_points, traj_points])
    # Для траектории добавляем красный цвет
    traj_colors = np.tile([1.0, 0.0, 0.0], (len(traj_points), 1))
    all_colors = np.vstack([all_colors, traj_colors])

# Создаём новое облако
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(all_points)
combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

# Сохраняем в .ply
output_file = "combined.ply"
o3d.io.write_point_cloud(output_file, combined_pcd)
print(f"Сохранено в {output_file} (точек: {len(all_points)})")
