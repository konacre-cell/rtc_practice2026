import open3d as o3d
import numpy as np


# Загрузка данных
def load_point_cloud(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 3:
                points.append(list(map(float, parts[:3])))
    return np.array(points)


def load_trajectory(filename):
    positions = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 4:
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(positions)


#  Математические модули

def fit_circle_ransac(u_coords, v_coords, max_expected_radius, iterations=1000, threshold=0.03):
    """Единый RANSAC с защитой от вырождения """
    best_inliers_count = 0
    best_circle = None
    best_mask = None

    n_pts = len(u_coords)
    if n_pts < 3: return None

    pts = np.column_stack((u_coords, v_coords))

    for _ in range(iterations):
        idx = np.random.choice(n_pts, 3, replace=False)
        sample = pts[idx]

        A = np.column_stack((sample[:, 0], sample[:, 1], np.ones(3)))
        B = sample[:, 0] ** 2 + sample[:, 1] ** 2
        try:
            res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            uc_temp = res[0] / 2.0
            vc_temp = res[1] / 2.0

            R_sq = res[2] + uc_temp ** 2 + vc_temp ** 2
            if R_sq < 0: continue
            R_temp = np.sqrt(R_sq)

            if R_temp > max_expected_radius * 1.3:
                continue

        except np.linalg.LinAlgError:
            continue

        dist_to_center = np.sqrt((u_coords - uc_temp) ** 2 + (v_coords - vc_temp) ** 2)
        deviations = np.abs(dist_to_center - R_temp)

        inlier_mask = deviations <= threshold
        inliers_count = np.sum(inlier_mask)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_circle = (uc_temp, vc_temp)
            best_mask = inlier_mask

    if best_circle is None: return None

    u_inliers = u_coords[best_mask]
    v_inliers = v_coords[best_mask]

    A = np.column_stack((u_inliers, v_inliers, np.ones_like(u_inliers)))
    B = u_inliers ** 2 + v_inliers ** 2
    try:
        res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        uc_final = res[0] / 2.0
        vc_final = res[1] / 2.0
        R_sq_final = res[2] + uc_final ** 2 + vc_final ** 2
        if R_sq_final < 0: return None
        R_final = np.sqrt(R_sq_final)

        # Защита финального радиуса МНК от захвата фоновых плоскостей
        if R_final > max_expected_radius * 1.3:
            return None

        return uc_final, vc_final, R_final, best_mask
    except np.linalg.LinAlgError:
        return None


def quick_find_center(points, P1, n_vector, expected_radius=0.5):
    """Первый проход с использованием RANSAC для защиты от неполных дуг """
    vectors_P1 = points - P1
    dot_p1 = np.dot(vectors_P1, n_vector)
    mask = (dot_p1 >= -0.05) & (dot_p1 <= 0.05)
    sliced = points[mask]
    if len(sliced) < 3: return None

    ref_vec = np.array([1.0, 0.0, 0.0]) if abs(n_vector[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(n_vector, ref_vec)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(n_vector, u_axis)

    pts_shifted = sliced - P1
    u = np.dot(pts_shifted, u_axis)
    v = np.dot(pts_shifted, v_axis)

    # Расширенный адаптивный радиус (компенсация смещения оператора)
    search_radius = expected_radius + 0.5
    dist_from_P1 = np.sqrt(u ** 2 + v ** 2)
    mask_radius = dist_from_P1 <= search_radius
    u = u[mask_radius]
    v = v[mask_radius]

    if len(u) < 3: return None

    # Используем быструю итерацию RANSAC для игнорирования неравномерных скоплений точек
    ransac_res = fit_circle_ransac(u, v, max_expected_radius=expected_radius, iterations=200, threshold=0.05)
    if ransac_res is None: return None

    uc, vc, _, _ = ransac_res
    return P1 + uc * u_axis + vc * v_axis


def estimate_perfect_pipe_axis(points, traj_points, center_idx, max_radius=0.5, window_size=5, iterations=2):
    start_idx = max(0, center_idx - window_size)
    end_idx = min(len(traj_points) - 1, center_idx + window_size)

    n_normals = []
    for i in range(start_idx, end_idx):
        if i + 1 >= len(traj_points): break
        v = traj_points[i + 1] - traj_points[i]
        n_normals.append(v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([0., 0., 1.]))

    true_axis = None
    centroid = None

    for it in range(iterations):
        centers = []
        for k, i in enumerate(range(start_idx, end_idx)):
            if i + 1 >= len(traj_points): break
            P_curr = traj_points[i]

            n_eval = true_axis if true_axis is not None else n_normals[k]

            c3d = quick_find_center(points, P_curr, n_eval, expected_radius=max_radius)
            if c3d is not None:
                centers.append(c3d)

        if len(centers) < 3:
            return None, None

        centers = np.array(centers)
        centroid = np.mean(centers, axis=0)
        centered_centers = centers - centroid
        _, _, Vh = np.linalg.svd(centered_centers)

        true_axis = Vh[0]

        v_local = traj_points[min(center_idx + 1, len(traj_points) - 1)] - traj_points[center_idx]
        if np.linalg.norm(v_local) == 0 and center_idx > 0:
            v_local = traj_points[center_idx] - traj_points[center_idx - 1]

        if np.dot(true_axis, v_local) < 0:
            true_axis = -true_axis

    return true_axis / np.linalg.norm(true_axis), centroid


# Главная функция анализа

def analyze_pipe_section_with_ransac(points, traj_points, idx1, idx2, max_radius=0.5, ransac_threshold=0.03,
                                     thickness=None):
    P1 = traj_points[idx1]
    P2 = traj_points[idx2]

    v_traj = P2 - P1
    v_traj_len = np.linalg.norm(v_traj)

    if thickness is None and idx2 + 1 < len(traj_points):
        v_next = traj_points[idx2 + 1] - P2
        if np.linalg.norm(v_traj) > 0 and np.linalg.norm(v_next) > 0:
            cos_curve = np.dot(v_traj, v_next) / (np.linalg.norm(v_traj) * np.linalg.norm(v_next))
            curve_angle = np.degrees(np.arccos(np.clip(cos_curve, -1.0, 1.0)))
            if curve_angle > 3.0:
                print(f"Обнаружен изгиб траектории ({curve_angle:.1f}°). Включен тонкий срез.")
                thickness = 0.02

    # [ИСПРАВЛЕНИЕ 3]: Предупреждение об игнорировании P2 в режиме тонкого среза
    if thickness is not None:
        print(f"ВНИМАНИЕ: Активирован режим тонкого среза (±{thickness} м).")
        print(f"Анализируется только локальная окрестность точки P1 (idx {idx1}). "
              f"Точка P2 (idx {idx2}) используется только для вектора направления.\n")

    n_pipe, centroid = estimate_perfect_pipe_axis(points, traj_points, idx1, max_radius=max_radius, window_size=7)

    if n_pipe is None:
        print("Внимание: Не удалось стабилизировать ось. Fallback на вектор траектории.")
        n_pipe = v_traj / v_traj_len if v_traj_len > 0 else np.array([0.0, 0.0, 1.0])
        rough_center = P1
    else:
        t = np.dot(P1 - centroid, n_pipe)
        rough_center = centroid + t * n_pipe

    cos_theta = np.dot(v_traj, n_pipe) / (v_traj_len * np.linalg.norm(n_pipe)) if v_traj_len > 0 else 1.0
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    vectors_P1 = points - P1
    dot_p1 = np.dot(vectors_P1, n_pipe)

    upper_bound = np.dot(v_traj, n_pipe)
    if thickness is None:
        if abs(upper_bound) < 0.02:
            thickness = 0.02
        else:
            if upper_bound >= 0:
                mask_between = (dot_p1 >= 0) & (dot_p1 <= upper_bound)
            else:
                mask_between = (dot_p1 >= upper_bound) & (dot_p1 <= 0)

    if thickness is not None:
        mask_between = (dot_p1 >= -thickness) & (dot_p1 <= thickness)

    indices_between = np.where(mask_between)[0]
    sliced_points = points[mask_between]

    if len(sliced_points) < 3:
        print("Ошибка: недостаточно точек в срезе.")
        return None

    ref_vec = np.array([1.0, 0.0, 0.0]) if abs(n_pipe[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(n_pipe, ref_vec)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(n_pipe, u_axis)

    u_coords = np.dot(sliced_points - rough_center, u_axis)
    v_coords = np.dot(sliced_points - rough_center, v_axis)

    dist_from_axis = np.sqrt(u_coords ** 2 + v_coords ** 2)
    mask_radius = dist_from_axis <= max_radius * 1.5  # Расширенный фильтр перед RANSAC

    u_coords = u_coords[mask_radius]
    v_coords = v_coords[mask_radius]
    indices_between = indices_between[mask_radius]
    sliced_points = sliced_points[mask_radius]

    if len(sliced_points) < 3: return None

    ransac_result = fit_circle_ransac(u_coords, v_coords, max_expected_radius=max_radius, iterations=1000,
                                      threshold=ransac_threshold)

    if ransac_result is None: return None

    uc, vc, R_fitted, mask_inliers = ransac_result

    indices_between = indices_between[mask_inliers]
    u_coords = u_coords[mask_inliers]
    v_coords = v_coords[mask_inliers]

    trajectory_offset = np.linalg.norm(P1 - rough_center)
    ransac_drift = np.sqrt(uc ** 2 + vc ** 2)

    # [ИСПРАВЛЕНИЕ 6]: Предупреждение о нецилиндрической геометрии трубы
    if ransac_drift > 0.05:
        print(
            f"ПРЕДУПРЕЖДЕНИЕ: Геометрия нецилиндрическая или сильно искажена. RANSAC дрейф = {ransac_drift:.4f} м. !!!")

    # [ИСПРАВЛЕНИЕ 2]: Жесткая привязка визуализации к идеальной SVD-оси
    angles = np.arctan2(v_coords, u_coords)  # Углы инлаеров относительно rough_center
    proj_u = R_fitted * np.cos(angles)
    proj_v = R_fitted * np.sin(angles)

    projections_3d = rough_center + np.outer(proj_u, u_axis) + np.outer(proj_v, v_axis)

    final_green_mask = np.zeros(len(points), dtype=bool)
    final_green_mask[indices_between] = True

    print(f"Радиус трубы примерно: {R_fitted*0.6165:.3f} м.")
    print(f"Поперечное смещение траектории (P1) примерно: {trajectory_offset*0.6165:.2f} м.")
    print(f"Угол между траекторией и осью: {angle_deg:.2f}°")
    print(f"RANSAC дрейф примерно: {ransac_drift*0.6165:.2f} м.")


    return {
        "green_mask": final_green_mask,
        "projections": projections_3d,
        "P1": P1, "P2": P2,
        "true_center": rough_center,
        "uc": uc, "vc": vc
    }


# ------------------- Основной скрипт -------------------
point_cloud_file = r"C:\Users\user\Desktop\9НИЦ\файлы с убунту\траектории и облака точек\путь1\point_cloud.txt"
trajectory_file = r"C:\Users\user\Desktop\9НИЦ\файлы с убунту\траектории и облака точек\путь1\CameraTrajectory.txt"

points = load_point_cloud(point_cloud_file)
traj_points = load_trajectory(trajectory_file)

print(f"Загружено облако: {len(points)} точек")
print(f"Загружена траектория: {len(traj_points)} позиций")

TRAJ_IDX_1 = 200
TRAJ_IDX_2 = 400
MAX_PIPE_RADIUS = 0.5
RANSAC_THRESHOLD = 0.03
THICKNESS_MODE = None

result = analyze_pipe_section_with_ransac(points, traj_points, TRAJ_IDX_1, TRAJ_IDX_2,
                                          max_radius=MAX_PIPE_RADIUS,
                                          ransac_threshold=RANSAC_THRESHOLD,
                                          thickness=THICKNESS_MODE)

# ------------------- Создаём объекты Open3D -------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
colors = np.ones((len(points), 3)) * 0.7

line_set = None
if len(traj_points) >= 2:
    lines = [[i, i + 1] for i in range(len(traj_points) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(traj_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])

traj_pcd = o3d.geometry.PointCloud()
traj_pcd.points = o3d.utility.Vector3dVector(traj_points)
traj_pcd.paint_uniform_color([1, 0, 0])

geometries = [pcd, line_set, traj_pcd]

if result is not None:
    colors[result["green_mask"]] = [0.0, 0.8, 0.0]

    proj_pcd = o3d.geometry.PointCloud()
    proj_pcd.points = o3d.utility.Vector3dVector(result["projections"])
    proj_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    geometries.append(proj_pcd)

    for p_coord in [result["P1"], result["P2"]]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1.0, 0.8, 0.0])
        sphere.translate(p_coord)
        geometries.append(sphere)

    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    center_sphere.compute_vertex_normals()
    center_sphere.paint_uniform_color([0.0, 1.0, 1.0])
    center_sphere.translate(result["true_center"])
    geometries.append(center_sphere)

pcd.colors = o3d.utility.Vector3dVector(colors)

# Визуализация
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Open3D Pipeline - Анализ сечения", width=1024, height=768)

for geo in geometries:
    if geo is not None: vis.add_geometry(geo)

opt = vis.get_render_option()
opt.point_size = 1.5
opt.line_width = 1.0

vis.run()
vis.destroy_window()
