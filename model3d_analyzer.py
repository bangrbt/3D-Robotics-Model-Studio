"""
Model 3D Analyzer - Core Python Engine
Phân tích cấu trúc, biến đổi hình học, so sánh các cách biểu diễn 3D và xuất kết quả.
"""

import os
import json
import math
import time
import numpy as np
import trimesh


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return super().default(obj)


class Model3DAnalyzer:
    """Lớp xử lý và phân tích mô hình 3D toàn diện cho HRI / Robotics."""

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.original_mesh = None
        self.current_mesh = None
        self.transform_history = []
        if filepath:
            self.load_model(filepath)

    def load_model(self, filepath):
        """Đọc mô hình 3D (STL, OBJ, PLY, v.v.) từ đường dẫn tệp."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

        self.filepath = filepath
        t0 = time.time()
        # Nạp mô hình bằng trimesh
        mesh = trimesh.load(filepath, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            # Nếu là Scene (nhiều mesh), ghép lại thành 1 Trimesh duy nhất
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
        
        self.original_mesh = mesh.copy()
        self.current_mesh = mesh.copy()
        self.transform_history = []
        load_time = time.time() - t0
        return {
            "status": "success",
            "file": os.path.basename(filepath),
            "load_time_sec": round(load_time, 3),
            "faces": len(self.current_mesh.faces),
            "vertices": len(self.current_mesh.vertices)
        }

    def analyze_structure(self):
        """Phân tích toàn diện cấu trúc hình học và tô-pô của mô hình 3D."""
        if self.current_mesh is None:
            raise ValueError("Chưa nạp mô hình 3D!")

        m = self.current_mesh
        file_size = os.path.getsize(self.filepath) if self.filepath and os.path.exists(self.filepath) else 0

        # Kích thước và Bounding Box
        bounds = m.bounds.tolist()  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        extents = m.extents.tolist()  # [dx, dy, dz]
        centroid = m.centroid.tolist()
        diag_length = float(np.linalg.norm(m.extents))

        # Diện tích bề mặt
        total_area = float(m.area)
        tri_areas = m.area_faces
        area_stats = {
            "min_mm2": float(np.min(tri_areas)),
            "max_mm2": float(np.max(tri_areas)),
            "mean_mm2": float(np.mean(tri_areas)),
            "std_mm2": float(np.std(tri_areas))
        }

        # Thể tích
        is_watertight = bool(m.is_watertight)
        mesh_volume = float(m.volume) if is_watertight else None
        hull = m.convex_hull
        hull_volume = float(hull.volume)

        # Tô-pô & Euler characteristic
        v_count = len(m.vertices)
        f_count = len(m.faces)
        e_count = len(m.edges_unique)
        euler_chi = v_count - e_count + f_count

        # Phân bố độ cao trục Z (Histogram 10 khoảng)
        z_vals = m.vertices[:, 2]
        z_hist, z_edges = np.histogram(z_vals, bins=10)
        z_dist = [
            {"range": f"[{round(z_edges[i], 1)}, {round(z_edges[i+1], 1)}]", "count": int(z_hist[i])}
            for i in range(10)
        ]

        # Mật độ lưới (triangles / cm^2)
        mesh_density = (f_count / (total_area / 100.0)) if total_area > 0 else 0.0

        return {
            "file_info": {
                "filename": os.path.basename(self.filepath) if self.filepath else "unknown",
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "format": os.path.splitext(self.filepath)[1].upper().replace(".", "") if self.filepath else "STL"
            },
            "mesh_counts": {
                "faces": f_count,
                "vertices": v_count,
                "edges_unique": e_count,
                "edges_total": len(m.edges),
                "euler_characteristic": euler_chi
            },
            "bounding_box": {
                "min": [round(x, 2) for x in bounds[0]],
                "max": [round(x, 2) for x in bounds[1]],
                "dimensions_mm": {
                    "width_x": round(extents[0], 2),
                    "depth_y": round(extents[1], 2),
                    "height_z": round(extents[2], 2)
                },
                "diagonal_length_mm": round(diag_length, 2),
                "centroid": [round(x, 2) for x in centroid]
            },
            "geometric_properties": {
                "surface_area_mm2": round(total_area, 2),
                "surface_area_cm2": round(total_area / 100.0, 2),
                "triangle_area_stats_mm2": area_stats,
                "mesh_density_tri_per_cm2": round(mesh_density, 1),
                "is_watertight": is_watertight,
                "is_convex": bool(m.is_convex),
                "mesh_volume_cm3": round(mesh_volume / 1000.0, 2) if mesh_volume else None,
                "convex_hull_volume_cm3": round(hull_volume / 1000.0, 2),
                "compactness": round((total_area ** 1.5) / (hull_volume if hull_volume > 0 else 1.0), 3)
            },
            "z_distribution": z_dist
        }

    def apply_transformations(self, translation=None, rotation_deg=None, scale=None, 
                              center=False, normalize=False, noise_std=0.0):
        """
        Thực hiện các phép biến đổi hình học (Geometric Transformations):
        - translation: [tx, ty, tz]
        - rotation_deg: [rx, ry, rz] theo độ quanh các trục X, Y, Z
        - scale: float s hoặc [sx, sy, sz]
        - center: đưa tâm bounding box về gốc tọa độ (0, 0, 0)
        - normalize: chuẩn hóa kích thước lớn nhất về 1.0 hoặc fit vào khối đơn vị
        - noise_std: độ lệch chuẩn nhiễu Gaussian cộng vào tọa độ đỉnh
        """
        if self.current_mesh is None:
            raise ValueError("Chưa nạp mô hình 3D!")

        m = self.current_mesh.copy()
        record = {}

        # 1. Căn tâm (Centering)
        if center:
            bbox_center = (m.bounds[0] + m.bounds[1]) / 2.0
            m.vertices -= bbox_center
            record["centered"] = True
            record["offset_removed"] = bbox_center.tolist()

        # 2. Chuẩn hóa kích thước (Normalization)
        if normalize:
            max_extent = float(np.max(m.extents))
            if max_extent > 0:
                scale_factor = 2.0 / max_extent  # đưa vào khoảng [-1, 1]
                m.vertices *= scale_factor
                record["normalized"] = True
                record["normalize_scale"] = scale_factor

        # 3. Phép co giãn (Scaling)
        if scale is not None:
            if isinstance(scale, (int, float)):
                sx, sy, sz = float(scale), float(scale), float(scale)
            else:
                sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
            
            # Ma trận scale 4x4
            scale_mat = np.diag([sx, sy, sz, 1.0])
            m.apply_transform(scale_mat)
            record["scale"] = [sx, sy, sz]

        # 4. Phép quay (Rotation theo Euler XYZ)
        if rotation_deg is not None:
            rx, ry, rz = [math.radians(a) for a in rotation_deg]
            # Ma trận quay Euler XYZ
            rot_mat = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
            m.apply_transform(rot_mat)
            record["rotation_deg"] = rotation_deg

        # 5. Phép tịnh tiến (Translation)
        if translation is not None:
            tx, ty, tz = [float(t) for t in translation]
            m.vertices += [tx, ty, tz]
            record["translation"] = [tx, ty, tz]

        # 6. Thêm nhiễu ngẫu nhiên (Gaussian Noise Injection)
        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, m.vertices.shape)
            m.vertices += noise
            record["noise_std"] = noise_std

        self.current_mesh = m
        self.transform_history.append(record)
        return {
            "status": "success",
            "applied": record,
            "new_bounds": self.current_mesh.bounds.tolist(),
            "new_extents": self.current_mesh.extents.tolist()
        }

    def reset_transformations(self):
        """Khôi phục lại hình dạng và vị trí ban đầu của mô hình."""
        if self.original_mesh is not None:
            self.current_mesh = self.original_mesh.copy()
            self.transform_history.clear()
            return {"status": "success", "message": "Đã khôi phục mô hình gốc"}
        return {"status": "error", "message": "Không có mô hình gốc"}

    def sample_point_cloud(self, point_count=20000):
        """Lấy mẫu đám mây điểm (Point Cloud) đồng đều trên bề mặt mô hình kèm vector pháp tuyến."""
        if self.current_mesh is None:
            raise ValueError("Chưa nạp mô hình 3D!")
        
        # Lấy mẫu theo phân bố diện tích bề mặt (area-weighted surface sampling)
        points, face_indices = trimesh.sample.sample_surface(self.current_mesh, point_count)
        normals = self.current_mesh.face_normals[face_indices]
        
        return {
            "points": points,
            "normals": normals,
            "count": len(points)
        }

    def voxelize(self, pitch=10.0):
        """Rời rạc hóa không gian mô hình thành mạng lưới Voxel (3D Occupancy Grid)."""
        if self.current_mesh is None:
            raise ValueError("Chưa nạp mô hình 3D!")
        
        # Tạo voxel grid
        v_grid = self.current_mesh.voxelized(pitch=pitch)
        centers = v_grid.points  # tọa độ tâm các voxel bị chiếm dụng (occupied)
        
        return {
            "pitch": pitch,
            "shape": list(v_grid.shape),
            "filled_count": int(v_grid.filled_count),
            "filled_volume_cm3": round(v_grid.filled_count * (pitch ** 3) / 1000.0, 2),
            "centers": centers
        }

    def compare_representations(self, sample_points=25000, voxel_pitch=8.0):
        """
        So sánh định lượng & định tính 4 cách biểu diễn 3D chính:
        1. Polygon Mesh (Lưới đa giác)
        2. Point Cloud (Đám mây điểm)
        3. Voxel Grid (Lưới thể tích occupancy)
        4. Wireframe / Edge Graph (Khung lưới cấu trúc)
        """
        if self.current_mesh is None:
            raise ValueError("Chưa nạp mô hình 3D!")

        m = self.current_mesh
        # 1. Mesh
        mesh_faces = len(m.faces)
        mesh_verts = len(m.vertices)
        mesh_mem_kb = round((m.vertices.nbytes + m.faces.nbytes + m.face_normals.nbytes) / 1024, 1)

        # 2. Point Cloud
        pc_data = self.sample_point_cloud(point_count=sample_points)
        pc_points = pc_data["count"]
        pc_mem_kb = round((pc_data["points"].nbytes + pc_data["normals"].nbytes) / 1024, 1)

        # 3. Voxel Grid
        vox_data = self.voxelize(pitch=voxel_pitch)
        vox_count = vox_data["filled_count"]
        vox_shape = vox_data["shape"]
        vox_mem_kb = round((vox_data["centers"].nbytes + (np.prod(vox_shape) // 8)) / 1024, 1)

        # 4. Wireframe
        wire_edges = len(m.edges_unique)
        wire_mem_kb = round((m.vertices.nbytes + m.edges_unique.nbytes) / 1024, 1)

        comparison_table = [
            {
                "representation": "Polygon Mesh (Lưới đa giác tam giác)",
                "type": "Surface Continuous (Bề mặt liên tục)",
                "elements_name": "Tam giác (Faces)",
                "elements_count": mesh_faces,
                "memory_kb": mesh_mem_kb,
                "memory_mb": round(mesh_mem_kb / 1024, 2),
                "continuity": "Liên tục, có thông tin topo & hướng pháp tuyến (Manifold/Surface)",
                "render_complexity": "O(F) - Tối ưu cực tốt nhờ GPU Vertex/Fragment Shaders phần cứng",
                "hri_robotics_use_case": "Mô hình CAD tay máy (Robot Gripper), va chạm động học Kinematics, mô phỏng vật lý PyBullet/MuJoCo, in 3D"
            },
            {
                "representation": "Point Cloud (Đám mây điểm)",
                "type": "Discrete Unorganized (Rời rạc không cấu trúc)",
                "elements_name": "Điểm 3D (Points + Normals)",
                "elements_count": pc_points,
                "memory_kb": pc_mem_kb,
                "memory_mb": round(pc_mem_kb / 1024, 2),
                "continuity": "Rời rạc, không có liên kết mặt, nhạy với nhiễu khoảng cách",
                "render_complexity": "O(N) - Render điểm hạt (Points/Particles), siêu nhẹ khi xử lý song song",
                "hri_robotics_use_case": "Dữ liệu thô từ cảm biến RGB-D Realsense/Kinect, quét LiDAR xe tự hành, thuật toán đăng ký ICP (Iterative Closest Point), nhận dạng cử chỉ tay 3D"
            },
            {
                "representation": "Voxel Grid (Mạng lưới thể tích 3D)",
                "type": "Volumetric Occupancy (Thể tích không gian)",
                "elements_name": "Khối Voxel (Occupied Cubes)",
                "elements_count": vox_count,
                "memory_kb": vox_mem_kb,
                "memory_mb": round(vox_mem_kb / 1024, 2),
                "continuity": "Rời rạc theo lưới đều (Regular grid), độ phân giải phụ thuộc bước pitch",
                "render_complexity": "O(V) - Có thể tối ưu bằng InstancedMesh hoặc Octree/SDF",
                "hri_robotics_use_case": "Bản đồ 3D tránh vật cản (OctoMap), lập quỹ đạo chuyển động tay máy (MoveIt/OMPL), mạng nơ-ron học sâu 3D (3D CNN/VoxNet)"
            },
            {
                "representation": "Wireframe (Khung dây / Edge Graph)",
                "type": "Topological Graph (Đồ thị cạnh)",
                "elements_name": "Đoạn thẳng (Edges)",
                "elements_count": wire_edges,
                "memory_kb": wire_mem_kb,
                "memory_mb": round(wire_mem_kb / 1024, 2),
                "continuity": "Cấu trúc cạnh khung xương, không có bề mặt che khuất",
                "render_complexity": "O(E) - Vẽ đường thẳng LineSegments, nhẹ nhàng cho CPU/low-spec devices",
                "hri_robotics_use_case": "Quan sát cấu trúc khung cơ khí tay gắp robot, hiển thị AR mờ đè lên camera thời gian thực, phân tích phần tử hữu hạn (FEA)"
            }
        ]

        return {
            "comparison": comparison_table,
            "voxel_params": {"pitch_mm": voxel_pitch, "grid_shape": vox_shape},
            "point_cloud_params": {"sampled_count": pc_points}
        }

    def export_transformed_mesh(self, output_path, file_format='stl'):
        """Xuất mô hình hiện tại (đã biến đổi) sang tệp STL, OBJ hoặc PLY."""
        if self.current_mesh is None:
            raise ValueError("Chưa nạp mô hình 3D!")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fmt = file_format.lower()
        if fmt in ('stl', 'obj', 'ply', 'off', 'glb', 'gltf'):
            self.current_mesh.export(output_path, file_type=fmt)
        else:
            self.current_mesh.export(output_path)

        return {
            "status": "success",
            "saved_file": output_path,
            "format": fmt,
            "size_bytes": os.path.getsize(output_path)
        }

    def export_point_cloud_ply(self, output_path, point_count=30000):
        """Xuất đám mây điểm đã lấy mẫu sang định dạng chuẩn PLY."""
        pc_data = self.sample_point_cloud(point_count=point_count)
        points = pc_data["points"]
        normals = pc_data["normals"]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Tạo trimesh PointCloud
        pcloud = trimesh.PointCloud(vertices=points)
        # Tính màu sắc dựa theo độ cao Z (gradient)
        z_vals = points[:, 2]
        z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-6)
        colors = np.zeros((len(points), 4), dtype=np.uint8)
        colors[:, 0] = (255 * z_norm).astype(np.uint8)       # R
        colors[:, 1] = (180 * (1 - z_norm)).astype(np.uint8) # G
        colors[:, 2] = 220                                    # B
        colors[:, 3] = 255                                    # Alpha
        pcloud.colors = colors

        pcloud.export(output_path)
        return {
            "status": "success",
            "saved_file": output_path,
            "point_count": len(points),
            "size_bytes": os.path.getsize(output_path)
        }

    def export_voxel_obj(self, output_path, pitch=12.0):
        """Xuất mô hình voxel dạng tập hợp khối hộp (Cubes) sang file OBJ."""
        vox_data = self.voxelize(pitch=pitch)
        centers = vox_data["centers"]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        boxes = []
        box_template = trimesh.creation.box(extents=[pitch * 0.9, pitch * 0.9, pitch * 0.9])
        for center in centers:
            b = box_template.copy()
            b.apply_translation(center)
            boxes.append(b)

        if boxes:
            combined = trimesh.util.concatenate(boxes)
            combined.export(output_path, file_type='obj')
            return {
                "status": "success",
                "saved_file": output_path,
                "voxel_cubes": len(boxes),
                "size_bytes": os.path.getsize(output_path)
            }
        return {"status": "error", "message": "Không có voxel nào"}

    def generate_full_report(self, json_path=None, md_path=None):
        """Tạo báo cáo phân tích toàn diện lưu dưới dạng JSON và Markdown."""
        structure = self.analyze_structure()
        comparison = self.compare_representations()

        report_data = {
            "title": "Báo Cáo Phân Tích Mô Hình 3D & So Sánh Cách Biểu Diễn",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "structure_analysis": structure,
            "representations_comparison": comparison,
            "transform_history": self.transform_history
        }

        # Lưu JSON
        if json_path:
            os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        # Lưu Markdown
        if md_path:
            os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
            md_content = self._format_markdown_report(report_data)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

        return report_data

    def _format_markdown_report(self, data):
        """Định dạng báo cáo ra Markdown chất lượng cao."""
        sa = data["structure_analysis"]
        bb = sa["bounding_box"]
        gp = sa["geometric_properties"]
        mc = sa["mesh_counts"]
        fi = sa["file_info"]
        comp = data["representations_comparison"]["comparison"]

        md = f"""# 📊 BÁO CÁO PHÂN TÍCH CẤU TRÚC & BIỂU DIỄN MÔ HÌNH 3D

**Mô hình:** `{fi['filename']}`  
**Thời gian tạo:** {data['timestamp']}  
**Dung lượng tệp:** {fi['file_size_mb']} MB ({fi['file_size_bytes']:,} bytes)  
**Định dạng:** {fi['format']}

---

## 1. 📐 Phân Tích Cấu Trúc Hình Học & Tô-pô (Structural Analysis)

### 1.1. Thống kê phần tử lưới
| Thuộc tính | Giá trị | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- |
| **Số mặt tam giác (Faces)** | **{mc['faces']:,}** | Mức độ chi tiết bề mặt của mô hình tay gắp robot |
| **Số đỉnh (Vertices)** | **{mc['vertices']:,}** | Điểm tọa độ không gian 3D cấu thành |
| **Số cạnh duy nhất (Unique Edges)** | **{mc['edges_unique']:,}** | Đoạn thẳng liên kết các đỉnh |
| **Đặc trưng Euler ($\chi = V - E + F$)** | **{mc['euler_characteristic']}** | Bất biến tô-pô hình học |
| **Tính khép kín (Watertight/Manifold)** | **{'Có (Khép kín)' if gp['is_watertight'] else 'Không (Tồn tại mép hở/nhiều cụm lắp ghép)'}** | Phù hợp cho tính toán thể tích và in 3D |

### 1.2. Kích thước & Hộp bao (Bounding Box)
- **Kích thước bao (Dx × Dy × Dz):** `{bb['dimensions_mm']['width_x']} mm × {bb['dimensions_mm']['depth_y']} mm × {bb['dimensions_mm']['height_z']} mm`
- **Tọa độ cực tiểu (Min Box):** `[{bb['min'][0]}, {bb['min'][1]}, {bb['min'][2]}]`
- **Tọa độ cực đại (Max Box):** `[{bb['max'][0]}, {bb['max'][1]}, {bb['max'][2]}]`
- **Đường chéo hộp bao:** `{bb['diagonal_length_mm']} mm`
- **Tâm hình học (Centroid):** `[{bb['centroid'][0]}, {bb['centroid'][1]}, {bb['centroid'][2]}]`

### 1.3. Đặc tính diện tích & Thể tích
- **Tổng diện tích bề mặt:** `{gp['surface_area_mm2']:,} mm²` (~ `{gp['surface_area_cm2']} cm²`)
- **Thể tích vỏ bao lồi (Convex Hull Volume):** `{gp['convex_hull_volume_cm3']} cm³` (~ `{round(gp['convex_hull_volume_cm3'] / 1000.0, 3)} Lít`)
- **Mật độ lưới:** `{gp['mesh_density_tri_per_cm2']} tam giác/cm²`
- **Diện tích tam giác nhỏ nhất:** `{gp['triangle_area_stats_mm2']['min_mm2']:.6f} mm²`
- **Diện tích tam giác trung bình:** `{gp['triangle_area_stats_mm2']['mean_mm2']:.4f} mm²`

---

## 2. 🔄 So Sánh Các Cách Biểu Diễn Mô Hình 3D (3D Representations Comparison)

Trong lĩnh vực **Tương tác Người - Robot (Human-Robot Interaction - HRI)** và Thị giác máy tính 3D, mô hình có thể được biểu diễn qua 4 dạng chính:

| Cách biểu diễn | Dạng dữ liệu | Số phần tử | Dung lượng RAM/Disk | Tính liên tục & Tô-pô | Vai trò trong HRI & Robotics |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for r in comp:
            md += f"| **{r['representation']}** | {r['type']} | {r['elements_count']:,} {r['elements_name']} | ~{r['memory_kb']} KB ({r['memory_mb']} MB) | {r['continuity']} | {r['hri_robotics_use_case']} |\n"

        md += """
---

## 3. 🛠️ Biến Đổi Hình Học (Geometric Transformations)

Các phép toán biến đổi đã được kiểm chứng và hỗ trợ trực quan:
1. **Tịnh tiến (Translation $\\vec{v}' = \\vec{v} + \\vec{t}$):** Dịch chuyển đối tượng đến vị trí làm việc hoặc gốc tọa độ tay máy.
2. **Phép quay (Rotation qua ma trận Euler $R = R_z(\\gamma) R_y(\\beta) R_x(\\alpha)$):** Định hướng tư thế (Pose) của tay gắp trong không gian thao tác.
3. **Phép co giãn (Scaling $S = \\text{diag}(s_x, s_y, s_z)$):** Phù hợp hóa kích thước mô hình theo tỷ lệ thực tế hoặc chuẩn hóa đơn vị.
4. **Căn tâm về gốc tọa độ (Centering to $[0, 0, 0]$):** Đưa tâm hình học hoặc tâm hộp bao về gốc để phục vụ điều khiển quay quanh tâm.
5. **Chuẩn hóa kích thước (Normalization):** Co dãn vừa vặn trong khối hộp chuẩn $[-1, 1]$ cho mạng nơ-ron học sâu (PointNet, 3D CNN).
6. **Thêm nhiễu ngẫu nhiên (Gaussian Noise Injection):** Kiểm thử tính bền vững (robustness) của thuật toán khớp mẫu và nhận dạng 3D.

---

## 4. 💾 Kết Quả Xuất File (Export Artifacts)

Mô hình và dữ liệu đã được lưu trữ sẵn sàng sử dụng:
- **Mesh STL đã biến đổi:** `results/3D_model_analysis/grasp_quad_transformed.stl`
- **Mesh OBJ đa giác chuẩn:** `results/3D_model_analysis/grasp_quad_transformed.obj`
- **Đám mây điểm Point Cloud (PLY):** `results/3D_model_analysis/grasp_quad_pointcloud.ply`
- **Mạng lưới Voxel 3D (OBJ Cubes):** `results/3D_model_analysis/grasp_quad_voxels.obj`
- **Báo cáo cấu trúc (JSON):** `results/3D_model_analysis/bao_cao_cau_truc_3d.json`
"""
        return md
