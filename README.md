# 3D Robotics Model Studio & Topological Analysis Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![WebGL 2.0](https://img.shields.io/badge/WebGL-2.0-blueviolet.svg)](https://get.webgl.org/webgl2/)
[![Three.js: r128](https://img.shields.io/badge/Three.js-r128-black.svg)](https://threejs.org/)
[![GitHub Pages](https://img.shields.io/badge/Deployment-GitHub%20Pages-success.svg)](https://pages.github.com/)

Nền tảng kỹ thuật trực quan hóa mô hình 3D thời gian thực, phân tích cấu trúc tô-pô, biến đổi hình học và so sánh định lượng 4 cách biểu diễn 3D phục vụ nghiên cứu **Tương tác Người - Robot (Human-Robot Interaction — HRI)** và Thao tác robot (Robotic Manipulation).

---

## 🌐 Trải Nghiệm Trực Tuyến (Live Web Demo)

Trang web tĩnh hoàn chỉnh sẵn sàng triển khai trên GitHub Pages, hỗ trợ 100% tương tác trên trình duyệt không cần cài đặt Python:
* **Giao diện chuẩn:** Nền trắng kỹ thuật (CAD Studio White Theme), biểu tượng vector SVG sắc nét, loại bỏ toàn bộ emoji, đạt chuẩn xuất bản khoa học.
* **Đường dẫn GitHub Pages:** `https://<your-username>.github.io/<repo-name>/` (sau khi kích hoạt Pages từ thư mục `docs/`).
* **Khởi chạy cục bộ:** Nhấp đúp [run_3d_studio.bat](file:///c:/HRI/run_3d_studio.bat) hoặc truy cập `http://localhost:8050`.

---

## 🎯 Tính Năng Cốt Lõi (Key Capabilities)

### 1. Trực Quan Hóa 3D Tương Tác (Interactive CAD Viewport)
- **Tốc độ khung hình cao:** Đạt 140–165 FPS với mô hình chi tiết hơn 426,500 tam giác.
- **Điều khiển góc nhìn:** Xoay 360° (Orbit), thu phóng (Zoom), trượt khung nhìn (Pan), phím tắt góc nhìn nhanh `Isometric`, `Front`, `Top`, `Right`.
- **Hệ thống đổ bóng kỹ thuật:** Đổi qua lại giữa `Titanium Studio PBR`, `Bản đồ cao độ Z (Elevation Heatmap)`, `Pháp tuyến (Normal RGB)`, và `Flat Shading`.
- **Mặt cắt ruột chi tiết (Section Slicer):** Trục cắt X, Y, Z kèm thanh trượt vị trí quan sát cơ cấu bánh răng và ngón gắp bên trong.
- **Thước đo & Hộp bao (Bounding Box):** Hiển thị trực quan kích thước $D_x, D_y, D_z$ và hệ trục tọa độ XYZ.

### 2. Phân Tích Cấu Trúc Tô-Pô (Topological & Geometric Analysis)
Phân tích trích xuất tự động qua động cơ [model3d_analyzer.py](file:///c:/HRI/model3d_analyzer.py):
* **Mô hình mục tiêu:** `grasp_quad_full_assembly_v7.STL` (Cụm tay gắp 4 ngón đa khớp).
* **Số mặt tam giác (Faces):** `426,500`
* **Số đỉnh tọa độ (Vertices):** `211,406`
* **Số cạnh phân đoạn (Edges):** `638,616`
* **Kích thước bao (Dx × Dy × Dz):** `315.06 mm × 316.72 mm × 286.69 mm`
* **Tâm hình học (Centroid):** `[13.60, -52.13, 0.73] mm` (phục vụ giải động học nghịch IK).
* **Tổng diện tích bề mặt:** `3,016.10 cm²` (phân tích diện tích tiếp xúc cầm nắm).
* **Thể tích vỏ lồi (Convex Hull):** `13.29 Lít` (phục vụ lập quy hoạch vùng an toàn tránh va chạm).
* **Biểu đồ phân bố cao độ:** Biểu đồ histogram 14 khoảng cao độ đỉnh thời gian thực.

### 3. Phép Toán Biến Đổi Hình Học (Geometric Transformations)
- **Tịnh tiến (Translation):** $T_x, T_y, T_z$ trong phạm vi $\pm 150\text{ mm}$.
- **Phép quay (Rotation):** Các góc Euler $R_x, R_y, R_z$ từ $-180^\circ$ đến $+180^\circ$.
- **Co giãn tỉ lệ (Scale):** Co dãn đồng dạng từ $0.1\times$ đến $3.0\times$.
- **Thao tác 1-chạm:** Căn tâm về gốc tọa độ `(0,0,0)`, Chuẩn hóa hộp bao `200mm`, Bơm nhiễu ngẫu nhiên Gaussian `±1.2mm`, Khôi phục gốc.

### 4. So Sánh Định Lượng 4 Cách Biểu Diễn 3D trong HRI

| Cách biểu diễn | Bản chất hình học | Số phần tử | Dung lượng RAM | Dung lượng File | Ứng dụng chủ đạo trong HRI & Robotics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🔷 Polygon Mesh** | Bề mặt liên tục (Manifold) | 426,500 tam giác | 24.36 MB | 20.3 MB (STL) | Mô hình CAD tay gắp, tính va chạm động học, in 3D |
| **🌌 Point Cloud** | Điểm rời rạc (Discrete) | 25,000 điểm | 1.14 MB (giảm 95%) | 390.9 KB (PLY) | Dữ liệu camera Realsense/LiDAR, nhận dạng cử chỉ tay |
| **🧊 Voxel Grid** | Khối thể tích (Occupancy) | 1,387 khối (12mm) | 0.03 MB (siêu nhẹ) | 302.9 KB (OBJ) | Bản đồ không gian 3D tránh vật cản (OctoMap), MoveIt |
| **📐 Wireframe** | Đồ thị cạnh (Graph) | 638,616 đoạn thẳng | 14.58 MB | 15.0 MB | Hiển thị AR thời gian thực, quan sát khung xương khớp |

### 5. Xuất Dữ Liệu Đa Định Dạng (Multi-Format Exporters)
Hỗ trợ tải trực tiếp từ trình duyệt hoặc ghi vào thư mục `results/3D_model_analysis/`:
- **STL (Binary):** Xuất mô hình lưới tam giác sau biến đổi.
- **Wavefront OBJ:** Xuất đa giác kèm vector pháp tuyến đỉnh.
- **Stanford PLY:** Xuất Point Cloud kèm màu sắc gradient cao độ.
- **Voxel OBJ:** Xuất tập hợp các khối voxel 3D.
- **Báo cáo JSON & Markdown:** Lưu toàn bộ thông số kỹ thuật phục vụ báo cáo khoa học.

---

## 📁 Cấu Trúc Thư Mục Dự Án (Repository Structure)

```
c:/HRI/
├── .github/
│   └── workflows/
│       └── deploy.yml              # Tự động triển khai GitHub Pages khi push main
├── 3D_model/
│   └── grasp_quad_full_assembly_v7.STL   # Tệp mô hình 3D gốc
├── docs/                           # Thư mục xuất bản GitHub Pages (Static Web)
│   ├── index.html                  # Giao diện web chuẩn nền trắng
│   ├── style.css                   # Định kiểu CAD Studio White Theme
│   ├── app.js                      # Mã nguồn Three.js (Hỗ trợ 100% offline & static)
│   ├── libs/                       # Thư viện Three.js, STLLoader, Exporters
│   ├── models/                     # Mô hình 3D phục vụ web tĩnh
│   └── data/                       # Dữ liệu phân tích JSON tĩnh (structure, voxels, compare)
├── results/
│   └── 3D_model_analysis/          # Kết quả phân tích và mô hình đã biến đổi
│       ├── grasp_quad_transformed.stl
│       ├── grasp_quad_transformed.obj
│       ├── grasp_quad_pointcloud.ply
│       ├── grasp_quad_voxels.obj
│       ├── bao_cao_cau_truc_3d.json
│       └── bao_cao_so_sanh_bieu_dien.md
├── web3d/                          # Giao diện web máy chủ cục bộ
├── model3d_analyzer.py             # Động cơ Python phân tích và biến đổi 3D
├── server_3d.py                    # Máy chủ HTTP / API cục bộ
├── run_3d_pipeline.py              # CLI tự động hóa phân tích, xuất file và mở web
├── run_3d_studio.bat               # File chạy nhanh 1-click trên Windows
├── requirements.txt                # Danh sách thư viện Python
└── README.md                       # Tài liệu hướng dẫn dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local Quickstart)

### Yêu cầu môi trường:
- Windows 10/11, macOS, hoặc Linux.
- Python 3.10+ (đã tích hợp virtualenv `.venv`).

### 1. Chạy nhanh bằng 1 cú nhấp đúp
Nhấp đúp vào tệp **`run_3d_studio.bat`**. Trình duyệt sẽ tự động mở giao diện tại `http://localhost:8050`.

### 2. Chạy qua dòng lệnh Terminal
```powershell
# Chạy toàn bộ pipeline phân tích và khởi động Web Studio:
.\.venv\Scripts\python.exe run_3d_pipeline.py

# Chỉ chạy phân tích và cập nhật các tệp xuất (không mở trình duyệt):
.\.venv\Scripts\python.exe run_3d_pipeline.py --no-server

# Khởi chạy riêng máy chủ Web Studio:
.\.venv\Scripts\python.exe server_3d.py
```

---

## 📤 Hướng Dẫn Đẩy Lên GitHub & Bật GitHub Pages

### Bước 1: Khởi tạo và Commit Git
```bash
git init
git branch -M main
git add .
git commit -m "feat: complete 3D Robotics Model Studio with white CAD theme and GitHub Pages"
```

### Bước 2: Liên kết kho lưu trữ GitHub của bạn
Tạo một repository mới trên GitHub (ví dụ: `3D-Robotics-Model-Studio`), sau đó chạy:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### Bước 3: Kích hoạt GitHub Pages để ai cũng truy cập được
1. Truy cập vào kho lưu trữ trên GitHub -> chọn tab **Settings**.
2. Chọn mục **Pages** ở cột menu bên trái.
3. Tại phần **Build and deployment**:
   - **Source:** Chọn `Deploy from a branch`
   - **Branch:** Chọn `main`, thư mục chọn `/docs`, rồi bấm **Save**.
   *(Hoặc chọn GitHub Actions nếu muốn chạy tự động theo file workflow `.github/workflows/deploy.yml` đã được cấu hình sẵn)*.
4. Sau 1 phút, GitHub sẽ cung cấp link công khai dạng:
   `https://<your-username>.github.io/<repo-name>/`
   Bất kỳ ai cũng có thể truy cập, quan sát, phân tích và xuất file 3D trực tiếp trên trình duyệt máy tính hoặc điện thoại!

---

## 📌 Phần Bổ Sung: Motion Studio — Robot Vẽ Theo Tay (HRI Air Drawing)

Hệ thống cử chỉ tay máy gốc phục vụ tương tác vẽ trong không gian:
* Nhấp đúp **`run.bat`**, đưa bàn tay vào khung camera.
* Chụm ngón cái và ngón trỏ để hạ bút vẽ nét tự do hoặc nhận dạng hình học phẳng (đường thẳng, tròn, vuông, tam giác, elip).
* Bật robot động học nghịch 2 khâu mô phỏng đầu bút vẽ theo nét di chuyển thực tế.

---

## 📄 Bản Quyền (License)
Dự án được phát hành theo giấy phép [MIT License](LICENSE).
Tác giả: **anhbang2810** (VNU - Vietnam National University).
