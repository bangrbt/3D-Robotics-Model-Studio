"""
Pipeline Phân Tích & Khởi Chạy 3D Model Studio
Chạy phân tích cấu trúc, biến đổi hình học, trích xuất 4 cách biểu diễn,
lưu toàn bộ kết quả và khởi động giao diện 3D tương tác trên trình duyệt.
"""

import os
import sys
import argparse
import webbrowser
import time
from model3d_analyzer import Model3DAnalyzer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(WORKSPACE_DIR, "3D_model", "grasp_quad_full_assembly_v7.STL")
DEFAULT_EXPORT_DIR = os.path.join(WORKSPACE_DIR, "results", "3D_model_analysis")


def main():
    parser = argparse.ArgumentParser(description="Hệ thống phân tích, biến đổi & trực quan hóa 3D Model cho HRI")
    parser.add_argument("--file", default=DEFAULT_MODEL, help="Đường dẫn file 3D (STL, OBJ, PLY)")
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR, help="Thư mục xuất kết quả")
    parser.add_argument("--no-server", action="store_true", help="Chỉ phân tích và xuất file, không bật giao diện web")
    parser.add_argument("--port", type=int, default=8050, help="Cổng chạy server 3D Studio")
    args = parser.parse_args()

    model_path = os.path.abspath(args.file)
    export_dir = os.path.abspath(args.export_dir)
    os.makedirs(export_dir, exist_ok=True)

    print("=" * 72)
    print("💠 HỆ THỐNG PHÂN TÍCH, BIẾN ĐỔI & TRỰC QUAN HÓA MÔ HÌNH 3D ROBOTICS")
    print("=" * 72)
    print(f"📂 Tệp mô hình 3D: {os.path.basename(model_path)}")
    print(f"📁 Thư mục xuất:   {export_dir}")
    print("-" * 72)

    if not os.path.exists(model_path):
        print(f"❌ LỖI: Không tìm thấy tệp {model_path}")
        sys.exit(1)

    # 1. Khởi tạo và nạp mô hình
    print("⏳ [1/5] Đang nạp mô hình 3D và cấu trúc dữ liệu hình học...")
    t0 = time.time()
    analyzer = Model3DAnalyzer(model_path)
    load_time = time.time() - t0
    print(f"✅ Đã nạp thành công trong {load_time:.2f} giây!")

    # 2. Phân tích cấu trúc
    print("\n⏳ [2/5] Đang phân tích cấu trúc tô-pô, bounding box, diện tích & thể tích...")
    structure = analyzer.analyze_structure()
    fi = structure["file_info"]
    mc = structure["mesh_counts"]
    bb = structure["bounding_box"]
    gp = structure["geometric_properties"]

    print("   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │ 📊 THÔNG SỐ CẤU TRÚC: {fi['filename']:<37} │")
    print("   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ • Số mặt tam giác (Faces):     {mc['faces']:>12,}                │")
    print(f"   │ • Số đỉnh tọa độ (Vertices):   {mc['vertices']:>12,}                │")
    print(f"   │ • Số cạnh độc nhất (Edges):    {mc['edges_unique']:>12,}                │")
    print(f"   │ • Kích thước (X × Y × Z mm):   {bb['dimensions_mm']['width_x']:>6.1f} × {bb['dimensions_mm']['depth_y']:>6.1f} × {bb['dimensions_mm']['height_z']:>6.1f}    │")
    print(f"   │ • Tâm hình học (Centroid mm):  [{bb['centroid'][0]:>5.1f}, {bb['centroid'][1]:>5.1f}, {bb['centroid'][2]:>5.1f}]         │")
    print(f"   │ • Tổng diện tích bề mặt:       {gp['surface_area_cm2']:>12,.1f} cm²          │")
    print(f"   │ • Thể tích vỏ bao lồi:         {gp['convex_hull_volume_cm3']:>12,.1f} cm³          │")
    print(f"   │ • Tính khép kín (Watertight):  {'Có' if gp['is_watertight'] else 'Không (Cụm lắp ghép đa thân)':<28} │")
    print("   └─────────────────────────────────────────────────────────────┘")

    # 3. So sánh 4 cách biểu diễn 3D
    print("\n⏳ [3/5] Đang so sánh định lượng 4 cách biểu diễn mô hình 3D...")
    comparison = analyzer.compare_representations(sample_points=25000, voxel_pitch=12.0)
    print("   " + "-" * 67)
    print(f"   {'Cách biểu diễn':<24} | {'Phần tử':<14} | {'Bộ nhớ RAM':<10} | {'Đặc trưng'}")
    print("   " + "-" * 67)
    for c in comparison["comparison"]:
        name_short = c["representation"].split("(")[0].strip()
        elem_str = f"{c['elements_count']:,} {c['elements_name'].split()[0]}"
        mem_str = f"{c['memory_mb']} MB"
        type_str = c["type"].split("(")[0].strip()
        print(f"   {name_short:<24} | {elem_str:<14} | {mem_str:<10} | {type_str}")
    print("   " + "-" * 67)

    # 4. Thực hiện phép biến đổi hình học mẫu & xuất file
    print("\n⏳ [4/5] Đang thực hiện biến đổi hình học mẫu và xuất file...")
    # Biến đổi: Căn tâm về [0,0,0], quay Euler 30 độ quanh trục Z, scale 0.8
    analyzer.apply_transformations(
        center=True,
        rotation_deg=[0, 0, 30],
        scale=0.8
    )

    stl_out = os.path.join(export_dir, "grasp_quad_transformed.stl")
    obj_out = os.path.join(export_dir, "grasp_quad_transformed.obj")
    ply_out = os.path.join(export_dir, "grasp_quad_pointcloud.ply")
    vox_out = os.path.join(export_dir, "grasp_quad_voxels.obj")
    json_out = os.path.join(export_dir, "bao_cao_cau_truc_3d.json")
    md_out = os.path.join(export_dir, "bao_cao_so_sanh_bieu_dien.md")

    analyzer.export_transformed_mesh(stl_out, 'stl')
    print(f"   ✅ Đã xuất Mesh STL:         {os.path.basename(stl_out)} ({os.path.getsize(stl_out)/(1024*1024):.1f} MB)")
    analyzer.export_transformed_mesh(obj_out, 'obj')
    print(f"   ✅ Đã xuất Mesh OBJ:         {os.path.basename(obj_out)} ({os.path.getsize(obj_out)/(1024*1024):.1f} MB)")
    analyzer.export_point_cloud_ply(ply_out, point_count=25000)
    print(f"   ✅ Đã xuất Point Cloud PLY:  {os.path.basename(ply_out)} ({os.path.getsize(ply_out)/1024:.1f} KB)")
    analyzer.export_voxel_obj(vox_out, pitch=15.0)
    print(f"   ✅ Đã xuất Voxel Grid OBJ:   {os.path.basename(vox_out)} ({os.path.getsize(vox_out)/1024:.1f} KB)")
    analyzer.generate_full_report(json_out, md_out)
    print(f"   ✅ Đã xuất Báo cáo JSON:     {os.path.basename(json_out)}")
    print(f"   ✅ Đã xuất Báo cáo Markdown: {os.path.basename(md_out)}")

    if args.no_server:
        print("\n🎉 Hoàn thành quá trình phân tích và xuất file!")
        return

    # 5. Khởi động WebGL 3D Studio Server
    url = f"http://localhost:{args.port}"
    print(f"\n🚀 [5/5] Đang khởi động 3D Web Studio tại {url}...")
    print("   (Trình duyệt sẽ tự động mở sau 1 giây. Nhấn Ctrl+C để dừng server)")
    
    # Mở trình duyệt web
    webbrowser.open(url)

    # Chạy HTTP Server
    from server_3d import run_server
    run_server(args.port)


if __name__ == "__main__":
    main()
