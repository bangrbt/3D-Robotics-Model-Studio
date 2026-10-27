"""
Server 3D Studio - Lightweight HTTP & API Server for 3D Model Studio
Serves WebGL Studio, handles 3D model streaming, structural analysis APIs, and exports.
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from model3d_analyzer import Model3DAnalyzer, NumpyEncoder

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB3D_DIR = os.path.join(WORKSPACE_DIR, "web3d")
MODEL_PATH = os.path.join(WORKSPACE_DIR, "3D_model", "grasp_quad_full_assembly_v7.STL")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results", "3D_model_analysis")

# Global analyzer instance
ANALYZER = None


def get_analyzer():
    global ANALYZER
    if ANALYZER is None:
        if os.path.exists(MODEL_PATH):
            ANALYZER = Model3DAnalyzer(MODEL_PATH)
        else:
            ANALYZER = Model3DAnalyzer()
    return ANALYZER


class Studio3DHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB3D_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS and disable caching for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. API: Get structural analysis
        if path == "/api/structure":
            try:
                analyzer = get_analyzer()
                data = analyzer.analyze_structure()
                self._send_json(data)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 2. API: Get representations comparison
        if path == "/api/comparison":
            try:
                analyzer = get_analyzer()
                data = analyzer.compare_representations()
                self._send_json(data)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 3. API: Get point cloud data
        if path == "/api/pointcloud":
            try:
                analyzer = get_analyzer()
                qs = urllib.parse.parse_qs(parsed.query)
                count = int(qs.get('count', [15000])[0])
                pc = analyzer.sample_point_cloud(point_count=count)
                # Convert to simple list format [x, y, z, nx, ny, nz]
                points = pc["points"].tolist()
                normals = pc["normals"].tolist()
                self._send_json({"count": len(points), "points": points, "normals": normals})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 4. API: Get voxel grid centers
        if path == "/api/voxels":
            try:
                analyzer = get_analyzer()
                qs = urllib.parse.parse_qs(parsed.query)
                pitch = float(qs.get('pitch', [10.0])[0])
                vox = analyzer.voxelize(pitch=pitch)
                self._send_json({
                    "pitch": vox["pitch"],
                    "shape": vox["shape"],
                    "filled_count": vox["filled_count"],
                    "filled_volume_cm3": vox["filled_volume_cm3"],
                    "centers": vox["centers"].tolist()
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 5. Model streaming route: /model/default.stl
        if path in ("/model/default.stl", "/3D_model/grasp_quad_full_assembly_v7.STL"):
            if os.path.exists(MODEL_PATH):
                self._serve_file(MODEL_PATH, "model/stl")
            else:
                self.send_error(404, "Default 3D model not found")
            return

        # 6. Serve exported files from results folder
        if path.startswith("/results/"):
            rel_path = path[len("/results/"):]
            full_path = os.path.join(RESULTS_DIR, rel_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                self._serve_file(full_path)
                return

        # Fallback to serving static files from web3d
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len).decode('utf-8') if content_len > 0 else "{}"
        try:
            params = json.loads(body)
        except Exception:
            params = {}

        # 1. API: Apply transformations
        if path == "/api/transform":
            try:
                analyzer = get_analyzer()
                translation = params.get("translation")
                rotation_deg = params.get("rotation_deg")
                scale = params.get("scale")
                center = params.get("center", False)
                normalize = params.get("normalize", False)
                noise_std = float(params.get("noise_std", 0.0))

                res = analyzer.apply_transformations(
                    translation=translation,
                    rotation_deg=rotation_deg,
                    scale=scale,
                    center=center,
                    normalize=normalize,
                    noise_std=noise_std
                )
                updated_structure = analyzer.analyze_structure()
                self._send_json({
                    "status": "success",
                    "transform_result": res,
                    "updated_structure": updated_structure
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 2. API: Reset transformations
        if path == "/api/reset_transform":
            try:
                analyzer = get_analyzer()
                res = analyzer.reset_transformations()
                updated_structure = analyzer.analyze_structure()
                self._send_json({
                    "status": "success",
                    "reset_result": res,
                    "updated_structure": updated_structure
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 3. API: Save and export results
        if path == "/api/save_exports":
            try:
                analyzer = get_analyzer()
                os.makedirs(RESULTS_DIR, exist_ok=True)
                
                stl_file = os.path.join(RESULTS_DIR, "grasp_quad_transformed.stl")
                obj_file = os.path.join(RESULTS_DIR, "grasp_quad_transformed.obj")
                ply_file = os.path.join(RESULTS_DIR, "grasp_quad_pointcloud.ply")
                vox_file = os.path.join(RESULTS_DIR, "grasp_quad_voxels.obj")
                json_file = os.path.join(RESULTS_DIR, "bao_cao_cau_truc_3d.json")
                md_file = os.path.join(RESULTS_DIR, "bao_cao_so_sanh_bieu_dien.md")

                analyzer.export_transformed_mesh(stl_file, 'stl')
                analyzer.export_transformed_mesh(obj_file, 'obj')
                analyzer.export_point_cloud_ply(ply_file, point_count=25000)
                analyzer.export_voxel_obj(vox_file, pitch=15.0)
                analyzer.generate_full_report(json_file, md_file)

                self._send_json({
                    "status": "success",
                    "message": "Đã lưu tất cả các định dạng xuất và báo cáo thành công!",
                    "files": [
                        "grasp_quad_transformed.stl",
                        "grasp_quad_transformed.obj",
                        "grasp_quad_pointcloud.ply",
                        "grasp_quad_voxels.obj",
                        "bao_cao_cau_truc_3d.json",
                        "bao_cao_so_sanh_bieu_dien.md"
                    ]
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self.send_error(404, "Endpoint not found")

    def _send_json(self, data, status=200):
        body = json.dumps(data, cls=NumpyEncoder, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path, content_type=None):
        try:
            size = os.path.getsize(file_path)
            self.send_response(200)
            if content_type:
                self.send_header('Content-Type', content_type)
            elif file_path.endswith('.stl'):
                self.send_header('Content-Type', 'model/stl')
            elif file_path.endswith('.obj'):
                self.send_header('Content-Type', 'text/plain')
            elif file_path.endswith('.ply'):
                self.send_header('Content-Type', 'application/octet-stream')
            elif file_path.endswith('.json'):
                self.send_header('Content-Type', 'application/json')
            elif file_path.endswith('.md'):
                self.send_header('Content-Type', 'text/markdown; charset=utf-8')
            
            self.send_header('Content-Length', str(size))
            self.end_headers()
            with open(file_path, 'rb') as f:
                # Stream file in 64KB chunks
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")


def run_server(port=8050):
    server_address = ('', port)
    httpd = HTTPServer(server_address, Studio3DHandler)
    print(f">> Studio 3D Server running at http://localhost:{port}")
    print(f">> Web directory: {WEB3D_DIR}")
    print(f">> Default 3D model: {MODEL_PATH}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n>> Stopping 3D Studio Server...")
        httpd.server_close()


if __name__ == '__main__':
    run_server(8050)
