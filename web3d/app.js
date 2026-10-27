/**
 * 3D Robotics Model Studio - Core WebGL Application Logic
 * Clean White / Light CAD Studio Theme with Static GitHub Pages & Local Server Support.
 */

(function () {
  'use strict';

  // --- STATE ---
  const state = {
    activeRep: 'mesh',          // 'mesh', 'pointcloud', 'voxel', 'wireframe'
    matStyle: 'metallic',       // 'metallic', 'heatmap', 'normals', 'matte'
    wireOverlay: false,
    flatShading: false,
    metalness: 0.75,
    roughness: 0.35,
    pointSize: 2.0,
    voxelPitch: 12.0,
    
    // Clipping plane (Slicer)
    sliceEnabled: false,
    sliceAxis: 'x',
    slicePos: 0,
    
    // View Guides
    showGrid: true,
    showAxes: true,
    showBBox: false,
    autoRotate: false,
    
    // Transforms
    translation: { x: 0, y: 0, z: 0 },
    rotationDeg: { x: 0, y: 0, z: 0 },
    scale: 1.0,
    
    // Model Data
    originalGeometry: null,
    currentGeometry: null,
    modelName: 'grasp_quad_full_assembly_v7.STL',
    analysisData: null,
    comparisonData: null,
    voxelData: null,
    
    // FPS counter
    fps: 60,
    frames: 0,
    lastTime: performance.now()
  };

  // --- THREE.JS OBJECTS ---
  let scene, camera, renderer, controls;
  let modelGroup, mainMesh, wireMesh, pointsObject, voxelObject, bboxHelper, gridHelper, axesHelper;
  let clipPlane;

  // DOM Elements
  const canvas = document.getElementById('canvas-3d');
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingProgressText = document.getElementById('loading-progress-text');

  // --- INITIALIZATION ---
  function init() {
    setupScene();
    setupLights();
    setupHelpers();
    setupEventListeners();
    setupTabs();
    
    // Load default model (supports both local server and static GitHub Pages)
    loadDefaultModel();
    
    // Fetch structural analysis & comparison info
    fetchAnalysisData();
    fetchComparisonData();

    // Start render loop
    requestAnimationFrame(renderLoop);
  }

  // --- SCENE SETUP (WHITE / LIGHT CAD THEME) ---
  function setupScene() {
    scene = new THREE.Scene();
    // Clean studio light gray
    scene.background = new THREE.Color(0xf1f5f9);
    scene.fog = new THREE.FogExp2(0xf1f5f9, 0.0004);

    const aspect = canvas.clientWidth / canvas.clientHeight;
    camera = new THREE.PerspectiveCamera(45, aspect, 1, 5000);
    camera.position.set(380, 260, 420);

    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.localClippingEnabled = true;

    // Orbit Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxDistance = 3000;
    controls.minDistance = 20;
    controls.target.set(0, 0, 0);

    // Model group
    modelGroup = new THREE.Group();
    scene.add(modelGroup);

    // Clipping Plane
    clipPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);

    // Resize observer
    window.addEventListener('resize', onWindowResize);
  }

  function setupLights() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    // Main Key light (Top-front)
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.3);
    keyLight.position.set(350, 550, 450);
    scene.add(keyLight);

    // Fill light (Side soft fill)
    const fillLight = new THREE.DirectionalLight(0xe2e8f0, 0.7);
    fillLight.position.set(-350, 200, -250);
    scene.add(fillLight);

    // Bottom soft bounce light
    const bounceLight = new THREE.DirectionalLight(0xcbd5e1, 0.4);
    bounceLight.position.set(0, -400, 200);
    scene.add(bounceLight);
  }

  function setupHelpers() {
    // Ground Grid (Subtle light gray lines)
    gridHelper = new THREE.GridHelper(1000, 50, 0x94a3b8, 0xe2e8f0);
    gridHelper.position.y = -150;
    scene.add(gridHelper);

    // Axes Helper (RGB = XYZ)
    axesHelper = new THREE.AxesHelper(140);
    scene.add(axesHelper);

    // Bounding Box Helper
    bboxHelper = new THREE.Box3Helper(new THREE.Box3(), 0x2563eb);
    bboxHelper.visible = false;
    scene.add(bboxHelper);
  }

  // --- MODEL LOADING (HYBRID LOCAL + GITHUB PAGES) ---
  function loadDefaultModel() {
    showLoading(true, 'Đang nạp mô hình grasp_quad_full_assembly_v7.STL...');
    const loader = new THREE.STLLoader();

    // Try relative static path first, fallback to API
    const urlPrimary = './models/grasp_quad_full_assembly_v7.STL';
    const urlFallback = '/model/default.stl';

    loader.load(
      urlPrimary,
      function (geometry) {
        onGeometryLoaded(geometry, 'grasp_quad_full_assembly_v7.STL');
      },
      function (xhr) {
        if (xhr.lengthComputable) {
          const pct = Math.round((xhr.loaded / xhr.total) * 100);
          showLoading(true, `Đang tải: ${pct}% (${(xhr.loaded / 1048576).toFixed(1)} MB)`);
        }
      },
      function () {
        // Fallback to server route
        loader.load(
          urlFallback,
          function (geom) {
            onGeometryLoaded(geom, 'grasp_quad_full_assembly_v7.STL');
          },
          null,
          function (err) {
            console.error('Error loading STL:', err);
            showLoading(false);
            showToast('Không thể nạp mô hình 3D mặc định!', 'error');
          }
        );
      }
    );
  }

  function onGeometryLoaded(geometry, filename) {
    state.modelName = filename;
    document.getElementById('active-filename').textContent = filename;

    // Center geometry to origin
    geometry.computeBoundingBox();
    const center = geometry.boundingBox.getCenter(new THREE.Vector3());
    geometry.translate(-center.x, -center.y, -center.z);

    // Vertex normals
    geometry.computeVertexNormals();

    state.originalGeometry = geometry.clone();
    state.currentGeometry = geometry;

    // Align ground grid under model
    geometry.computeBoundingBox();
    gridHelper.position.y = geometry.boundingBox.min.y - 5;

    // Build representations
    rebuildRepresentations();

    // Fit camera
    fitCameraToObject(modelGroup);

    // Update UI Stats
    updateUIModelStats(geometry);

    showLoading(false);
    showToast(`Đã nạp thành công mô hình: ${filename}`, 'success');
  }

  // --- BUILD 3D REPRESENTATIONS ---
  function rebuildRepresentations() {
    while (modelGroup.children.length > 0) {
      modelGroup.remove(modelGroup.children[0]);
    }

    if (!state.currentGeometry) return;
    const geom = state.currentGeometry;

    // 1. Polygon Mesh
    const meshMat = createMeshMaterial();
    mainMesh = new THREE.Mesh(geom, meshMat);
    mainMesh.name = 'mesh';
    modelGroup.add(mainMesh);

    // Wireframe overlay (if active)
    if (state.wireOverlay) {
      const wireMat = new THREE.MeshBasicMaterial({
        color: 0x0f172a,
        wireframe: true,
        transparent: true,
        opacity: 0.3
      });
      wireMesh = new THREE.Mesh(geom, wireMat);
      modelGroup.add(wireMesh);
    }

    // 2. Point Cloud (Dark saturated points for light mode)
    const pointColors = generateHeightColors(geom);
    const pointGeom = geom.clone();
    pointGeom.setAttribute('color', pointColors);
    const pointMat = new THREE.PointsMaterial({
      size: state.pointSize,
      vertexColors: true,
      transparent: true,
      opacity: 0.95
    });
    pointsObject = new THREE.Points(pointGeom, pointMat);
    pointsObject.name = 'pointcloud';
    pointsObject.visible = (state.activeRep === 'pointcloud');
    modelGroup.add(pointsObject);

    // 3. Voxel Grid
    loadOrBuildVoxels(state.voxelPitch);

    // Apply active visibility
    applyRepresentationVisibility();

    // BBox helper update
    geom.computeBoundingBox();
    bboxHelper.box.copy(geom.boundingBox);
  }

  function createMeshMaterial() {
    const clippingPlanes = state.sliceEnabled ? [clipPlane] : [];

    switch (state.matStyle) {
      case 'heatmap': {
        const colors = generateHeightColors(state.currentGeometry);
        const geom = state.currentGeometry;
        geom.setAttribute('color', colors);
        return new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: state.roughness,
          metalness: state.metalness * 0.3,
          flatShading: state.flatShading,
          clippingPlanes: clippingPlanes,
          side: THREE.DoubleSide
        });
      }
      case 'normals':
        return new THREE.MeshNormalMaterial({
          flatShading: state.flatShading,
          clippingPlanes: clippingPlanes,
          side: THREE.DoubleSide
        });
      case 'matte':
        return new THREE.MeshStandardMaterial({
          color: 0x64748b,
          roughness: 0.85,
          metalness: 0.1,
          flatShading: state.flatShading,
          clippingPlanes: clippingPlanes,
          side: THREE.DoubleSide
        });
      case 'metallic':
      default:
        // Dark Titanium CAD look (high contrast on white background)
        return new THREE.MeshStandardMaterial({
          color: 0x475569,
          roughness: state.roughness,
          metalness: state.metalness,
          flatShading: state.flatShading,
          clippingPlanes: clippingPlanes,
          side: THREE.DoubleSide
        });
    }
  }

  // Generate vertex colors based on Z height (Jet/Turbo gradient)
  function generateHeightColors(geometry) {
    const pos = geometry.attributes.position;
    const count = pos.count;
    const colors = new Float32Array(count * 3);

    geometry.computeBoundingBox();
    const minZ = geometry.boundingBox.min.z;
    const maxZ = geometry.boundingBox.max.z;
    const spanZ = (maxZ - minZ) || 1.0;

    for (let i = 0; i < count; i++) {
      const z = pos.getZ(i);
      const t = (z - minZ) / spanZ;

      // Color mapping: Blue -> Teal -> Green -> Amber -> Red
      const r = Math.min(Math.max(1.5 - Math.abs(t * 4.0 - 3.0), 0.0), 1.0);
      const g = Math.min(Math.max(1.5 - Math.abs(t * 4.0 - 2.0), 0.0), 1.0);
      const b = Math.min(Math.max(1.5 - Math.abs(t * 4.0 - 1.0), 0.0), 1.0);

      colors[i * 3 + 0] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }
    return new THREE.BufferAttribute(colors, 3);
  }

  // Load voxels from static data or server API
  function loadOrBuildVoxels(pitch) {
    const staticUrl = './data/voxels.json';
    const apiUrl = `/api/voxels?pitch=${pitch}`;

    fetch(staticUrl)
      .then(r => r.ok ? r.json() : fetch(apiUrl).then(res => res.json()))
      .then(data => {
        state.voxelData = data;
        const centers = data.centers;
        const count = centers.length;

        if (voxelObject) {
          modelGroup.remove(voxelObject);
        }

        const boxGeom = new THREE.BoxGeometry(pitch * 0.88, pitch * 0.88, pitch * 0.88);
        const boxMat = new THREE.MeshStandardMaterial({
          color: 0x2563eb,
          metalness: 0.3,
          roughness: 0.4,
          clippingPlanes: state.sliceEnabled ? [clipPlane] : []
        });

        voxelObject = new THREE.InstancedMesh(boxGeom, boxMat, count);
        voxelObject.name = 'voxel';
        const dummy = new THREE.Object3D();

        for (let i = 0; i < count; i++) {
          const c = centers[i];
          dummy.position.set(c[0], c[1], c[2]);
          dummy.updateMatrix();
          voxelObject.setMatrixAt(i, dummy.matrix);
        }
        voxelObject.instanceMatrix.needsUpdate = true;
        voxelObject.visible = (state.activeRep === 'voxel');
        modelGroup.add(voxelObject);

        const sub = document.getElementById('voxel-sub-text');
        if (sub) sub.textContent = `${count.toLocaleString()} khối (${pitch}mm)`;
      })
      .catch(err => {
        console.warn('Voxel fetch warning:', err);
      });
  }

  function applyRepresentationVisibility() {
    if (!mainMesh) return;

    if (state.activeRep === 'mesh') {
      mainMesh.visible = true;
      mainMesh.material.wireframe = false;
      if (pointsObject) pointsObject.visible = false;
      if (voxelObject) voxelObject.visible = false;
    } else if (state.activeRep === 'wireframe') {
      mainMesh.visible = true;
      mainMesh.material.wireframe = true;
      if (pointsObject) pointsObject.visible = false;
      if (voxelObject) voxelObject.visible = false;
    } else if (state.activeRep === 'pointcloud') {
      mainMesh.visible = false;
      if (pointsObject) pointsObject.visible = true;
      if (voxelObject) voxelObject.visible = false;
    } else if (state.activeRep === 'voxel') {
      mainMesh.visible = false;
      if (pointsObject) pointsObject.visible = false;
      if (voxelObject) voxelObject.visible = true;
    }
  }

  // --- GEOMETRIC TRANSFORMATIONS ---
  function updateTransformations() {
    if (!modelGroup) return;

    modelGroup.position.set(
      state.translation.x,
      state.translation.y,
      state.translation.z
    );

    modelGroup.rotation.set(
      THREE.MathUtils.degToRad(state.rotationDeg.x),
      THREE.MathUtils.degToRad(state.rotationDeg.y),
      THREE.MathUtils.degToRad(state.rotationDeg.z)
    );

    modelGroup.scale.set(state.scale, state.scale, state.scale);

    const statusText = `Vị trí: [${state.translation.x}, ${state.translation.y}, ${state.translation.z}] mm | Xoay: [${state.rotationDeg.x}°, ${state.rotationDeg.y}°, ${state.rotationDeg.z}°] | Tỉ lệ: ${state.scale.toFixed(2)}x`;
    const statusBox = document.getElementById('transform-status-text');
    if (statusBox) statusBox.textContent = statusText;

    if (bboxHelper && state.currentGeometry) {
      bboxHelper.box.setFromObject(modelGroup);
    }
  }

  function actionCenterModel() {
    state.translation = { x: 0, y: 0, z: 0 };
    syncSliderInputs();
    updateTransformations();
    showToast('Đã căn tâm mô hình về gốc tọa độ [0, 0, 0]', 'success');
  }

  function actionNormalizeModel() {
    if (!state.currentGeometry) return;
    state.currentGeometry.computeBoundingBox();
    const extents = new THREE.Vector3();
    state.currentGeometry.boundingBox.getSize(extents);
    const maxDim = Math.max(extents.x, extents.y, extents.z);
    if (maxDim > 0) {
      state.scale = 200.0 / maxDim;
      syncSliderInputs();
      updateTransformations();
      showToast(`Đã chuẩn hóa kích thước bao về tỉ lệ chuẩn (Scale: ${state.scale.toFixed(3)}x)`, 'success');
    }
  }

  function actionAddNoise() {
    if (!state.currentGeometry) return;
    const pos = state.currentGeometry.attributes.position;
    const std = 1.2;
    for (let i = 0; i < pos.count; i++) {
      const nx = (Math.random() - 0.5) * 2 * std;
      const ny = (Math.random() - 0.5) * 2 * std;
      const nz = (Math.random() - 0.5) * 2 * std;
      pos.setXYZ(i, pos.getX(i) + nx, pos.getY(i) + ny, pos.getZ(i) + nz);
    }
    pos.needsUpdate = true;
    state.currentGeometry.computeVertexNormals();
    rebuildRepresentations();
    showToast('Đã thêm nhiễu ngẫu nhiên lên tọa độ đỉnh (±1.2 mm)', 'success');
  }

  function actionResetTransforms() {
    state.translation = { x: 0, y: 0, z: 0 };
    state.rotationDeg = { x: 0, y: 0, z: 0 };
    state.scale = 1.0;
    syncSliderInputs();

    if (state.originalGeometry) {
      state.currentGeometry = state.originalGeometry.clone();
      rebuildRepresentations();
    }
    updateTransformations();
    showToast('Đã khôi phục hoàn toàn trạng thái mô hình gốc', 'success');
  }

  function syncSliderInputs() {
    document.getElementById('tr-x').value = state.translation.x;
    document.getElementById('num-tr-x').value = state.translation.x;
    document.getElementById('tr-y').value = state.translation.y;
    document.getElementById('num-tr-y').value = state.translation.y;
    document.getElementById('tr-z').value = state.translation.z;
    document.getElementById('num-tr-z').value = state.translation.z;

    document.getElementById('rot-x').value = state.rotationDeg.x;
    document.getElementById('num-rot-x').value = state.rotationDeg.x;
    document.getElementById('rot-y').value = state.rotationDeg.y;
    document.getElementById('num-rot-y').value = state.rotationDeg.y;
    document.getElementById('rot-z').value = state.rotationDeg.z;
    document.getElementById('num-rot-z').value = state.rotationDeg.z;

    document.getElementById('scale-all').value = state.scale;
    document.getElementById('num-scale-all').value = state.scale;
  }

  // --- CROSS-SECTION SLICER ---
  function updateClippingPlane() {
    if (!state.sliceEnabled) {
      clipPlane.constant = 10000;
      if (mainMesh) mainMesh.material.clippingPlanes = [];
      if (voxelObject) voxelObject.material.clippingPlanes = [];
      return;
    }

    if (state.sliceAxis === 'x') {
      clipPlane.normal.set(-1, 0, 0);
    } else if (state.sliceAxis === 'y') {
      clipPlane.normal.set(0, -1, 0);
    } else if (state.sliceAxis === 'z') {
      clipPlane.normal.set(0, 0, -1);
    }
    clipPlane.constant = state.slicePos;

    if (mainMesh) mainMesh.material.clippingPlanes = [clipPlane];
    if (voxelObject) voxelObject.material.clippingPlanes = [clipPlane];
  }

  // --- CAMERA CONTROLS ---
  function setCameraView(preset) {
    if (!modelGroup) return;
    const dist = 450;
    switch (preset) {
      case 'iso':
        camera.position.set(dist * 0.7, dist * 0.7, dist * 0.7);
        break;
      case 'front':
        camera.position.set(0, 0, dist);
        break;
      case 'top':
        camera.position.set(0, dist, 0);
        break;
      case 'right':
        camera.position.set(dist, 0, 0);
        break;
    }
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function fitCameraToObject(object) {
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 1.5;

    camera.position.set(center.x + cameraZ * 0.7, center.y + cameraZ * 0.6, center.z + cameraZ * 0.8);
    camera.near = maxDim / 100;
    camera.far = maxDim * 100;
    camera.updateProjectionMatrix();

    controls.target.copy(center);
    controls.maxDistance = maxDim * 10;
    controls.update();
  }

  // --- DATA FETCHING (HYBRID STATIC & LOCAL API) ---
  function fetchAnalysisData() {
    const staticUrl = './data/structure.json';
    const apiUrl = '/api/structure';

    fetch(staticUrl)
      .then(r => r.ok ? r.json() : fetch(apiUrl).then(res => res.json()))
      .then(data => {
        state.analysisData = data;
        populateStructureUI(data);
      })
      .catch(err => console.warn('Could not fetch structure data:', err));
  }

  function fetchComparisonData() {
    const staticUrl = './data/comparison.json';
    const apiUrl = '/api/comparison';

    fetch(staticUrl)
      .then(r => r.ok ? r.json() : fetch(apiUrl).then(res => res.json()))
      .then(data => {
        state.comparisonData = data;
        populateComparisonUI(data.comparison);
      })
      .catch(err => console.warn('Could not fetch comparison data:', err));
  }

  function updateUIModelStats(geometry) {
    const pos = geometry.attributes.position;
    const vertexCount = pos.count;
    const faceCount = vertexCount / 3;

    geometry.computeBoundingBox();
    const dims = new THREE.Vector3();
    geometry.boundingBox.getSize(dims);

    document.getElementById('hdr-faces').textContent = Math.round(faceCount).toLocaleString();
    document.getElementById('hdr-vertices').textContent = vertexCount.toLocaleString();
    document.getElementById('hdr-dims').textContent = `${dims.x.toFixed(0)} × ${dims.y.toFixed(0)} × ${dims.z.toFixed(0)} mm`;

    document.getElementById('m-faces').textContent = Math.round(faceCount).toLocaleString();
    document.getElementById('m-vertices').textContent = vertexCount.toLocaleString();
    document.getElementById('tb-dim-x').textContent = `${dims.x.toFixed(2)} mm`;
    document.getElementById('tb-dim-y').textContent = `${dims.y.toFixed(2)} mm`;
    document.getElementById('tb-dim-z').textContent = `${dims.z.toFixed(2)} mm`;
    document.getElementById('tb-dim-diag').textContent = `${dims.length().toFixed(2)} mm`;

    drawElevationHistogram(geometry);
  }

  function populateStructureUI(data) {
    if (!data) return;
    const mc = data.mesh_counts;
    const bb = data.bounding_box;
    const gp = data.geometric_properties;

    if (mc) {
      document.getElementById('hdr-faces').textContent = mc.faces.toLocaleString();
      document.getElementById('hdr-vertices').textContent = mc.vertices.toLocaleString();
      document.getElementById('m-faces').textContent = mc.faces.toLocaleString();
      document.getElementById('m-vertices').textContent = mc.vertices.toLocaleString();
    }
    if (gp) {
      document.getElementById('m-area').textContent = gp.surface_area_cm2.toLocaleString();
      document.getElementById('m-volume').textContent = (gp.convex_hull_volume_cm3 / 1000.0).toFixed(2);
      document.getElementById('tb-watertight').textContent = gp.is_watertight ? 'Khép kín' : 'Cụm lắp ghép đa thân';
    }
    if (bb) {
      document.getElementById('tb-centroid').textContent = `[${bb.centroid.join(', ')}]`;
    }
  }

  function populateComparisonUI(comparisons) {
    const container = document.getElementById('comparison-cards-container');
    if (!container || !comparisons) return;

    container.innerHTML = comparisons.map(item => `
      <div class="comp-card">
        <div class="comp-card-hdr">
          <span class="comp-title">${item.representation}</span>
          <span class="comp-pill">${item.elements_count.toLocaleString()} ${item.elements_name}</span>
        </div>
        <div class="comp-stats-row">
          <span class="comp-stat">RAM: <b>${item.memory_mb} MB</b></span>
          <span class="comp-stat">Cấu trúc: <b>${item.type.split(' ')[0]}</b></span>
        </div>
        <div class="comp-hri-use">
          <strong>Ứng dụng HRI:</strong> ${item.hri_robotics_use_case}
        </div>
      </div>
    `).join('');
  }

  // Draw 14-bin elevation histogram on clean canvas
  function drawElevationHistogram(geometry) {
    const canvas = document.getElementById('hist-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const pos = geometry.attributes.position;
    const count = pos.count;
    geometry.computeBoundingBox();
    const minZ = geometry.boundingBox.min.z;
    const maxZ = geometry.boundingBox.max.z;
    const spanZ = (maxZ - minZ) || 1.0;

    const bins = new Array(14).fill(0);
    for (let i = 0; i < count; i++) {
      const z = pos.getZ(i);
      const idx = Math.min(Math.floor(((z - minZ) / spanZ) * bins.length), bins.length - 1);
      bins[idx]++;
    }

    const maxBin = Math.max(...bins) || 1;
    const barWidth = (width - 20) / bins.length;

    for (let i = 0; i < bins.length; i++) {
      const barHeight = (bins[i] / maxBin) * (height - 22);
      const x = 10 + i * barWidth;
      const y = height - barHeight - 12;

      // Clean engineering blue gradient
      ctx.fillStyle = '#2563eb';
      ctx.fillRect(x + 1, y, barWidth - 2, barHeight);
    }

    // Baseline
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(10, height - 12);
    ctx.lineTo(width - 10, height - 12);
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#64748b';
    ctx.font = '9px monospace';
    ctx.fillText(`${minZ.toFixed(0)}mm`, 10, height - 2);
    ctx.textAlign = 'right';
    ctx.fillText(`${maxZ.toFixed(0)}mm`, width - 10, height - 2);
    ctx.textAlign = 'left';
  }

  // --- EXPORT FUNCTIONS (100% CLIENT-SIDE & SERVER) ---
  function exportTransformedSTL() {
    if (!mainMesh) return;
    const exporter = new THREE.STLExporter();
    const result = exporter.parse(mainMesh, { binary: true });
    const blob = new Blob([result], { type: 'application/octet-stream' });
    downloadBlob(blob, 'grasp_quad_transformed.stl');
    showToast('Đã xuất file STL (Binary) thành công!', 'success');
  }

  function exportTransformedOBJ() {
    if (!mainMesh) return;
    const exporter = new THREE.OBJExporter();
    const result = exporter.parse(mainMesh);
    const blob = new Blob([result], { type: 'text/plain' });
    downloadBlob(blob, 'grasp_quad_transformed.obj');
    showToast('Đã xuất file OBJ (Wavefront) thành công!', 'success');
  }

  function exportTransformedPLY() {
    if (!state.currentGeometry) return;
    const targetObj = pointsObject && pointsObject.visible ? pointsObject : mainMesh;
    const exporter = new THREE.PLYExporter();
    exporter.parse(targetObj, function (result) {
      const blob = new Blob([result], { type: 'application/octet-stream' });
      downloadBlob(blob, 'grasp_quad_pointcloud.ply');
      showToast('Đã xuất file PLY thành công!', 'success');
    }, { binary: true });
  }

  function downloadAnalysisJSON() {
    if (state.analysisData) {
      const blob = new Blob([JSON.stringify(state.analysisData, null, 2)], { type: 'application/json' });
      downloadBlob(blob, 'bao_cao_cau_truc_3d.json');
      showToast('Đã tải báo cáo cấu trúc JSON!', 'success');
    } else {
      window.open('./data/structure.json', '_blank');
    }
  }

  function saveAllToDisk() {
    showLoading(true, 'Đang lưu toàn bộ 6 định dạng xuất...');
    fetch('/api/save_exports', { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        showLoading(false);
        showToast(data.message || 'Đã lưu kết quả thành công vào results/3D_model_analysis/', 'success');
      })
      .catch(() => {
        // Fallback for static GitHub Pages: trigger download of STL and JSON directly
        exportTransformedSTL();
        downloadAnalysisJSON();
        showLoading(false);
      });
  }

  function captureScreenshot() {
    renderer.render(scene, camera);
    const dataUrl = renderer.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `screenshot_${state.modelName}_${Date.now()}.png`;
    a.click();
    showToast('Đã chụp ảnh màn hình 3D độ nét cao!', 'success');
  }

  function downloadBlob(blob, filename) {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    // Representation buttons
    document.querySelectorAll('.rep-btn').forEach(btn => {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.rep-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        state.activeRep = this.dataset.rep;
        applyRepresentationVisibility();
      });
    });

    // Material Style
    const matSelect = document.getElementById('mat-style-select');
    matSelect.addEventListener('change', function () {
      state.matStyle = this.value;
      if (mainMesh) {
        mainMesh.material = createMeshMaterial();
      }
    });

    // Wireframe overlay
    document.getElementById('chk-wire-overlay').addEventListener('change', function () {
      state.wireOverlay = this.checked;
      rebuildRepresentations();
    });

    // Flat shading
    document.getElementById('chk-flat-shading').addEventListener('change', function () {
      state.flatShading = this.checked;
      if (mainMesh) {
        mainMesh.material = createMeshMaterial();
      }
    });

    // Metalness & Roughness
    document.getElementById('rng-metalness').addEventListener('input', function () {
      state.metalness = parseFloat(this.value);
      document.getElementById('val-metalness').textContent = state.metalness.toFixed(2);
      if (mainMesh && mainMesh.material.metalness !== undefined) {
        mainMesh.material.metalness = state.metalness;
      }
    });

    document.getElementById('rng-roughness').addEventListener('input', function () {
      state.roughness = parseFloat(this.value);
      document.getElementById('val-roughness').textContent = state.roughness.toFixed(2);
      if (mainMesh && mainMesh.material.roughness !== undefined) {
        mainMesh.material.roughness = state.roughness;
      }
    });

    // Point size
    document.getElementById('rng-point-size').addEventListener('input', function () {
      state.pointSize = parseFloat(this.value);
      document.getElementById('val-point-size').textContent = `${state.pointSize.toFixed(1)} px`;
      if (pointsObject) {
        pointsObject.material.size = state.pointSize;
      }
    });

    // Voxel pitch
    document.getElementById('rng-voxel-pitch').addEventListener('change', function () {
      state.voxelPitch = parseFloat(this.value);
      document.getElementById('val-voxel-pitch').textContent = `${state.voxelPitch} mm`;
      loadOrBuildVoxels(state.voxelPitch);
    });

    // Slicer
    document.getElementById('chk-slice-enable').addEventListener('change', function () {
      state.sliceEnabled = this.checked;
      updateClippingPlane();
    });

    document.querySelectorAll('.btn-axis').forEach(btn => {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.btn-axis').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        state.sliceAxis = this.dataset.axis;
        updateClippingPlane();
      });
    });

    document.getElementById('rng-slice-pos').addEventListener('input', function () {
      state.slicePos = parseFloat(this.value);
      document.getElementById('val-slice-pos').textContent = `${state.slicePos} mm`;
      updateClippingPlane();
    });

    // Camera toolbar
    document.getElementById('cam-iso').addEventListener('click', () => setCameraView('iso'));
    document.getElementById('cam-front').addEventListener('click', () => setCameraView('front'));
    document.getElementById('cam-top').addEventListener('click', () => setCameraView('top'));
    document.getElementById('cam-right').addEventListener('click', () => setCameraView('right'));

    // Guide toggles
    const btnGrid = document.getElementById('btn-toggle-grid');
    btnGrid.addEventListener('click', () => {
      state.showGrid = !state.showGrid;
      gridHelper.visible = state.showGrid;
      btnGrid.classList.toggle('active', state.showGrid);
    });

    const btnAxes = document.getElementById('btn-toggle-axes');
    btnAxes.addEventListener('click', () => {
      state.showAxes = !state.showAxes;
      axesHelper.visible = state.showAxes;
      btnAxes.classList.toggle('active', state.showAxes);
    });

    const btnBBox = document.getElementById('btn-toggle-bbox');
    btnBBox.addEventListener('click', () => {
      state.showBBox = !state.showBBox;
      bboxHelper.visible = state.showBBox;
      btnBBox.classList.toggle('active', state.showBBox);
    });

    const btnAutoRot = document.getElementById('btn-auto-rotate');
    btnAutoRot.addEventListener('click', () => {
      state.autoRotate = !state.autoRotate;
      btnAutoRot.classList.toggle('active', state.autoRotate);
    });

    // Transform Sliders
    bindSliderAndNumber('tr-x', 'num-tr-x', val => { state.translation.x = val; updateTransformations(); });
    bindSliderAndNumber('tr-y', 'num-tr-y', val => { state.translation.y = val; updateTransformations(); });
    bindSliderAndNumber('tr-z', 'num-tr-z', val => { state.translation.z = val; updateTransformations(); });

    bindSliderAndNumber('rot-x', 'num-rot-x', val => { state.rotationDeg.x = val; updateTransformations(); });
    bindSliderAndNumber('rot-y', 'num-rot-y', val => { state.rotationDeg.y = val; updateTransformations(); });
    bindSliderAndNumber('rot-z', 'num-rot-z', val => { state.rotationDeg.z = val; updateTransformations(); });

    bindSliderAndNumber('scale-all', 'num-scale-all', val => { state.scale = val; updateTransformations(); });

    // Transform Buttons
    document.getElementById('btn-action-center').addEventListener('click', actionCenterModel);
    document.getElementById('btn-action-normalize').addEventListener('click', actionNormalizeModel);
    document.getElementById('btn-action-noise').addEventListener('click', actionAddNoise);
    document.getElementById('btn-action-reset-transform').addEventListener('click', actionResetTransforms);

    // Export Buttons
    document.getElementById('btn-export-stl').addEventListener('click', exportTransformedSTL);
    document.getElementById('btn-export-obj').addEventListener('click', exportTransformedOBJ);
    document.getElementById('btn-export-ply').addEventListener('click', exportTransformedPLY);
    document.getElementById('btn-export-json').addEventListener('click', downloadAnalysisJSON);
    document.getElementById('btn-save-server-disk').addEventListener('click', saveAllToDisk);
    document.getElementById('btn-save-all').addEventListener('click', saveAllToDisk);
    document.getElementById('btn-snapshot').addEventListener('click', captureScreenshot);
    document.getElementById('btn-reset-view').addEventListener('click', () => setCameraView('iso'));

    // Open file
    const fileInput = document.getElementById('file-input');
    fileInput.addEventListener('change', function (e) {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      const filename = file.name;
      const ext = filename.split('.').pop().toLowerCase();

      showLoading(true, `Đang đọc file ${filename}...`);

      reader.onload = function (event) {
        const buffer = event.target.result;
        try {
          if (ext === 'stl') {
            const loader = new THREE.STLLoader();
            const geometry = loader.parse(buffer);
            onGeometryLoaded(geometry, filename);
          } else {
            showToast('Vui lòng chọn tệp .stl!', 'error');
            showLoading(false);
          }
        } catch (err) {
          console.error(err);
          showToast('Lỗi khi phân tích tệp 3D: ' + err, 'error');
          showLoading(false);
        }
      };
      reader.readAsArrayBuffer(file);
    });

    // Comparison shortcut
    document.getElementById('btn-toggle-split-view').addEventListener('click', function () {
      switchTab('tab-compare');
    });
  }

  function bindSliderAndNumber(sliderId, numId, callback) {
    const slider = document.getElementById(sliderId);
    const num = document.getElementById(numId);
    if (!slider || !num) return;

    slider.addEventListener('input', function () {
      num.value = this.value;
      callback(parseFloat(this.value));
    });

    num.addEventListener('input', function () {
      slider.value = this.value;
      callback(parseFloat(this.value));
    });
  }

  // --- TABS ---
  function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', function () {
        switchTab(this.dataset.tab);
      });
    });
  }

  function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === tabId));
  }

  // --- RESIZE ---
  function onWindowResize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  // --- RENDER LOOP ---
  function renderLoop(now) {
    requestAnimationFrame(renderLoop);

    state.frames++;
    if (now > state.lastTime + 1000) {
      state.fps = Math.round((state.frames * 1000) / (now - state.lastTime));
      document.getElementById('hdr-fps').textContent = state.fps;
      state.frames = 0;
      state.lastTime = now;
    }

    if (state.autoRotate && modelGroup) {
      modelGroup.rotation.y += 0.006;
    }

    controls.update();
    renderer.render(scene, camera);
  }

  // --- HELPERS ---
  function showLoading(show, message) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
    if (message) loadingProgressText.textContent = message;
  }

  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      toast.style.transition = 'all 0.25s ease';
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  window.addEventListener('DOMContentLoaded', init);
})();
