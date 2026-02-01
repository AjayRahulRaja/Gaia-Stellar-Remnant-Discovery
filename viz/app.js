import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer, controls;
let starPoints, candidatesPoints;
const mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 20; // Maximum hit area for reliable clicking
let allStarsData = [];

init();

async function init() {
    // 1. Setup Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020205);
    // Removed fog entirely to prevent black screen issues

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
    camera.position.set(200, 200, 200);

    renderer = new THREE.WebGLRenderer({
        antialias: true,
        preserveDrawingBuffer: true // Required for screenshots
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // --- GLOBAL VIZ CONTROLS ---
    window.vizZoom = (factor) => {
        camera.position.multiplyScalar(1 / factor);
        controls.update();
    };

    window.vizCenter = () => {
        controls.reset();
        camera.position.set(200, 200, 200);
    };

    window.vizScreenshot = () => {
        const link = document.createElement('a');
        link.download = `gaia_discovery_${Date.now()}.png`;
        link.href = renderer.domElement.toDataURL('image/png');
        link.click();
    };
    // -------------------------

    // 4. Add Cosmic Environment
    addCosmicEnvironment();

    // 2. Load Data
    try {
        const response = await fetch('../results/viz_data.json');
        allStarsData = await response.json();

        plotStars(allStarsData);
        updateStats(allStarsData);

        document.getElementById('loading-overlay').style.opacity = '0';
        setTimeout(() => {
            document.getElementById('loading-overlay').style.display = 'none';
        }, 500);

    } catch (error) {
        console.error("Error loading viz data:", error);
        document.querySelector('#loading-overlay p').textContent = "Error: Please run pipeline.py first.";
    }

    // 5. Add Selection Marker
    // 5. Add Selection Marker (Cyan Crosshair)
    // 5. Add Selection Marker (Cyan Crosshair)
    const selectionGeometry = new THREE.RingGeometry(12, 16, 32); // Larger marker
    const selectionMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ffff,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 1.0
    });
    window.selectionMarker = new THREE.Mesh(selectionGeometry, selectionMaterial);
    window.selectionMarker.visible = false;
    scene.add(window.selectionMarker);

    animate();
}

function addCosmicEnvironment() {
    // 1. Distant Starfield (50k points)
    const starGeometry = new THREE.BufferGeometry();
    const starPositions = [];
    const starColors = [];

    for (let i = 0; i < 50000; i++) {
        const r = 4000; // Far out
        const theta = 2 * Math.PI * Math.random();
        const phi = Math.acos(2 * Math.random() - 1);

        starPositions.push(
            r * Math.sin(phi) * Math.cos(theta),
            r * Math.sin(phi) * Math.sin(theta),
            r * Math.cos(phi)
        );

        const rand = Math.random();
        if (rand > 0.8) starColors.push(0.7, 0.7, 1.0); // Blueish
        else if (rand > 0.6) starColors.push(1.0, 1.0, 0.7); // Yellowish
        else starColors.push(1.0, 1.0, 1.0); // White
    }

    starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3));
    starGeometry.setAttribute('color', new THREE.Float32BufferAttribute(starColors, 3));

    const starMaterial = new THREE.PointsMaterial({
        size: 3.0, // Increased size for visibility
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        sizeAttenuation: false,
        fog: false
    });

    const worldStars = new THREE.Points(starGeometry, starMaterial);
    scene.add(worldStars);

    // 2. Nebula Glows (Artistic)
    const nebulaColors = [0x0a0a25, 0x1a0a35, 0x0a1a35];
    nebulaColors.forEach((color, i) => {
        const nebulaGeometry = new THREE.SphereGeometry(2500 + i * 500, 32, 32);
        const nebulaMaterial = new THREE.MeshBasicMaterial({
            color: color,
            side: THREE.BackSide,
            transparent: true,
            opacity: 0.15,
            fog: false
        });
        const nebula = new THREE.Mesh(nebulaGeometry, nebulaMaterial);
        scene.add(nebula);
    });
}

function onMouseMove(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(candidatesPoints);

    if (intersects.length > 0) {
        document.body.style.cursor = 'pointer';
    } else {
        document.body.style.cursor = 'default';
    }
}

function plotStars(stars) {
    const geometries = {
        background: new THREE.BufferGeometry(),
        candidates: new THREE.BufferGeometry()
    };

    const positions = { background: [], candidates: [] };
    const colors = { background: [], candidates: [] };
    const sizes = { background: [], candidates: [] };

    stars.forEach(star => {
        // Convert RA, Dec, Parallax to Cartesian
        // d = 1000 / parallax (pc)
        const d = star.parallax > 0 ? 1000.0 / star.parallax : 1000.0;
        const raRad = star.ra * Math.PI / 180;
        const decRad = star.dec * Math.PI / 180;

        const x = d * Math.cos(decRad) * Math.cos(raRad);
        const y = d * Math.cos(decRad) * Math.sin(raRad);
        const z = d * Math.sin(decRad);

        if (star.is_candidate) {
            positions.candidates.push(x, y, z);

            // Color by anomaly type
            let color = new THREE.Color(0xffffff);
            if (star.type === 'astrometric_binary') color.setHex(0xff3e3e); // Red
            else if (star.type === 'kinematic_outlier') color.setHex(0x3effde); // Teal
            else if (star.type === 'quality_issue') color.setHex(0xffbb3e); // Orange

            colors.candidates.push(color.r, color.g, color.b);
            sizes.candidates.push(2.5);
        } else {
            positions.background.push(x, y, z);
            colors.background.push(0.4, 0.4, 0.6); // Muted blue
            sizes.background.push(1.0);
        }
    });

    // Create background stars
    geometries.background.setAttribute('position', new THREE.Float32BufferAttribute(positions.background, 3));
    geometries.background.setAttribute('color', new THREE.Float32BufferAttribute(colors.background, 3));

    const bgMaterial = new THREE.PointsMaterial({
        size: 1.5,
        vertexColors: true,
        transparent: true,
        opacity: 0.5,
        sizeAttenuation: true
    });
    starPoints = new THREE.Points(geometries.background, bgMaterial);
    scene.add(starPoints);

    // Create highlighted candidates
    geometries.candidates.setAttribute('position', new THREE.Float32BufferAttribute(positions.candidates, 3));
    geometries.candidates.setAttribute('color', new THREE.Float32BufferAttribute(colors.candidates, 3));

    const candidateMaterial = new THREE.PointsMaterial({
        size: 4.0,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9
    });
    candidatesPoints = new THREE.Points(geometries.candidates, candidateMaterial);
    scene.add(candidatesPoints);

    // Add coordinate helpers
    const axes = new THREE.AxesHelper(100);
    scene.add(axes);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function onStarClick(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(candidatesPoints);

    if (intersects.length > 0) {
        const index = intersects[0].index;
        // Find the corresponding star in our data
        const candidates = allStarsData.filter(s => s.is_candidate);
        const selected = candidates[index];

        console.log("Selected Star:", selected); // Debugging

        // Update highlight marker
        window.selectionMarker.position.copy(intersects[0].point);
        window.selectionMarker.visible = true;

        showStarInfo(selected);
    } else {
        document.getElementById('info-panel').style.display = 'none';
        window.selectionMarker.visible = false;
    }
}

function showStarInfo(star) {
    const panel = document.getElementById('info-panel');
    panel.style.display = 'block';

    // Calculate XYZ for display
    const d = star.parallax > 0 ? 1000.0 / star.parallax : 0;
    const raRad = star.ra * Math.PI / 180;
    const decRad = star.dec * Math.PI / 180;
    const x = d * Math.cos(decRad) * Math.cos(raRad);
    const y = d * Math.cos(decRad) * Math.sin(raRad);
    const z = d * Math.sin(decRad);

    document.getElementById('star-id').textContent = `ID: ${star.id}`;
    document.getElementById('star-score').textContent = star.score.toFixed(4);
    document.getElementById('star-dist').textContent = d.toFixed(1);
    document.getElementById('star-ra').textContent = star.ra.toFixed(2);
    document.getElementById('star-dec').textContent = star.dec.toFixed(2);
    document.getElementById('star-x').textContent = x.toFixed(1);
    document.getElementById('star-y').textContent = y.toFixed(1);
    document.getElementById('star-z').textContent = z.toFixed(1);

    const typeLabel = document.getElementById('star-type');
    typeLabel.textContent = star.type.toUpperCase().replace('_', ' ');
    typeLabel.className = 'discovery-tag ' + (star.type === 'astrometric_binary' ? 'tag-binary' :
        star.type === 'kinematic_outlier' ? 'tag-kinematic' : 'tag-normal');
}

function updateStats(stars) {
    const total = stars.length;
    const candidates = stars.filter(s => s.is_candidate).length;
    document.getElementById('stats-content').innerHTML = `
        <strong>Universe View</strong><br>
        Showing ${total.toLocaleString()} stars<br>
        <span style="color: #ff3e3e;">${candidates} Discovery Candidates</span>
    `;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    if (window.selectionMarker && window.selectionMarker.visible) {
        window.selectionMarker.lookAt(camera.position); // Always face camera

        const time = performance.now() * 0.003;
        const s = 1.0 + Math.sin(time) * 0.1;
        window.selectionMarker.scale.set(s, s, s);
    }

    renderer.render(scene, camera);
}
