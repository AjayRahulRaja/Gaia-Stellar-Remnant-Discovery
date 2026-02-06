import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer, controls;
let starPoints, candidatesPoints, worldStars, nebulaMeshes = [];
const mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 30;
let allStarsData = [];
let selectedStarIndex = null; // Track selected star for color change
let originalColors = []; // Store original colors

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
        const dataUrl = renderer.domElement.toDataURL('image/png');

        // Create toast notification
        const toast = document.createElement('div');
        toast.id = 'screenshot-toast';
        toast.innerHTML = `
            <div style="background: rgba(10,10,25,0.95); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2); padding: 20px; border-radius: 12px; position: fixed; bottom: 80px; right: 20px; z-index: 9999; min-width: 250px;">
                <div style="font-weight: bold; margin-bottom: 10px; color: #4a9eff;">📸 Screenshot Captured</div>
                <button onclick="window.saveScreenshot('png')" style="width: 100%; margin: 5px 0; padding: 10px; background: #4a9eff; border: none; color: white; border-radius: 6px; cursor: pointer; font-size: 14px;">💾 Save as PNG</button>
                <button onclick="window.saveScreenshot('jpg')" style="width: 100%; margin: 5px 0; padding: 10px; background: rgba(60,60,90,0.8); border: 1px solid rgba(255,255,255,0.2); color: white; border-radius: 6px; cursor: pointer; font-size: 14px;">💾 Save as JPG</button>
                <button onclick="window.shareScreenshot()" style="width: 100%; margin: 5px 0; padding: 10px; background: rgba(60,60,90,0.8); border: 1px solid rgba(255,255,255,0.2); color: white; border-radius: 6px; cursor: pointer; font-size: 14px;">🔗 Share</button>
            </div>
        `;
        document.body.appendChild(toast);

        // Store data URL globally
        window.currentScreenshot = dataUrl;

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) toast.remove();
        }, 5000);
    };

    window.saveScreenshot = (format) => {
        const link = document.createElement('a');
        link.download = `gaia_discovery_${Date.now()}.${format}`;

        if (format === 'jpg') {
            // Convert PNG to JPG
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                link.href = canvas.toDataURL('image/jpeg', 0.95);
                link.click();
            };
            img.src = window.currentScreenshot;
        } else {
            link.href = window.currentScreenshot;
            link.click();
        }

        // Remove toast
        const toast = document.getElementById('screenshot-toast');
        if (toast) toast.remove();
    };

    window.shareScreenshot = async () => {
        try {
            const blob = await (await fetch(window.currentScreenshot)).blob();
            const file = new File([blob], `gaia_discovery_${Date.now()}.png`, { type: 'image/png' });

            if (navigator.share && navigator.canShare({ files: [file] })) {
                await navigator.share({
                    files: [file],
                    title: 'Gaia Stellar Remnant Discovery',
                    text: 'Check out this stellar discovery visualization!'
                });
            } else {
                // Fallback: copy to clipboard
                const item = new ClipboardItem({ 'image/png': blob });
                await navigator.clipboard.write([item]);
                alert('Screenshot copied to clipboard!');
            }
        } catch (err) {
            console.error('Share failed:', err);
            alert('Sharing not supported. Use Save instead.');
        }

        // Remove toast
        const toast = document.getElementById('screenshot-toast');
        if (toast) toast.remove();
    };
    // -------------------------

    // 4. Add Cosmic Environment
    addCosmicEnvironment();

    // 2. Load Data
    try {
        const response = await fetch('./viz_data.json'); // Fetch from local viz folder
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        allStarsData = await response.json();

        plotStars(allStarsData);
        updateStats(allStarsData);

        document.getElementById('loading-overlay').style.opacity = '0';
        setTimeout(() => {
            document.getElementById('loading-overlay').style.display = 'none';
        }, 500);

    } catch (error) {
        console.error("Error loading viz data:", error);
        document.querySelector('#loading-overlay p').textContent = "Error loading data. See console.";
        alert("Failed to load visualization data. Please check if viz_data.json is present.");
    }

    // 5. Selection marker removed - using color change instead

    // 6. Add Event Listeners
    window.addEventListener('click', onStarClick);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('resize', onWindowResize);

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
        size: 1.5, // Reduced from 3.0 to minimize flickering
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        sizeAttenuation: false,
        fog: false
    });



    worldStars = new THREE.Points(starGeometry, starMaterial);
    scene.add(worldStars);

    // 2. Add "Sun" Label at Origin (Static Position)
    const sunDiv = document.createElement('div');
    sunDiv.textContent = '☉ Sun (Origin)';
    sunDiv.style.position = 'absolute';
    sunDiv.style.left = '50%';
    sunDiv.style.top = '50%';
    sunDiv.style.transform = 'translate(-50%, -80px)';
    sunDiv.style.color = '#ffd700;
    sunDiv.style.fontWeight = 'bold';
    sunDiv.style.fontSize = '14px';
    sunDiv.style.pointerEvents = 'none';
    sunDiv.style.textShadow = '0 0 10px black, 0 0 5px black';
    sunDiv.style.zIndex = '100';
    sunDiv.style.background = 'rgba(0,0,0,0.6)';
    sunDiv.style.padding = '5px 10px';
    sunDiv.style.borderRadius = '4px';
    document.body.appendChild(sunDiv);
    
    const sunGeo = new THREE.SphereGeometry(3, 16, 16);
    const sunMat = new THREE.MeshBasicMaterial({ color: 0xffff00 });
    const sunMesh = new THREE.Mesh(sunGeo, sunMat);
    scene.add(sunMesh);



















    // 3. Nebula Glows (Artistic)
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
        nebulaMeshes.push(nebula); scene.add(nebula);
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

    // Store original colors for selection/deselection
    originalColors = [...colors.candidates];

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

    if (worldStars) {
        worldStars.position.copy(camera.position); // Lock stars to camera

        // Lock nebulas to camera to prevent black screen when inside them
        nebulaMeshes.forEach(nebula => {
            nebula.position.copy(camera.position);
        });
    }



    renderer.render(scene, camera);
}
function onStarClick(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(candidatesPoints);

    if (intersects.length > 0) {
        const index = intersects[0].index;
        const candidates = allStarsData.filter(s => s.is_candidate);
        const selected = candidates[index];

        console.log("Selected Star:", selected);

        // Restore previous selection's color
        if (selectedStarIndex !== null) {
            const colors = candidatesPoints.geometry.attributes.color.array;
            colors[selectedStarIndex * 3] = originalColors[selectedStarIndex * 3];
            colors[selectedStarIndex * 3 + 1] = originalColors[selectedStarIndex * 3 + 1];
            colors[selectedStarIndex * 3 + 2] = originalColors[selectedStarIndex * 3 + 2];
        }

        // Change selected star to bright yellow
        const colors = candidatesPoints.geometry.attributes.color.array;
        colors[index * 3] = 1.0;     // R
        colors[index * 3 + 1] = 1.0; // G
        colors[index * 3 + 2] = 0.0; // B (bright yellow)
        candidatesPoints.geometry.attributes.color.needsUpdate = true;

        selectedStarIndex = index;

        showStarInfo(selected);
    } else {
        // Restore color when clicking empty space
        if (selectedStarIndex !== null) {
            const colors = candidatesPoints.geometry.attributes.color.array;
            colors[selectedStarIndex * 3] = originalColors[selectedStarIndex * 3];
            colors[selectedStarIndex * 3 + 1] = originalColors[selectedStarIndex * 3 + 1];
            colors[selectedStarIndex * 3 + 2] = originalColors[selectedStarIndex * 3 + 2];
            candidatesPoints.geometry.attributes.color.needsUpdate = true;
            selectedStarIndex = null;
        }
        document.getElementById('info-panel').style.display = 'none';
    }
}
