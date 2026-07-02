// _max_cyan_ — project_mxsa — trainer v2 javascript

// Ensure HTTP Basic Auth credentials are sent with all API fetch requests
const _originalFetch = window.fetch;
window.fetch = function(url, options = {}) {
    options = options || {};
    if (!options.credentials) {
        options.credentials = 'include';
    }
    return _originalFetch.call(this, url, options);
};

// ---- state ----
let selectedFiles = [];
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimer = null;
let recordingSeconds = 0;
let cameraStream = null;
let trainingPollTimer = null;

// ---- init ----
document.addEventListener('DOMContentLoaded', () => {
    loadObjects();
    loadVoiceSamples();
    loadModelStats();
    setupDropZone();
    setupInputLowercase();
    loadPiCredentials();
    setupCameraBox();
});

// ---- toast ----
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(30px)';
        toast.style.transition = 'all 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ---- force lowercase ----
function setupInputLowercase() {
    const nameInput = document.getElementById('object-name');
    if (nameInput) {
        nameInput.addEventListener('input', (e) => {
            e.target.value = e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_');
        });
    }
}

// ---- camera ----
function setupCameraBox() {
    const box = document.getElementById('camera-box');
    const placeholder = document.getElementById('camera-placeholder');
    if (placeholder) {
        placeholder.addEventListener('click', startCamera);
    }
}

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280, max: 1920 }, height: { ideal: 720, max: 1080 }, frameRate: { ideal: 30, max: 60 } }
        });
        const video = document.getElementById('camera-preview');
        video.srcObject = cameraStream;
        video.style.display = 'block';
        document.getElementById('camera-placeholder').style.display = 'none';
        document.getElementById('camera-controls').style.display = 'flex';
        showToast('camera opened', 'info');
    } catch (err) {
        showToast('camera access denied: ' + err.message, 'error');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    const video = document.getElementById('camera-preview');
    video.style.display = 'none';
    video.srcObject = null;
    document.getElementById('camera-placeholder').style.display = 'flex';
    document.getElementById('camera-controls').style.display = 'none';
}

function snapPhoto() {
    const video = document.getElementById('camera-preview');
    const canvas = document.getElementById('camera-canvas');
    if (!video || !video.srcObject) { showToast('camera not active', 'error'); return; }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
        if (blob) {
            const file = new File([blob], `snap_${Date.now()}.jpg`, { type: 'image/jpeg' });
            selectedFiles.push(file);
            addThumbnail(file);
            updateImageCount();
            showToast('photo captured!', 'info');
        }
    }, 'image/jpeg', 0.9);
}

// ---- object management ----
function loadObjects() {
    fetch('/api/objects')
        .then(r => r.json())
        .then(data => {
            const grid = document.getElementById('object-grid');
            const empty = document.getElementById('no-objects');
            const classes = data.classes || [];

            if (classes.length === 0) { empty.style.display = 'block'; return; }
            empty.style.display = 'none';
            grid.innerHTML = '';

            classes.forEach(obj => {
                const card = document.createElement('div');
                card.className = 'object-card';
                let thumbsHtml = '';
                const images = obj.images || [];
                if (images.length > 0) {
                    thumbsHtml = '<div class="obj-thumbs">';
                    images.slice(0, 4).forEach(img => {
                        thumbsHtml += `<img src="/api/object_thumbnail/${obj.name}/${img}" alt="${obj.name}" loading="lazy" decoding="async" width="60" height="60" style="object-fit: cover;">`;
                    });
                    thumbsHtml += '</div>';
                }
                card.innerHTML = `
                    <button class="obj-delete" onclick="deleteObject('${obj.name}')" title="delete">✕</button>
                    <div class="obj-name">${obj.name}</div>
                    <div class="obj-count">${obj.count} images</div>
                    ${thumbsHtml}
                    <button class="btn btn-secondary" style="width: 100%; margin-top: 10px; padding: 5px; font-size: 0.8rem;" onclick="openManageObjectModal('${obj.name}')">⚙️ manage</button>
                `;
                grid.appendChild(card);
            });
            document.getElementById('stat-objects').textContent = classes.length;
        })
        .catch(err => console.error('load objects:', err));
}

function openAddObjectModal(presetName = '') {
    document.getElementById('add-object-modal').style.display = 'flex';
    const nameInput = document.getElementById('object-name');
    nameInput.value = presetName;
    nameInput.disabled = !!presetName; // lock input if preset
    document.getElementById('image-preview').innerHTML = '';
    selectedFiles = [];
    updateImageCount();
}

function closeAddObjectModal() {
    document.getElementById('add-object-modal').style.display = 'none';
    stopCamera();
}

// --- manage object modal ---
let currentManageClass = '';

function openManageObjectModal(className) {
    currentManageClass = className;
    document.getElementById('manage-modal-title').textContent = `manage: ${className}`;
    document.getElementById('manage-object-modal').style.display = 'flex';
    document.getElementById('manage-delete-class-btn').onclick = () => deleteObjectFromManage(className);
    loadManageObjectImages();
}

function closeManageObjectModal() {
    document.getElementById('manage-object-modal').style.display = 'none';
    currentManageClass = '';
    loadObjects(); // refresh the main grid
}

function openAddObjectModalFromManage() {
    closeManageObjectModal();
    openAddObjectModal(currentManageClass);
}

function loadManageObjectImages() {
    if (!currentManageClass) return;
    fetch(`/api/objects/${currentManageClass}/images`)
        .then(r => r.json())
        .then(data => {
            const grid = document.getElementById('manage-image-grid');
            grid.innerHTML = '';
            if (data.images && data.images.length > 0) {
                data.images.forEach(img => {
                    const el = document.createElement('div');
                    el.className = 'manage-thumb';
                    el.innerHTML = `
                        <img src="/api/object_thumbnail/${currentManageClass}/${img}" alt="thumbnail" loading="lazy" decoding="async">
                        <button class="remove" onclick="deleteObjectImage('${currentManageClass}', '${img}')" title="delete image">✕</button>
                    `;
                    grid.appendChild(el);
                });
            } else {
                grid.innerHTML = '<p class="muted">no images in this class.</p>';
            }
        })
        .catch(err => console.error('load object images:', err));
}

function deleteObjectImage(className, filename) {
    if (!confirm('delete this image permanently?')) return;
    fetch(`/api/objects/${className}/${filename}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(res => {
            if (res.error) showToast(res.error, 'error');
            else {
                showToast(res.message, 'success');
                loadManageObjectImages(); // reload the grid inside the modal
            }
        })
        .catch(err => console.error('delete image:', err));
}

function deleteObjectFromManage(className) {
    if (!confirm(`delete the ENTIRE class '${className}' and ALL its images?`)) return;
    fetch(`/api/objects/${className}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(res => {
            if (res.error) showToast(res.error, 'error');
            else {
                showToast(res.message, 'success');
                closeManageObjectModal();
            }
        })
        .catch(err => console.error('delete class:', err));
}

function submitObject() {
    const name = document.getElementById('object-name').value.trim().toLowerCase();
    if (!name) { showToast('enter an object name', 'error'); return; }
    
    const validFiles = selectedFiles.filter(f => f !== null);
    if (validFiles.length === 0) { showToast('add at least one image', 'error'); return; }

    const formData = new FormData();
    formData.append('name', name);
    validFiles.forEach(file => formData.append('images', file));

    fetch('/api/add_object', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.error) { showToast(data.error, 'error'); return; }
            showToast(`added "${name}" with ${data.saved || selectedFiles.length} images`);
            closeAddObjectModal();
            loadObjects();
        })
        .catch(err => showToast('upload failed: ' + err.message, 'error'));
}

function deleteObject(name) {
    if (!confirm(`delete "${name}" and all its images?`)) return;
    fetch(`/api/objects/${name}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(data => {
            if (data.error) { showToast(data.error, 'error'); return; }
            showToast(`deleted "${name}"`);
            loadObjects();
        });
}

// ---- default objects ----
function loadDefaultObjects() {
    const btn = document.getElementById('default-btn');
    btn.disabled = true;
    btn.textContent = 'creating...';

    fetch('/api/default_objects', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            btn.disabled = false;
            btn.textContent = '+ defaults';
            if (data.error) { showToast(data.error, 'error'); return; }
            showToast(data.message || `created ${data.classes_created} default classes`);
            loadObjects();
        })
        .catch(err => {
            btn.disabled = false;
            btn.textContent = '+ defaults';
            showToast('error: ' + err.message, 'error');
        });
}

// ---- drop zone ----
function setupDropZone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('image-input');
    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
}

function handleFiles(files) {
    for (const file of files) {
        if (!file.type.startsWith('image/')) continue;
        selectedFiles.push(file);
        addThumbnail(file);
    }
    updateImageCount();
}

function addThumbnail(file) {
    const preview = document.getElementById('image-preview');
    const idx = selectedFiles.length - 1;
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    thumb.dataset.index = idx;

    const img = document.createElement('img');
    const reader = new FileReader();
    reader.onload = (e) => { img.src = e.target.result; };
    reader.readAsDataURL(file);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove';
    removeBtn.textContent = '✕';
    removeBtn.onclick = (e) => {
        e.stopPropagation();
        const i = parseInt(thumb.dataset.index);
        selectedFiles[i] = null;
        thumb.remove();
        updateImageCount();
    };

    thumb.appendChild(img);
    thumb.appendChild(removeBtn);
    preview.appendChild(thumb);
}

function updateImageCount() {
    const count = selectedFiles.filter(f => f !== null).length;
    const el = document.getElementById('image-count');
    if (el) el.textContent = count > 0 ? `${count} image(s) selected` : '';
}

// ---- training ----
function trainVision() {
    const btn = document.getElementById('train-vision-btn');
    btn.disabled = true;
    btn.textContent = 'training...';

    const section = document.getElementById('progress-section');
    section.style.display = 'block';
    document.getElementById('progress-text').textContent = 'starting vision training...';
    document.getElementById('progress-bar').style.width = '5%';

    fetch('/api/train_vision', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                btn.disabled = false; btn.textContent = 'train model';
                showToast(data.error, 'error');
                document.getElementById('progress-text').textContent = 'error: ' + data.error;
                return;
            }
            showToast('training started');
            startTrainingPoll(btn, 'train model');
        })
        .catch(err => {
            btn.disabled = false; btn.textContent = 'train model';
            showToast('error: ' + err.message, 'error');
        });
}

function startTrainingPoll(btn, originalText) {
    if (trainingPollTimer) clearInterval(trainingPollTimer);
    trainingPollTimer = setInterval(() => {
        fetch('/api/training_status').then(r => r.json()).then(status => {
            const bar = document.getElementById('progress-bar');
            const text = document.getElementById('progress-text');
            if (status.progress != null) bar.style.width = status.progress + '%';
            if (status.message) text.textContent = status.message;

            if (!status.active) {
                clearInterval(trainingPollTimer);
                trainingPollTimer = null;
                btn.disabled = false;
                btn.textContent = originalText;
                if (status.error) {
                    showToast('training failed: ' + status.error, 'error');
                } else {
                    showToast('training complete!');
                    bar.style.width = '100%';
                }
                loadModelStats();
            }
        }).catch(() => {});
    }, 2000);
}

// ---- voice ----
function toggleRecording() { isRecording ? stopRecording() : startRecording(); }

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: 'audio/wav' });
            uploadVoiceSample(blob);
            stream.getTracks().forEach(t => t.stop());
        };
        mediaRecorder.start();
        isRecording = true;
        recordingSeconds = 0;
        document.getElementById('record-btn').classList.add('recording');
        document.getElementById('record-text').textContent = 'stop recording';
        recordingTimer = setInterval(() => {
            recordingSeconds++;
            const m = Math.floor(recordingSeconds / 60);
            const s = recordingSeconds % 60;
            document.getElementById('recording-time').textContent = `${m}:${s.toString().padStart(2, '0')}`;
        }, 1000);
    } catch (err) { showToast('microphone access denied', 'error'); }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    isRecording = false;
    clearInterval(recordingTimer);
    document.getElementById('record-btn').classList.remove('recording');
    document.getElementById('record-text').textContent = 'start recording';
}

function uploadVoiceSample(blob) {
    const formData = new FormData();
    formData.append('audio', blob, `voice_${Date.now()}.wav`);
    fetch('/api/record_voice', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.error) { showToast(data.error, 'error'); return; }
            showToast('voice sample saved');
            loadVoiceSamples();
        });
}

function loadVoiceSamples() {
    fetch('/api/voice_samples').then(r => r.json()).then(data => {
        const container = document.getElementById('voice-samples');
        container.innerHTML = '';
        const samples = data.samples || [];
        samples.forEach((s, i) => {
            const div = document.createElement('div');
            div.className = 'voice-sample';
            const size = s.size ? (s.size / 1024).toFixed(1) + ' kb' : '';
            div.innerHTML = `
                <span>🎵 sample ${i + 1} — ${s.name}</span>
                <div>
                    <span class="muted" style="margin-right: 10px;">${size}</span>
                    <button class="obj-delete" style="position:static; padding: 2px 6px;" onclick="deleteVoiceSample('${s.name}')" title="delete sample">✕</button>
                </div>
            `;
            container.appendChild(div);
        });
        document.getElementById('stat-voice').textContent = samples.length;
    }).catch(() => {});
}

function deleteVoiceSample(name) {
    if (!confirm(`delete voice sample ${name}?`)) return;
    fetch(`/api/voice_samples/${name}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(res => {
            if (res.error) showToast(res.error, 'error');
            else {
                showToast(res.message, 'success');
                loadVoiceSamples();
            }
        })
        .catch(err => console.error('delete voice:', err));
}

function trainVoice() {
    const btn = document.getElementById('train-voice-btn');
    btn.disabled = true; btn.textContent = 'training...';
    fetch('/api/train_voice', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.error) { showToast(data.error, 'error'); }
            else { showToast('voice training started'); startTrainingPoll(btn, 'train voice model'); return; }
            btn.disabled = false; btn.textContent = 'train voice model';
        })
        .catch(err => { btn.disabled = false; btn.textContent = 'train voice model'; showToast('error: ' + err.message, 'error'); });
}

// ---- export ----
function exportModels() {
    const btn = document.getElementById('export-btn');
    btn.disabled = true; btn.textContent = 'exporting...';
    fetch('/api/export', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            btn.disabled = false; btn.textContent = 'export to .zip';
            if (data.error) { showToast(data.error, 'error'); return; }
            showToast('export started');
            startTrainingPoll(btn, 'export to .zip');
        })
        .catch(err => { btn.disabled = false; btn.textContent = 'export to .zip'; showToast('error: ' + err.message, 'error'); });
}

// ---- model stats ----
function loadModelStats() {
    fetch('/api/model_stats').then(r => r.json()).then(stats => {
        const list = document.getElementById('model-list');
        list.innerHTML = '';
        const models = ['object_classifier', 'object_labels', 'speaker_model', 'behavior_model'];
        let totalSize = 0;
        models.forEach(name => {
            const s = stats[name] || {};
            const div = document.createElement('div');
            div.className = 'model-item';
            if (s.exists) {
                const sizeKb = (s.size_bytes / 1024).toFixed(1);
                totalSize += s.size_bytes;
                div.innerHTML = `<span>${name.replace(/_/g, ' ')}</span><span class="exists">✓ ${sizeKb} kb</span>`;
            } else {
                div.innerHTML = `<span>${name.replace(/_/g, ' ')}</span><span class="missing">not trained</span>`;
            }
            list.appendChild(div);
        });
        document.getElementById('stat-size').textContent = totalSize > 0 ? (totalSize / 1024).toFixed(0) + ' kb' : '—';
    }).catch(() => {});
}

// ---- deploy to pi ----
function loadPiCredentials() {
    const saved = localStorage.getItem('simba_pi_creds');
    if (saved) {
        try {
            const creds = JSON.parse(saved);
            document.getElementById('pi-host').value = creds.host || '';
            document.getElementById('pi-user').value = creds.username || '';
            document.getElementById('pi-path').value = creds.path || '/home/pi/MXSA/models';
        } catch (e) {}
    }
}

function savePiCredentials() {
    const creds = {
        host: document.getElementById('pi-host').value,
        username: document.getElementById('pi-user').value,
        path: document.getElementById('pi-path').value,
    };
    localStorage.setItem('simba_pi_creds', JSON.stringify(creds));
}

function deployToPi() {
    const host = document.getElementById('pi-host').value.trim();
    const username = document.getElementById('pi-user').value.trim();
    const password = document.getElementById('pi-pass').value;
    const remotePath = document.getElementById('pi-path').value.trim() || '/home/pi/MXSA/models';

    if (!host || !username || !password) { showToast('fill in all pi connection fields', 'error'); return; }

    savePiCredentials();

    const btn = document.getElementById('deploy-btn');
    btn.disabled = true;
    const status = document.getElementById('deploy-status');
    status.style.display = 'block';
    document.getElementById('deploy-message').textContent = 'connecting to pi...';
    document.getElementById('deploy-progress').style.width = '30%';

    fetch('/api/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ host, username, password, remote_path: remotePath }),
    })
    .then(r => r.json())
    .then(data => {
        btn.disabled = false;
        if (data.error) {
            document.getElementById('deploy-message').textContent = 'failed: ' + data.error;
            document.getElementById('deploy-progress').style.width = '0%';
            showToast('deploy failed: ' + data.error, 'error');
        } else {
            document.getElementById('deploy-message').textContent = data.message || 'models uploaded successfully!';
            document.getElementById('deploy-progress').style.width = '100%';
            showToast('models deployed to pi! 🎉');
        }
    })
    .catch(err => {
        btn.disabled = false;
        document.getElementById('deploy-message').textContent = 'connection error';
        showToast('deploy error: ' + err.message, 'error');
    });
}

// -----------------------------------------------------------------------------
// owner profile
// -----------------------------------------------------------------------------
async function loadOwner() {
    try {
        const res = await fetch('/api/owner');
        const data = await res.json();
        if (data.owner && data.owner !== 'unknown') {
            document.getElementById('owner-name').value = data.owner;
        }
    } catch (e) {
        console.error('Failed to load owner:', e);
    }
}

async function saveOwner() {
    const ownerName = document.getElementById('owner-name').value.trim();
    if (!ownerName) {
        showToast('Please enter a name first', 'error');
        return;
    }
    
    const btn = document.getElementById('owner-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">⏳</span> saving...';
    
    try {
        const res = await fetch('/api/owner', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({owner_name: ownerName})
        });
        const data = await res.json();
        
        if (data.error) {
            showToast('Error: ' + data.error, 'error');
        } else {
            showToast(data.message, 'success');
        }
    } catch (e) {
        showToast('Failed to save identity', 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">💾</span> save identity';
    }
}

// Call loadOwner on page load
document.addEventListener('DOMContentLoaded', () => {
    loadOwner();
});

// -----------------------------------------------------------------------------
// voice test
// -----------------------------------------------------------------------------
function testVoiceCommand() {
    const btn = document.getElementById('voice-test-btn');
    const input = document.getElementById('voice-test-input');
    
    // Check if SpeechRecognition is available
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showToast('Browser does not support voice dictation. Please type instead.', 'error');
        return;
    }
    
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onstart = function() {
        btn.innerHTML = '<span class="btn-icon">🛑</span> listening...';
        btn.classList.add('pulsing');
    };
    
    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        input.value = transcript;
        submitVoiceTest();
    };
    
    recognition.onerror = function(event) {
        showToast('Microphone error: ' + event.error, 'error');
    };
    
    recognition.onend = function() {
        btn.innerHTML = '<span class="btn-icon">🎤</span> dictate';
        btn.classList.remove('pulsing');
    };
    
    recognition.start();
}

async function submitVoiceTest() {
    const text = document.getElementById('voice-test-input').value.trim();
    if (!text) return;
    
    try {
        const res = await fetch('/api/test_voice', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text})
        });
        const data = await res.json();
        
        const resultBox = document.getElementById('voice-test-result');
        const heardSpan = document.getElementById('vt-heard');
        const parsedSpan = document.getElementById('vt-parsed');
        
        resultBox.style.display = 'block';
        heardSpan.textContent = data.text;
        
        if (data.command) {
            parsedSpan.innerHTML = `ACTION: ${data.command.action}<br>TARGET: ${data.command.target || 'none'}`;
            parsedSpan.style.color = '#00d4ff';
        } else {
            parsedSpan.innerHTML = 'NO COMMAND RECOGNIZED';
            parsedSpan.style.color = '#ff4444';
        }
    } catch (e) {
        showToast('Failed to test voice command', 'error');
    }
}
