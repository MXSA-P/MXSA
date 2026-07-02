// _max_cyan_ — project_mxsa
// hardware command center controller

(function () {
    "use strict";

    // Ensure HTTP Basic Auth credentials are sent with all API fetch requests
    const _originalFetch = window.fetch;
    window.fetch = function(url, options = {}) {
        options = options || {};
        if (!options.credentials) {
            options.credentials = 'include';
        }
        return _originalFetch.call(this, url, options);
    };

    const socket = io({
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: Infinity
    });

    let currentSpeed = 50;

    socket.on('status_update', function(data) {
        if(data && data.robot && data.robot.imu_data) {
            const imu = data.robot.imu_data;
            var imuEl = document.getElementById('imu-display');
            if (imuEl) {
                imuEl.textContent = '';
                var orientLabel = document.createTextNode('Orientation: ');
                var orientVal = document.createElement('span');
                orientVal.style.color = 'var(--accent)';
                orientVal.textContent = imu.orientation;
                var br = document.createElement('br');
                var tiltLabel = document.createTextNode('Tilt Angle: ');
                var tiltVal = document.createElement('span');
                tiltVal.style.color = 'var(--accent)';
                tiltVal.textContent = imu.tilt_angle + '°';
                imuEl.appendChild(orientLabel);
                imuEl.appendChild(orientVal);
                imuEl.appendChild(br);
                imuEl.appendChild(tiltLabel);
                imuEl.appendChild(tiltVal);
            }
        }
    });

    function updateSpeedVal(val) {
        currentSpeed = parseInt(val);
        document.getElementById('val-speed').innerText = val + '%';
    }

    function updateVal(id, val) {
        document.getElementById(id).innerText = val + (id.includes('grip') ? '' : '°');
    }

    function motor(action) {
        fetch('/api/hardware/motor', {
            method: 'POST',
            credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ action: action, speed: currentSpeed })
        }).then(function (res) {
            if (!res.ok) throw new Error('HTTP error: ' + res.status);
        }).catch(function (err) { console.error("Motor control error:", err); });
    }

    function arm(joint, angle) {
        fetch('/api/hardware/arm', {
            method: 'POST',
            credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ joint: joint, angle: parseInt(angle) })
        }).then(function (res) {
            if (!res.ok) throw new Error('HTTP error: ' + res.status);
        }).catch(function (err) { console.error("Arm control error:", err); });
    }

    function handGrip(val) {
        // Server expects grip as string "open" or "close"
        var gripVal = parseInt(val);
        var gripType = gripVal < 50 ? "open" : "close";
        fetch('/api/hardware/hand', {
            method: 'POST',
            credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ grip: gripType })
        }).then(function (res) {
            if (!res.ok) throw new Error('HTTP error: ' + res.status);
        }).catch(function (err) { console.error("Hand grip error:", err); });
    }

    function handAction(action) {
        fetch('/api/hardware/hand', {
            method: 'POST',
            credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ action: action })
        }).then(function (res) {
            if (!res.ok) throw new Error('HTTP error: ' + res.status);
        }).catch(function (err) { console.error("Hand action error:", err); });
    }

    function calibrateAll() {
        fetch('/api/arm/home', {
            method: 'POST',
            credentials: 'include',
            headers: {'Content-Type': 'application/json'}
        }).then(function (res) {
            if (!res.ok) throw new Error('HTTP error: ' + res.status);
            // Reset all servo sliders to home position
            document.querySelectorAll('input[type="range"]').forEach(function (slider) {
                if (slider.id === 'motor-speed') return;
                if (slider.id === 'tune-angle') return;
                if (slider.id === 'tune-pin' || slider.id === 'tune-min' || slider.id === 'tune-max') return;
                slider.value = 90;
            });
            document.getElementById('val-rotation').innerText = '90°';
            document.getElementById('val-elbow').innerText = '90°';
            document.getElementById('val-elbow2').innerText = '90°';
            document.getElementById('val-wrist').innerText = '90°';
            document.getElementById('val-grip').innerText = '50%';
            alert('Servos calibrated');
        }).catch(function (err) { console.error("Calibrate error:", err); });
    }

    // Keyboard shortcuts for D-pad (WASD)
    var keyActionMap = { w: 'forward', a: 'left', s: 'backward', d: 'right' };
    var keysDown = new Set();

    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT') return;
        var key = e.key.toLowerCase();
        if (keyActionMap[key] && !keysDown.has(key)) {
            keysDown.add(key);
            motor(keyActionMap[key]);
        }
    });

    document.addEventListener('keyup', function(e) {
        var key = e.key.toLowerCase();
        if (keyActionMap[key] && keysDown.has(key)) {
            keysDown.delete(key);
            if (keysDown.size === 0) motor('stop');
        }
    });

    // --- Advanced Servo Tuning ---
    function updateTuneAngle(val) {
        document.getElementById('tune-angle-val').innerText = val + '°';
        var pMin = parseInt(document.getElementById('tune-min').value);
        var pMax = parseInt(document.getElementById('tune-max').value);
        var pulse = Math.round(pMin + (parseInt(val) / 180.0) * (pMax - pMin));
        document.getElementById('tune-result').innerText = 'Calculated Pulse: ' + pulse + ' µs';
    }

    function sendTuneData() {
        var pin = document.getElementById('tune-pin').value;
        var pMin = document.getElementById('tune-min').value;
        var pMax = document.getElementById('tune-max').value;
        var angle = document.getElementById('tune-angle').value;
        
        fetch('/api/hardware/servo_test', {
            method: 'POST',
            credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                pin: parseInt(pin), 
                pulse_min: parseInt(pMin), 
                pulse_max: parseInt(pMax), 
                angle: parseInt(angle) 
            })
        }).then(function (res) {
            if (!res.ok) throw new Error('HTTP error: ' + res.status);
        }).catch(function (err) { console.error("Servo tune error:", err); });
    }

    // Expose functions globally for inline onclick/oninput handlers
    window.updateSpeedVal = updateSpeedVal;
    window.updateVal = updateVal;
    window.motor = motor;
    window.arm = arm;
    window.handGrip = handGrip;
    window.handAction = handAction;
    window.calibrateAll = calibrateAll;
    window.updateTuneAngle = updateTuneAngle;
    window.sendTuneData = sendTuneData;

})();

