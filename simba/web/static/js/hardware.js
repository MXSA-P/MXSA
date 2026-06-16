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
                document.getElementById('imu-display').innerHTML = 
                    `Orientation: <span style="color:var(--accent)">${imu.orientation}</span><br>` +
                    `Tilt Angle: <span style="color:var(--accent)">${imu.tilt_angle}°</span>`;
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
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ action: action, speed: currentSpeed })
            }).then(res => {
                if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
            }).catch(err => console.error("Motor control error:", err));
        }

        function arm(joint, angle) {
            fetch('/api/hardware/arm', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ joint: joint, angle: parseInt(angle) })
            }).then(res => {
                if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
            }).catch(err => console.error("Arm control error:", err));
        }

        function handGrip(val) {
            fetch('/api/hardware/hand', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ grip: parseInt(val) })
            }).then(res => {
                if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
            }).catch(err => console.error("Hand grip error:", err));
        }

        function handAction(action) {
            fetch('/api/hardware/hand', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ action: action })
            }).then(res => {
                if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
            }).catch(err => console.error("Hand action error:", err));
        }

        function calibrateAll() {
            fetch('/api/arm/home', {
                method: 'POST',
                credentials: 'include',
                headers: {'Content-Type': 'application/json'}
            }).then(res => {
                if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
                // Reset all servo sliders to home position
                document.querySelectorAll('.card:last-child input[type="range"]').forEach(slider => {
                    const isGrip = slider.onchange && slider.onchange.toString().includes('handGrip');
                    slider.value = isGrip ? 50 : 90;
                });
                document.getElementById('val-rotation').innerText = '90°';
                document.getElementById('val-elbow').innerText = '90°';
                document.getElementById('val-elbow2').innerText = '90°';
                document.getElementById('val-wrist').innerText = '90°';
                document.getElementById('val-grip').innerText = '50%';
                alert('Servos calibrated');
            }).catch(err => console.error("Calibrate error:", err));
        }

        // Keyboard shortcuts for D-pad (WASD)
        const keyActionMap = { w: 'forward', a: 'left', s: 'backward', d: 'right' };
        const keysDown = new Set();

        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT') return; // ignore when typing in inputs
            const key = e.key.toLowerCase();
            if (keyActionMap[key] && !keysDown.has(key)) {
                keysDown.add(key);
                motor(keyActionMap[key]);
            }
        });

        document.addEventListener('keyup', function(e) {
            const key = e.key.toLowerCase();
            if (keyActionMap[key] && keysDown.has(key)) {
                keysDown.delete(key);
                if (keysDown.size === 0) motor('stop');
            }
        });
