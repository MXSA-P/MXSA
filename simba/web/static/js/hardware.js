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
