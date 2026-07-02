// _max_cyan_ — project_mxsa
// real-time dashboard controller for simba

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

    // -----------------------------------------------------------------------
    // constants
    // -----------------------------------------------------------------------

    const GAUGE_CIRCUMFERENCE = 283; // 2 * pi * 45
    const EMOTION_MAP = {
        curious:   "🔍",
        happy:     "😊",
        excited:   "🤩",
        sad:       "😢",
        love:      "❤️",
        angry:     "😠",
        idle:      "😐",
        thinking:  "🤔",
        scared:    "😨",
        neutral:   "🙂",
        hello:     "👋",
    };

    const MAX_LOG_ENTRIES = 60;
    const TYPING_SPEED_MS = 18;

    // -----------------------------------------------------------------------
    // dom references
    // -----------------------------------------------------------------------

    const dom = {
        cpuGauge:        document.getElementById("cpu-gauge-fill"),
        cpuText:         document.getElementById("cpu-value"),
        ramGauge:        document.getElementById("ram-gauge-fill"),
        ramText:         document.getElementById("ram-value"),
        tempValue:       document.getElementById("temp-value"),
        ramDetail:       document.getElementById("ram-detail"),
        stateBadge:      document.getElementById("state-badge"),
        stateContext:    document.getElementById("state-context-text"),
        emotionEmoji:    document.getElementById("emotion-emoji"),
        thinkingBox:     document.getElementById("thinking-text"),
        logContainer:    document.getElementById("log-container"),
        memoryBody:      document.getElementById("memory-body"),
        memoryEmpty:     document.getElementById("memory-empty"),
        commandInput:    document.getElementById("command-input"),
        commandBtn:      document.getElementById("command-btn"),
        cameraFeed:      document.getElementById("camera-feed"),
        uptimeDisplay:   document.getElementById("uptime-display"),
        connectionDot:   document.getElementById("connection-dot"),
        objectChips:     document.getElementById("object-chips"),
    };

    // -----------------------------------------------------------------------
    // state
    // -----------------------------------------------------------------------

    let currentCpu        = 0;
    let currentRam        = 0;
    let lastState         = "";
    let lastEmotion       = "";
    let thinkingText      = "";
    let typingTarget      = "";
    let typingIndex       = 0;
    let typingTimer       = null;
    let renderedLogIds    = new Set();

    // -----------------------------------------------------------------------
    // socket.io
    // -----------------------------------------------------------------------

    const socket = io({ reconnection: true, reconnectionDelay: 2000 });

    socket.on("connect", function () {
        setConnectionStatus(true);
    });

    socket.on("disconnect", function () {
        setConnectionStatus(false);
    });

    socket.on("status_update", function (data) {
        if (data.system)  updateGauges(data.system);
        if (data.robot)   updateRobotState(data.robot);
        if (data.logs)    updateLogs(data.logs);
        if (data.uptime != null) updateUptime(data.uptime);
    });

    // -----------------------------------------------------------------------
    // connection status
    // -----------------------------------------------------------------------

    function setConnectionStatus(connected) {
        if (!dom.connectionDot) return;
        if (connected) {
            dom.connectionDot.classList.add("connection-dot--connected");
        } else {
            dom.connectionDot.classList.remove("connection-dot--connected");
        }
    }

    // -----------------------------------------------------------------------
    // gauge animation
    // -----------------------------------------------------------------------

    function animateGauge(element, textElement, currentVal, targetVal, suffix) {
        if (!element || !textElement) return targetVal;

        const offset = GAUGE_CIRCUMFERENCE - (targetVal / 100) * GAUGE_CIRCUMFERENCE;
        element.style.strokeDashoffset = offset;
        textElement.textContent = Math.round(targetVal) + (suffix || "%");

        return targetVal;
    }

    function updateGauges(system) {
        currentCpu = animateGauge(
            dom.cpuGauge, dom.cpuText,
            currentCpu, system.cpu_percent, "%"
        );

        currentRam = animateGauge(
            dom.ramGauge, dom.ramText,
            currentRam, system.ram_percent, "%"
        );

        if (dom.tempValue) {
            dom.tempValue.textContent = system.temperature
                ? system.temperature + "°c"
                : "—";
        }

        if (dom.ramDetail) {
            dom.ramDetail.textContent = system.ram_used_mb
                ? system.ram_used_mb + " / " + system.ram_total_mb + " mb"
                : "";
        }
    }

    // -----------------------------------------------------------------------
    // robot state
    // -----------------------------------------------------------------------

    function updateRobotState(robot) {
        // state badge
        if (dom.stateBadge && robot.state !== lastState) {
            dom.stateBadge.style.opacity = "0";
            setTimeout(function () {
                dom.stateBadge.textContent = robot.state;
                dom.stateBadge.style.opacity = "1";
            }, 200);
            lastState = robot.state;
        }

        if (dom.stateContext) {
            dom.stateContext.textContent = robot.state_description || "";
        }

        // emotion
        if (dom.emotionEmoji && robot.emotion !== lastEmotion) {
            dom.emotionEmoji.style.animation = "none";
            void dom.emotionEmoji.offsetHeight; // trigger reflow
            dom.emotionEmoji.style.animation = "";
            dom.emotionEmoji.textContent = EMOTION_MAP[robot.emotion] || "🙂";
            lastEmotion = robot.emotion;
        }

        // thinking
        if (robot.thinking != null) {
            startTypingEffect(robot.thinking);
        }

        // spoken text feedback
        if (dom.commandInput) {
            if (robot.spoken_text) {
                dom.commandInput.placeholder = '🎙️ Heard: "' + robot.spoken_text + '"';
            } else if (robot.spoken_text === "") {
                dom.commandInput.placeholder = 'type a command for simba...';
            }
        }

        // detected objects
        if (dom.objectChips) {
            updateObjectChips(robot.detected_objects || []);
        }

        // memory table
        if (robot.memory) {
            updateMemoryTable(robot.memory);
        }

        // servo gauges
        if (robot.servo_angles) {
            updateServoGauges(robot.servo_angles);
        }

        // motor bars
        if (robot.motor_state) {
            updateMotorStatus(robot.motor_state);
        }

        // imu data
        if (robot.imu_data) {
            updateImuData(robot.imu_data);
        }

        // yolo boxes
        if (robot.detected_objects) {
            updateYoloBoxes(robot.detected_objects);
        }
    }

    function updateYoloBoxes(detections) {
        var cameraContainer = document.getElementById("camera-container");
        if (!cameraContainer) return;

        // Clear old boxes
        var oldBoxes = cameraContainer.querySelectorAll(".yolo-box");
        oldBoxes.forEach(b => b.remove());

        if (!detections || detections.length === 0) return;

        detections.forEach(function(det) {
            if (det.box && det.box.length === 4) {
                // box is [ymin, xmin, ymax, xmax] in pixels
                // camera feed is 640x480
                var ymin = det.box[0];
                var xmin = det.box[1];
                var ymax = det.box[2];
                var xmax = det.box[3];

                var widthPct = ((xmax - xmin) / 640.0) * 100;
                var heightPct = ((ymax - ymin) / 480.0) * 100;
                var leftPct = (xmin / 640.0) * 100;
                var topPct = (ymin / 480.0) * 100;

                var boxDiv = document.createElement("div");
                boxDiv.className = "yolo-box";
                boxDiv.style.left = leftPct + "%";
                boxDiv.style.top = topPct + "%";
                boxDiv.style.width = widthPct + "%";
                boxDiv.style.height = heightPct + "%";

                var labelDiv = document.createElement("div");
                labelDiv.className = "yolo-label";
                var conf = det.confidence ? Math.round(det.confidence * 100) + "%" : "";
                labelDiv.textContent = (det.label || "unknown") + " " + conf;

                boxDiv.appendChild(labelDiv);
                cameraContainer.appendChild(boxDiv);
            }
        });
    }

    // -----------------------------------------------------------------------
    // servo, motor, imu updates
    // -----------------------------------------------------------------------

    const domServos = {
        rotationFill: document.getElementById("servo-rotation-fill"),
        rotationVal:  document.getElementById("servo-rotation-val"),
        elbowFill:    document.getElementById("servo-elbow-fill"),
        elbowVal:     document.getElementById("servo-elbow-val"),
        wristFill:    document.getElementById("servo-wrist-fill"),
        wristVal:     document.getElementById("servo-wrist-val"),
        finger1:      document.querySelector("#finger-1 .finger-angle"),
        finger2:      document.querySelector("#finger-2 .finger-angle"),
        finger3:      document.querySelector("#finger-3 .finger-angle"),
        gripState:    document.getElementById("grip-state-display"),
    };

    function updateServoGauges(servos) {
        if (!domServos.rotationFill) return;
        const trackLength = 157; // approximate length for A 50,50 arc

        const updateGauge = (fill, valEl, degrees) => {
            if (!fill) return;
            const percentage = Math.max(0, Math.min(180, degrees)) / 180;
            fill.style.strokeDasharray = trackLength;
            fill.style.strokeDashoffset = trackLength * (1 - percentage);
            valEl.textContent = Math.round(degrees) + "°";
        };

        updateGauge(domServos.rotationFill, domServos.rotationVal, servos.rotation || 90);
        updateGauge(domServos.elbowFill, domServos.elbowVal, servos.elbow || 90);
        updateGauge(domServos.wristFill, domServos.wristVal, servos.wrist || 90);

        if (domServos.finger1) domServos.finger1.textContent = Math.round(servos.finger_1 || 30) + "°";
        if (domServos.finger2) domServos.finger2.textContent = Math.round(servos.finger_2 || 30) + "°";
        if (domServos.finger3) domServos.finger3.textContent = Math.round(servos.finger_3 || 30) + "°";
    }

    const domMotors = {
        leftSpeed:    document.getElementById("motor-left-speed"),
        leftFill:     document.getElementById("motor-left-fill"),
        rightSpeed:   document.getElementById("motor-right-speed"),
        rightFill:    document.getElementById("motor-right-fill"),
        dirArrow:     document.getElementById("direction-arrow"),
        dirLabel:     document.getElementById("direction-label"),
        imuOrient:    document.getElementById("imu-orientation"),
        imuTilt:      document.getElementById("imu-tilt"),
    };

    function updateMotorStatus(motor) {
        if (!domMotors.leftSpeed) return;
        
        const updateMotorBar = (speedEl, fillEl, speedVal) => {
            const absSpeed = Math.abs(speedVal || 0);
            const percentage = Math.min(100, (absSpeed / 100) * 100);
            speedEl.textContent = Math.round(speedVal || 0) + "%";
            fillEl.style.width = percentage + "%";
            if (speedVal < 0) {
                fillEl.style.backgroundColor = "var(--red-soft)";
            } else {
                fillEl.style.backgroundColor = "var(--green-soft)";
            }
        };

        updateMotorBar(domMotors.leftSpeed, domMotors.leftFill, motor.left_speed);
        updateMotorBar(domMotors.rightSpeed, domMotors.rightFill, motor.right_speed);

        domMotors.dirLabel.textContent = (motor.direction || "STOPPED").toUpperCase();
        
        const dir = motor.direction || "stopped";
        if (dir === "forward") domMotors.dirArrow.style.transform = "rotate(0deg)";
        else if (dir === "backward") domMotors.dirArrow.style.transform = "rotate(180deg)";
        else if (dir === "turning_left") domMotors.dirArrow.style.transform = "rotate(-90deg)";
        else if (dir === "turning_right") domMotors.dirArrow.style.transform = "rotate(90deg)";
        else domMotors.dirArrow.style.transform = "rotate(0deg)";
    }

    function updateImuData(imu) {
        if (!domMotors.imuOrient) return;
        domMotors.imuOrient.textContent = imu.orientation || "level";
        domMotors.imuTilt.textContent = Math.round(imu.tilt_angle || 0) + "°";
    }

    // -----------------------------------------------------------------------
    // detected objects
    // -----------------------------------------------------------------------

    function updateObjectChips(objects) {
        if (!dom.objectChips) return;

        dom.objectChips.innerHTML = "";
        if (objects.length === 0) return;

        objects.forEach(function (obj) {
            var chip = document.createElement("span");
            chip.className = "object-chip";
            chip.textContent = typeof obj === "string" ? obj : (obj.label || obj.name || "unknown");
            dom.objectChips.appendChild(chip);
        });
    }

    // -----------------------------------------------------------------------
    // typing effect
    // -----------------------------------------------------------------------

    function startTypingEffect(text) {
        if (text === typingTarget) return;
        typingTarget = text;
        typingIndex = 0;

        if (typingTimer) {
            clearInterval(typingTimer);
            typingTimer = null;
        }

        if (!text) {
            if (dom.thinkingBox) {
                dom.thinkingBox.innerHTML = '<span class="thinking-cursor"></span>';
            }
            return;
        }

        typingTimer = setInterval(function () {
            if (typingIndex <= typingTarget.length) {
                if (dom.thinkingBox) {
                    var escaped = escapeHtml(typingTarget.substring(0, typingIndex));
                    dom.thinkingBox.innerHTML = escaped + '<span class="thinking-cursor"></span>';
                }
                typingIndex++;
            } else {
                clearInterval(typingTimer);
                typingTimer = null;
            }
        }, TYPING_SPEED_MS);
    }

    function escapeHtml(text) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

    // -----------------------------------------------------------------------
    // logs
    // -----------------------------------------------------------------------

    function updateLogs(logs) {
        if (!dom.logContainer) return;

        // render only new entries
        var newEntries = [];
        logs.forEach(function (log) {
            var logId = log.timestamp + "|" + log.message;
            if (!renderedLogIds.has(logId)) {
                renderedLogIds.add(logId);
                newEntries.push(log);
            }
        });

        newEntries.forEach(function (log) {
            var entry = createLogEntry(log);
            dom.logContainer.prepend(entry);
        });

        // trim rendered list
        while (dom.logContainer.children.length > MAX_LOG_ENTRIES) {
            dom.logContainer.removeChild(dom.logContainer.lastChild);
        }

        // trim id cache
        if (renderedLogIds.size > MAX_LOG_ENTRIES * 2) {
            var arr = Array.from(renderedLogIds);
            renderedLogIds = new Set(arr.slice(arr.length - MAX_LOG_ENTRIES));
        }
    }

    function createLogEntry(log) {
        var entry = document.createElement("div");
        var category = (log.category || "system").toLowerCase();
        entry.className = "log-entry log-entry--" + category;

        var badge = document.createElement("span");
        badge.className = "log-badge log-badge--" + category;
        badge.textContent = category;

        var message = document.createElement("span");
        message.className = "log-message";
        message.textContent = log.message || "";

        var time = document.createElement("span");
        time.className = "log-time";
        time.textContent = formatRelativeTime(log.timestamp);

        entry.appendChild(badge);
        entry.appendChild(message);
        entry.appendChild(time);

        return entry;
    }

    // -----------------------------------------------------------------------
    // memory table
    // -----------------------------------------------------------------------

    function updateMemoryTable(memory) {
        if (!dom.memoryBody) return;

        if (!memory || memory.length === 0) {
            dom.memoryBody.innerHTML = "";
            if (dom.memoryEmpty) dom.memoryEmpty.style.display = "block";
            return;
        }

        if (dom.memoryEmpty) dom.memoryEmpty.style.display = "none";

        dom.memoryBody.innerHTML = "";
        memory.forEach(function (item) {
            var row = document.createElement("tr");

            var nameCell = document.createElement("td");
            nameCell.textContent = item.label || item.name || "—";

            var countCell = document.createElement("td");
            countCell.textContent = item.count != null ? item.count : "—";

            var confidenceCell = document.createElement("td");
            var conf = item.confidence != null ? (item.confidence * 100).toFixed(0) + "%" : "—";
            confidenceCell.textContent = conf;

            var timeCell = document.createElement("td");
            timeCell.textContent = item.last_seen
                ? formatRelativeTime(item.last_seen)
                : "—";

            row.appendChild(nameCell);
            row.appendChild(countCell);
            row.appendChild(confidenceCell);
            row.appendChild(timeCell);
            dom.memoryBody.appendChild(row);
        });
        
        updateRadarMap(memory);
    }

    function updateRadarMap(memory) {
        var radarContainer = document.getElementById("radar-container");
        if (!radarContainer) return;
        
        // Remove old dots
        var oldDots = radarContainer.querySelectorAll('.radar-dot');
        oldDots.forEach(d => d.remove());
        
        var maxDistance = 100.0; // cm range
        
        memory.forEach(function (item) {
            if (item.x != null && item.y != null) {
                var dot = document.createElement("div");
                dot.className = "radar-dot";
                dot.setAttribute("data-label", item.label || item.name || "");
                
                // Map x, y (-100 to 100) to percentage 0-100%
                var leftPct = 50 + (item.x / maxDistance) * 50;
                // y is forward, so higher y is lower 'top' (closer to center from top)
                var topPct = 50 - (item.y / maxDistance) * 50;
                
                // Clamp
                leftPct = Math.max(0, Math.min(100, leftPct));
                topPct = Math.max(0, Math.min(100, topPct));
                
                dot.style.left = leftPct + "%";
                dot.style.top = topPct + "%";
                
                radarContainer.appendChild(dot);
            }
        });
    }

    // -----------------------------------------------------------------------
    // uptime
    // -----------------------------------------------------------------------

    function updateUptime(seconds) {
        if (!dom.uptimeDisplay) return;
        dom.uptimeDisplay.textContent = formatUptime(seconds);
    }

    function formatUptime(totalSeconds) {
        var days    = Math.floor(totalSeconds / 86400);
        var hours   = Math.floor((totalSeconds % 86400) / 3600);
        var minutes = Math.floor((totalSeconds % 3600) / 60);
        var secs    = Math.floor(totalSeconds % 60);

        var parts = [];
        if (days > 0)    parts.push(days + "d");
        if (hours > 0)   parts.push(hours + "h");
        parts.push(minutes + "m");
        parts.push(secs + "s");

        return parts.join(" ");
    }

    // -----------------------------------------------------------------------
    // relative time
    // -----------------------------------------------------------------------

    function formatRelativeTime(isoString) {
        if (!isoString) return "";
        try {
            var then = new Date(isoString);
            var now  = new Date();
            var diff = Math.floor((now - then) / 1000);

            if (diff < 5)    return "just now";
            if (diff < 60)   return diff + "s ago";
            if (diff < 3600) return Math.floor(diff / 60) + " min ago";
            if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
            return Math.floor(diff / 86400) + "d ago";
        } catch (e) {
            return "";
        }
    }

    // -----------------------------------------------------------------------
    // navigation polling
    // -----------------------------------------------------------------------

    function pollNavigation() {
        fetch("/api/path", { credentials: "include" })
            .then(function (res) { return res.json(); })
            .then(function (data) {
                var stepsEl  = document.getElementById("nav-steps");
                var statusEl = document.getElementById("nav-status");
                if (stepsEl)  stepsEl.textContent  = (data.steps != null ? data.steps : 0) + " steps";
                if (statusEl) statusEl.textContent = data.status || "at base";
            })
            .catch(function (err) {
                console.error("nav poll error:", err);
            });
    }

    pollNavigation();
    setInterval(pollNavigation, 5000);

    // -----------------------------------------------------------------------
    // command input
    // -----------------------------------------------------------------------

    function sendCommand(directCmd) {
        var command;
        if (directCmd) {
            command = directCmd.trim().toLowerCase();
        } else {
            if (!dom.commandInput) return;
            command = dom.commandInput.value.trim().toLowerCase();
        }
        if (!command) return;

        // send via rest
        fetch("/api/command", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: command })
        }).catch(function (err) {
            console.error("command send error:", err);
        });

        // also emit via socket for instant feedback
        socket.emit("send_command", { command: command });

        if (!directCmd && dom.commandInput) {
            dom.commandInput.value = "";
            dom.commandInput.focus();
        }
    }

    // expose globally so inline onclick handlers can call it
    window.sendCommand = sendCommand;

    if (dom.commandInput) {
        dom.commandInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") {
                e.preventDefault();
                sendCommand();
            }
        });
    }

    if (dom.commandBtn) {
        dom.commandBtn.addEventListener("click", function () {
            sendCommand();
        });
    }

    // -----------------------------------------------------------------------
    // camera feed
    // -----------------------------------------------------------------------

    function initCameraFeed() {
        if (!dom.cameraFeed) return;
        dom.cameraFeed.src = "/video_feed";
        dom.cameraFeed.onerror = function () {
            // retry after 3 seconds on error
            setTimeout(function () {
                dom.cameraFeed.src = "/video_feed?" + Date.now();
            }, 3000);
        };
    }

    initCameraFeed();

    // -----------------------------------------------------------------------
    // init
    // -----------------------------------------------------------------------

    // set initial thinking state
    if (dom.thinkingBox) {
        dom.thinkingBox.innerHTML = '<span class="thinking-cursor"></span>';
    }

})();
