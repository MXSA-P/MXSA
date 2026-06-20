import re

with open("simba/web/server.py", "r") as f:
    content = f.read()

# 1. Update the throttle limit from 0.1 to 0.015
content = content.replace("if interval < 0.1:\n        interval = 0.1", "if interval < 0.015:\n        interval = 0.015")
content = content.replace("Throttle to max 10Hz (0.1s)", "Throttle to max ~60Hz (0.015s)")

# 2. Refactor _build_status_payload to use caching for slow parts
slow_cache_code = """_last_slow_update = 0.0
_slow_payload_cache = {"system": {}, "robot_base": {}, "logs": []}

def _build_status_payload() -> dict:
    global _last_slow_update, _slow_payload_cache
    now = time.time()
    
    if now - _last_slow_update > 1.0:
        system = _get_system_stats()
        robot_base = _get_robot_state()
        logs = get_log_history(30)
        
        # if brain has get_status(), use that for more complete data
        if _brain is not None and hasattr(_brain, "get_status"):
            try:
                brain_status = _brain.get_status()
                robot_base.update({
                    "state": brain_status.get("state", robot_base["state"]),
                    "emotion": brain_status.get("emotion", robot_base["emotion"]),
                    "emotion_intensity": brain_status.get("emotion_intensity", 0.5),
                    "emotion_emoji": brain_status.get("emotion_emoji", "🤖"),
                    "thinking": brain_status.get("thinking", robot_base["thinking"]),
                    "grip_state": brain_status.get("grip_state", "unknown"),
                    "state_context": brain_status.get("state_context", {}),
                })
                system.update({
                    "cpu_percent": brain_status.get("cpu_percent", system["cpu_percent"]),
                    "ram_percent": brain_status.get("ram_percent", system["ram_percent"]),
                    "ram_used_mb": brain_status.get("ram_used_mb", system["ram_used_mb"]),
                    "ram_total_mb": brain_status.get("ram_total_mb", system["ram_total_mb"]),
                    "temperature": brain_status.get("cpu_temp", system["temperature"]),
                })
            except Exception as exc:
                logger.debug(f"brain get_status error: {exc}")
                
        _slow_payload_cache = {"system": system, "robot_base": robot_base, "logs": logs}
        _last_slow_update = now

    system = _slow_payload_cache["system"]
    robot = _slow_payload_cache["robot_base"].copy()
    logs = _slow_payload_cache["logs"]
    uptime = _get_uptime_seconds()"""

old_build_func = """def _build_status_payload() -> dict:
    \"\"\"build the complete status payload for websocket push.

    returns:
        dict combining system stats, robot state, logs, and uptime.
    \"\"\"
    system = _get_system_stats()
    robot = _get_robot_state()
    logs = get_log_history(30)
    uptime = _get_uptime_seconds()

    # if brain has get_status(), use that for more complete data
    if _brain is not None and hasattr(_brain, "get_status"):
        try:
            brain_status = _brain.get_status()
            robot.update({
                "state": brain_status.get("state", robot["state"]),
                "emotion": brain_status.get("emotion", robot["emotion"]),
                "emotion_intensity": brain_status.get("emotion_intensity", 0.5),
                "emotion_emoji": brain_status.get("emotion_emoji", "🤖"),
                "thinking": brain_status.get("thinking", robot["thinking"]),
                "grip_state": brain_status.get("grip_state", "unknown"),
                "state_context": brain_status.get("state_context", {}),
            })
            system.update({
                "cpu_percent": brain_status.get("cpu_percent", system["cpu_percent"]),
                "ram_percent": brain_status.get("ram_percent", system["ram_percent"]),
                "ram_used_mb": brain_status.get("ram_used_mb", system["ram_used_mb"]),
                "ram_total_mb": brain_status.get("ram_total_mb", system["ram_total_mb"]),
                "temperature": brain_status.get("cpu_temp", system["temperature"]),
            })
        except Exception as exc:
            logger.debug(f"brain get_status error: {exc}")"""

if old_build_func in content:
    content = content.replace(old_build_func, slow_cache_code)
else:
    print("WARNING: Could not find old _build_status_payload function")

# 3. Optimize the state hashing
old_hash_code = """            import json
            state_to_hash = {
                "system": payload.get("system"),
                "robot": payload.get("robot"),
                "logs_len": len(payload.get("logs", []))
            }
            try:
                current_hash = hash(json.dumps(state_to_hash, sort_keys=True))
            except Exception:
                current_hash = None"""

new_hash_code = """            try:
                # Optimized hashing for 60Hz telemetry: avoid json.dumps
                # We use str() on isolated high-freq dicts and combine hashes
                r = payload.get("robot", {})
                fast_str = str(r.get("servo_angles")) + str(r.get("motor_state")) + str(r.get("imu_data"))
                slow_hash = _last_slow_update # Changes every 1s
                current_hash = hash(fast_str) ^ hash(slow_hash)
            except Exception:
                current_hash = None"""

if old_hash_code in content:
    content = content.replace(old_hash_code, new_hash_code)
else:
    print("WARNING: Could not find old hash code")

with open("simba/web/server.py", "w") as f:
    f.write(content)

print("Patch applied successfully.")
