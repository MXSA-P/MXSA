# _max_cyan_ — project_mxsa
"""simba main entry point — boots the robot on startup.

usage:
    python -m simba.main
    python -m simba.main --no-web
    python -m simba.main --web-only
"""

from simba.utils.logger import get_logger, log_event
import os
import sys
import signal
import argparse
import yaml
import threading

# add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


logger = get_logger("simba.main")

# global brain reference for signal handler
_brain = None


_shutdown_requested = False

def _signal_handler(sig, frame):
    """handle ctrl+c for clean shutdown."""
    global _shutdown_requested
    print("\n[!] Ctrl+C detected. Initiating clean shutdown sequence...")
    logger.critical("Received shutdown signal (Ctrl+C).")
    _shutdown_requested = True


def load_config():
    """load configuration from yaml file."""
    config_path = os.path.join(_PROJECT_ROOT, "config", "simba_config.yaml")
    if not os.path.isfile(config_path):
        logger.error(f"config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("configuration loaded")
    return config


def main():
    """main entry point for simba robot."""
    global _brain

    parser = argparse.ArgumentParser(description="simba bionic arm robot")
    parser.add_argument("--no-web", action="store_true",
                        help="start without web dashboard")
    parser.add_argument("--web-only", action="store_true",
                        help="start web dashboard only (no hardware)")
    parser.add_argument("--port", type=int, default=None,
                        help="web dashboard port (default: from config)")
    args = parser.parse_args()

    # register signal handler
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    banner = """
    ███████╗██╗███╗   ███╗██████╗  █████╗
    ██╔════╝██║████╗ ████║██╔══██╗██╔══██╗
    ███████╗██║██╔████╔██║██████╔╝███████║
    ╚════██║██║██║╚██╔╝██║██╔══██╗██╔══██║
    ███████║██║██║ ╚═╝ ██║██████╔╝██║  ██║
    ╚══════╝╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝

    🦁 SIMBA BIONIC ARM ROBOT
    _max_cyan_ — project_mxsa
    """
    print(banner)
    logger.info("Boot sequence initiated.")

    # load config
    logger.info("Loading configuration...")
    config = load_config()

    # initialize brain
    if not args.web_only:
        logger.info("Initializing hardware subsystems (Brain)...")
        try:
            from simba.core.brain import SimbaBrain
            _brain = SimbaBrain(config)
        except Exception as e:
            logger.critical(f"Failed to initialize brain: {e}")
            sys.exit(1)

    # start web dashboard
    web_server = None
    if not args.no_web:
        try:
            logger.info("Starting web dashboard...")
            from simba.web.server import create_app
            web_cfg = config.get("web", {})
            host = web_cfg.get("host", "0.0.0.0")
            port = args.port or web_cfg.get("port", 8080)

            web_server, socketio = create_app(_brain)
            if _brain:
                _brain.web_server = web_server

            # run web server in background thread
            web_thread = threading.Thread(
                target=lambda: socketio.run(
                    web_server, host=host, port=port,
                    debug=False, allow_unsafe_werkzeug=True,
                    use_reloader=False,
                ),
                daemon=True, name="web-server",
            )
            web_thread.start()

            logger.info(
                f"Web dashboard successfully started: http://{host}:{port}")
            log_event("system", f"web dashboard started on port {port}")
            print(f"  📊 dashboard: http://localhost:{port}")
            print()
        except Exception as e:
            logger.error(f"web dashboard failed: {e}")
            import traceback
            traceback.print_exc()

    if not args.web_only:
        # start the brain (camera, voice, main loop)
        logger.info("Starting main brain loop...")
        try:
            _brain.start()
            print("  🦁 simba is alive and listening!")
            print("  speak commands or use the web dashboard.")
            print("  press ctrl+c to shutdown.")
            print()
        except Exception as e:
            logger.critical(f"Brain failed to start: {e}")
            sys.exit(1)
    else:
        logger.info("Web-only mode active.")
        print("  🌐 web-only mode — no hardware control")
        print("  press ctrl+c to shutdown.")

    # consolidated main loop
    try:
        while not _shutdown_requested:
            if hasattr(signal, "pause"):
                signal.pause()
            else:
                threading.Event().wait(1)
    except KeyboardInterrupt:
        pass

    if _brain is not None:
        _brain.stop()
        print("[!] Shutdown complete. Goodbye.")


if __name__ == "__main__":
    main()
