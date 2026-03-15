import sys, traceback
print("step 1: basic imports")
import base64, io, json, logging, math, time, threading, os
from pathlib import Path

print("step 2: ASYNC_MODE = threading")
ASYNC_MODE = "threading"

print("step 3: numpy/PIL")
import numpy as np
from PIL import Image as PILImage

print("step 4: flask")
from flask import Flask, request, jsonify
from flask_cors import CORS

print("step 5: flask_socketio")
from flask_socketio import SocketIO, emit

print("step 6: simd_engine import")
NATIVE_ENGINE = False
_eng = None
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import simd_engine as _eng
    NATIVE_ENGINE = True
    print("  simd_engine: OK, version =", _eng.__version__)
except Exception as e:
    print("  simd_engine: FAIL ->", e)
    traceback.print_exc()

print("step 7: Flask app init, static_folder='.'")
try:
    app = Flask(__name__, static_folder=".", static_url_path="")
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    print("  Flask app: OK")
except Exception as e:
    print("  Flask app: FAIL ->", e)
    traceback.print_exc()

print("step 8: SocketIO init")
try:
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode=ASYNC_MODE,
        max_http_buffer_size=128 * 1024 * 1024,
        ping_timeout=60,
        ping_interval=25,
        logger=False,
        engineio_logger=False,
    )
    print("  SocketIO: OK")
except Exception as e:
    print("  SocketIO: FAIL ->", e)
    traceback.print_exc()

print("step 9: socketio.run (starting server)")
try:
    socketio.run(app, host="127.0.0.1", port=5000, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
except Exception as e:
    print("  socketio.run: FAIL ->", e)
    traceback.print_exc()
