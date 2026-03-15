import sys, traceback
print("Python:", sys.version)

mods = ["flask", "flask_cors", "flask_socketio", "numpy", "PIL", "scipy", "simd_engine"]
for m in mods:
    try:
        __import__(m)
        print("OK:", m)
    except Exception as e:
        print("FAIL:", m, "->", e)
        traceback.print_exc()
