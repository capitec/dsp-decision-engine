"""Just the polars side of the handshake, for the n=1 breakdown."""
import ctypes, sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arrow-strings-in-tree"))
import arrowc
P = ctypes.POINTER(arrowc.ArrowArrayStream)
schema = arrowc.ArrowSchema(); arr = arrowc.ArrowArray(); arr2 = arrowc.ArrowArray()
def walk_stream(df):
    cap = df.__arrow_c_stream__()
    stream = ctypes.cast(arrowc._PyCapsule_GetPointer(cap, b"arrow_array_stream"), P).contents
    stream.get_schema(ctypes.byref(stream), ctypes.byref(schema))
    stream.get_next(ctypes.byref(stream), ctypes.byref(arr))
    stream.get_next(ctypes.byref(stream), ctypes.byref(arr2))
    arr.release(ctypes.byref(arr)); schema.release(ctypes.byref(schema)); stream.release(ctypes.byref(stream))
