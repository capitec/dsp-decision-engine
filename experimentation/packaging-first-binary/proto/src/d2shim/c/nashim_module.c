/* nashim_module.c -- the smallest possible CPython module around nashim.c.
 *
 * nashim.c is a plain C library with no Python in it, and the numba kernel
 * calls its functions through raw addresses. This file exists only so that
 * the shared object is a real extension module: pip, auditwheel, delocate,
 * cibuildwheel and MSVC all know exactly what to do with one, and on Windows
 * the PyInit symbol is the only export the linker needs (no __declspec, no
 * .def file, no changes to nashim.c).
 *
 * It exposes each sm_* function's address as a Python int. The Python side
 * turns those into ctypes callables and hands them to numba as arguments.
 *
 * Built against the stable ABI (Py_LIMITED_API=3.10) so ONE wheel per
 * platform covers every CPython >= 3.10. It uses four limited-API calls.
 */
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030A0000
#endif
#include <Python.h>
#include <stddef.h>
#include <stdint.h>

/* nashim.c has no header; these are its exported signatures. Keeping them
 * here means nashim.c stays byte-identical to the experiment's copy. */
struct ArrowArrayView; struct ArrowSchema; struct ArrowArray; struct ArrowError; struct ArrowArrayStream;
size_t sm_sizeof_array_view(void);
size_t sm_sizeof_schema(void);
size_t sm_sizeof_array(void);
size_t sm_sizeof_error(void);
const char* sm_nanoarrow_version(void);
int sm_import_single(struct ArrowArrayStream*, struct ArrowSchema*, struct ArrowArray*,
                     struct ArrowArray*, struct ArrowArrayView*, struct ArrowError*);
int sm_view_set(struct ArrowArrayView*, const struct ArrowSchema*, const struct ArrowArray*,
                struct ArrowError*);
int sm_view_validate(struct ArrowArrayView*, int, struct ArrowError*);
int sm_view_storage_type(const struct ArrowArrayView*);
int64_t sm_view_length(const struct ArrowArrayView*);
int64_t sm_view_offset(const struct ArrowArrayView*);
void sm_view_reset(struct ArrowArrayView*);
void sm_array_release(struct ArrowArray*);
void sm_schema_release(struct ArrowSchema*);
const char* sm_error_message(const struct ArrowError*);
int64_t sm_get_string(const struct ArrowArrayView*, int64_t, const uint8_t**);
int64_t sm_get_string_checked(const struct ArrowArrayView*, int64_t, const uint8_t**);

/* Bump when the set or signatures of exported functions change; the Python
 * side refuses a module whose ABI it does not know. */
#define NASHIM_ABI 1

#define ADD_FN(m, fn)                                                         \
  do {                                                                        \
    PyObject* v = PyLong_FromVoidPtr((void*)(fn));                            \
    if (v == NULL || PyModule_AddObjectRef((m), #fn, v) < 0) {                \
      Py_XDECREF(v); Py_DECREF(m); return NULL;                               \
    }                                                                         \
    Py_DECREF(v);                                                             \
  } while (0)

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_nashim",
    "Addresses of the nanoarrow string shim (nashim.c), built into this module.",
    -1, NULL, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__nashim(void) {
  PyObject* m = PyModule_Create(&moduledef);
  if (m == NULL) return NULL;
  if (PyModule_AddIntConstant(m, "ABI", NASHIM_ABI) < 0) { Py_DECREF(m); return NULL; }
  if (PyModule_AddStringConstant(m, "NANOARROW_VERSION", sm_nanoarrow_version()) < 0) {
    Py_DECREF(m); return NULL;
  }
  ADD_FN(m, sm_sizeof_array_view);
  ADD_FN(m, sm_sizeof_schema);
  ADD_FN(m, sm_sizeof_array);
  ADD_FN(m, sm_sizeof_error);
  ADD_FN(m, sm_import_single);
  ADD_FN(m, sm_view_set);
  ADD_FN(m, sm_view_validate);
  ADD_FN(m, sm_view_storage_type);
  ADD_FN(m, sm_view_length);
  ADD_FN(m, sm_view_offset);
  ADD_FN(m, sm_view_reset);
  ADD_FN(m, sm_array_release);
  ADD_FN(m, sm_schema_release);
  ADD_FN(m, sm_error_message);
  ADD_FN(m, sm_get_string);
  ADD_FN(m, sm_get_string_checked);
  return m;
}
