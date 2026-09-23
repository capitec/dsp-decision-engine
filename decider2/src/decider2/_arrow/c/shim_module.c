/* shim_module.c -- the smallest possible CPython module around shim.c.
 *
 * shim.c is a plain C library with no Python in it, and the numba kernel
 * calls its functions through raw addresses. This file exists only so that
 * the shared object is a real extension module: pip, auditwheel, delocate,
 * cibuildwheel and MSVC all know exactly what to do with one, and on Windows
 * the PyInit symbol is the only export the linker needs (no __declspec, no
 * .def file, no changes to shim.c).
 *
 * It exposes each sm_* function's address as a Python int. The Python side
 * (decider2/_arrow/_shim.py) turns those into ctypes callables and hands
 * them to numba as ARGUMENTS.
 *
 * Built against the stable ABI (Py_LIMITED_API = 3.10) so ONE wheel per
 * platform covers every CPython >= 3.10. Four limited-API calls, all in
 * python3.dll on Windows. Adopted from experimentation/packaging-first-binary/
 * proto/src/d2shim/c/nashim_module.c (RESULTS.md §2).
 */
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030A0000
#endif
#include <Python.h>
#include <stddef.h>
#include <stdint.h>

/* shim.c has no header; these are its exported signatures. */
struct ArrowArrayView; struct ArrowSchema; struct ArrowArray; struct ArrowError;
struct ArrowArrayStream; struct SmColDesc; struct SmRowPlan;
size_t sm_sizeof_array_view(void);
size_t sm_sizeof_schema(void);
size_t sm_sizeof_array(void);
size_t sm_sizeof_error(void);
size_t sm_sizeof_coldesc(void);
size_t sm_sizeof_plan(void);
const char* sm_nanoarrow_version(void);
const char* sm_error_message(const struct ArrowError*);
int sm_import_frame(struct ArrowArrayStream*, struct ArrowSchema*, struct ArrowArray*,
                    struct ArrowArray*, struct ArrowArrayView*, struct ArrowError*);
int sm_stream_get_next(struct ArrowArrayStream*, struct ArrowArray*, struct ArrowError*);
int sm_array_is_valid(const struct ArrowArray*);
int sm_view_set(struct ArrowArrayView*, const struct ArrowSchema*, const struct ArrowArray*,
                struct ArrowError*);
int sm_view_validate(struct ArrowArrayView*, int, struct ArrowError*);
int sm_view_storage_type(const struct ArrowArrayView*);
int64_t sm_view_length(const struct ArrowArrayView*);
int64_t sm_view_offset(const struct ArrowArrayView*);
int64_t sm_view_null_count(const struct ArrowArrayView*);
int64_t sm_view_n_children(const struct ArrowArrayView*);
const struct ArrowArrayView* sm_view_child(const struct ArrowArrayView*, int64_t);
const struct ArrowArrayView* sm_view_dictionary(const struct ArrowArrayView*);
int sm_view_n_variadic_buffers(const struct ArrowArrayView*);
int sm_view_has_validity(const struct ArrowArrayView*);
void sm_view_reset(struct ArrowArrayView*);
void sm_array_release(struct ArrowArray*);
void sm_schema_release(struct ArrowSchema*);
int64_t sm_schema_n_children(const struct ArrowSchema*);
const char* sm_schema_format(const struct ArrowSchema*);
const char* sm_schema_child_name(const struct ArrowSchema*, int64_t);
const char* sm_schema_child_format(const struct ArrowSchema*, int64_t);
int64_t sm_schema_child_to_string(const struct ArrowSchema*, int64_t, char*, int64_t);
int sm_is_null(const struct ArrowArrayView*, int64_t);
double sm_get_f64(const struct ArrowArrayView*, int64_t);
int64_t sm_get_i64(const struct ArrowArrayView*, int64_t);
int64_t sm_get_string(const struct ArrowArrayView*, int64_t, const uint8_t**);
int sm_resolve_col(const struct ArrowArrayView*, int32_t, int32_t, double, int64_t,
                   struct SmColDesc*);
int sm_resolve_all(const struct ArrowArrayView*, const int32_t*, const int32_t*, const int32_t*,
                   const double*, const int64_t*, int32_t, struct SmRowPlan*, uint64_t*);
void sm_gather_row(const struct SmRowPlan*, int64_t);

/* Bump when the set or signatures of exported functions, or the struct
 * layouts, change; _shim.py refuses a module whose ABI it does not know. */
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
    "Addresses of decider2's nanoarrow shim (shim.c), built into this module.",
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
  ADD_FN(m, sm_sizeof_coldesc);
  ADD_FN(m, sm_sizeof_plan);
  ADD_FN(m, sm_error_message);
  ADD_FN(m, sm_import_frame);
  ADD_FN(m, sm_stream_get_next);
  ADD_FN(m, sm_array_is_valid);
  ADD_FN(m, sm_view_set);
  ADD_FN(m, sm_view_validate);
  ADD_FN(m, sm_view_storage_type);
  ADD_FN(m, sm_view_length);
  ADD_FN(m, sm_view_offset);
  ADD_FN(m, sm_view_null_count);
  ADD_FN(m, sm_view_n_children);
  ADD_FN(m, sm_view_child);
  ADD_FN(m, sm_view_dictionary);
  ADD_FN(m, sm_view_n_variadic_buffers);
  ADD_FN(m, sm_view_has_validity);
  ADD_FN(m, sm_view_reset);
  ADD_FN(m, sm_array_release);
  ADD_FN(m, sm_schema_release);
  ADD_FN(m, sm_schema_n_children);
  ADD_FN(m, sm_schema_format);
  ADD_FN(m, sm_schema_child_name);
  ADD_FN(m, sm_schema_child_format);
  ADD_FN(m, sm_schema_child_to_string);
  ADD_FN(m, sm_is_null);
  ADD_FN(m, sm_get_f64);
  ADD_FN(m, sm_get_i64);
  ADD_FN(m, sm_get_string);
  ADD_FN(m, sm_resolve_col);
  ADD_FN(m, sm_resolve_all);
  ADD_FN(m, sm_gather_row);
  return m;
}
