/* nashim.c -- the thin C shim between numba and vendored nanoarrow.
 *
 * Every function here is a wrapper around a nanoarrow call; the ONLY
 * Arrow-layout knowledge in this file is inside sm_get_string_checked(),
 * which is marked and counted in RESULTS.md. Everything else is plumbing.
 *
 * Built by build.sh into libnashim.so together with ../vendor/nanoarrow.c.
 */
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"

/* ---- sizes, so Python can allocate the structs opaquely ---------------- */
size_t sm_sizeof_array_view(void) { return sizeof(struct ArrowArrayView); }
size_t sm_sizeof_schema(void) { return sizeof(struct ArrowSchema); }
size_t sm_sizeof_array(void) { return sizeof(struct ArrowArray); }
size_t sm_sizeof_error(void) { return sizeof(struct ArrowError); }
const char* sm_nanoarrow_version(void) { return NANOARROW_VERSION; }

/* ---- the whole polars -> nanoarrow handshake in ONE call ---------------
 * Pull the schema and the first chunk from an ArrowArrayStream, confirm the
 * stream has exactly one chunk, init the view from the schema and set the
 * array (SetArrayMinimal). Returns 0 on success, 1 if the stream has more
 * than one chunk (the second is left in `array2`; the caller keeps pulling
 * and builds a view per chunk with sm_view_set), or a negative code with
 * the message in `err`.
 */
int sm_import_single(struct ArrowArrayStream* stream, struct ArrowSchema* schema,
                     struct ArrowArray* array, struct ArrowArray* array2,
                     struct ArrowArrayView* view, struct ArrowError* err) {
  int rc = stream->get_schema(stream, schema);
  if (rc != 0) {
    ArrowErrorSet(err, "get_schema: %s", stream->get_last_error(stream));
    return -1;
  }
  rc = stream->get_next(stream, array);
  if (rc != 0) {
    ArrowErrorSet(err, "get_next: %s", stream->get_last_error(stream));
    return -2;
  }
  if (array->release == NULL) {
    ArrowErrorSet(err, "stream had no chunks");
    return -3;
  }
  rc = stream->get_next(stream, array2);
  if (rc != 0) {
    ArrowErrorSet(err, "get_next(2): %s", stream->get_last_error(stream));
    return -4;
  }
  if (array2->release != NULL) {
    return 1; /* more than one chunk: array2 holds the second, caller keeps pulling */
  }
  rc = ArrowArrayViewInitFromSchema(view, schema, err);
  if (rc != NANOARROW_OK) return -5;
  rc = ArrowArrayViewSetArrayMinimal(view, array, err);
  if (rc != NANOARROW_OK) return -6;
  return 0;
}

/* Per-chunk variant for the multi-chunk fallback: schema already read. */
int sm_view_set(struct ArrowArrayView* view, const struct ArrowSchema* schema,
                const struct ArrowArray* array, struct ArrowError* err) {
  int rc = ArrowArrayViewInitFromSchema(view, schema, err);
  if (rc != NANOARROW_OK) return -5;
  rc = ArrowArrayViewSetArrayMinimal(view, array, err);
  if (rc != NANOARROW_OK) return -6;
  return 0;
}

/* Validate at a nanoarrow level: 1 minimal, 2 default, 3 full. */
int sm_view_validate(struct ArrowArrayView* view, int level, struct ArrowError* err) {
  return ArrowArrayViewValidate(view, (enum ArrowValidationLevel)level, err);
}

int sm_view_storage_type(const struct ArrowArrayView* view) { return (int)view->storage_type; }
int64_t sm_view_length(const struct ArrowArrayView* view) { return view->length; }
int64_t sm_view_offset(const struct ArrowArrayView* view) { return view->offset; }

void sm_view_reset(struct ArrowArrayView* view) { ArrowArrayViewReset(view); }
void sm_array_release(struct ArrowArray* array) {
  if (array->release != NULL) ArrowArrayRelease(array);
}
void sm_schema_release(struct ArrowSchema* schema) {
  if (schema->release != NULL) ArrowSchemaRelease(schema);
}
const char* sm_error_message(const struct ArrowError* err) { return err->message; }

/* ---- the per-row accessor the kernel calls ------------------------------
 * Returns the string length, or -1 for a null. Writes the byte pointer to
 * *data. This is nanoarrow's own accessor: it handles utf8 ("u"),
 * large_utf8 ("U") and utf8_view ("vu") behind one call. It is the
 * *Unsafe* accessor: nanoarrow does no bounds checking here.
 */
int64_t sm_get_string(const struct ArrowArrayView* view, int64_t i, const uint8_t** data) {
  if (ArrowArrayViewIsNull(view, i)) {
    *data = NULL;
    return -1;
  }
  struct ArrowStringView sv = ArrowArrayViewGetStringUnsafe(view, i);
  *data = (const uint8_t*)sv.data;
  return sv.size_bytes;
}

/* ---- a bounds-checked variant --------------------------------------------
 * nanoarrow ships no checked string accessor, so this is hand-written. It
 * uses nanoarrow's fields rather than raw buffers, but every line between
 * the BEGIN/END markers is Arrow layout knowledge the project would own.
 * Returns -2 when the element points outside its buffers.
 */
int64_t sm_get_string_checked(const struct ArrowArrayView* view, int64_t i,
                              const uint8_t** data) {
  if (i < 0 || i >= view->length) { *data = NULL; return -2; }
  if (ArrowArrayViewIsNull(view, i)) { *data = NULL; return -1; }
  /* BEGIN layout knowledge */
  int64_t j = i + view->offset;
  switch (view->storage_type) {
    case NANOARROW_TYPE_STRING_VIEW: {
      const union ArrowBinaryView* bv = &view->buffer_views[1].data.as_binary_view[j];
      int32_t n = bv->inlined.size;
      if (n < 0) { *data = NULL; return -2; }
      if (n <= NANOARROW_BINARY_VIEW_INLINE_SIZE) {
        *data = bv->inlined.data;
        return n;
      }
      if (bv->ref.buffer_index < 0 || bv->ref.buffer_index >= view->n_variadic_buffers) {
        *data = NULL; return -2;
      }
      int64_t size = view->variadic_buffer_sizes[bv->ref.buffer_index];
      if (bv->ref.offset < 0 || (int64_t)bv->ref.offset + n > size) { *data = NULL; return -2; }
      *data = (const uint8_t*)view->variadic_buffers[bv->ref.buffer_index] + bv->ref.offset;
      return n;
    }
    case NANOARROW_TYPE_STRING: {
      const int32_t* off = view->buffer_views[1].data.as_int32;
      int64_t size = view->buffer_views[2].size_bytes;
      if (off[j] < 0 || off[j + 1] < off[j] || off[j + 1] > size) { *data = NULL; return -2; }
      *data = view->buffer_views[2].data.as_uint8 + off[j];
      return off[j + 1] - off[j];
    }
    case NANOARROW_TYPE_LARGE_STRING: {
      const int64_t* off = view->buffer_views[1].data.as_int64;
      int64_t size = view->buffer_views[2].size_bytes;
      if (off[j] < 0 || off[j + 1] < off[j] || off[j + 1] > size) { *data = NULL; return -2; }
      *data = view->buffer_views[2].data.as_uint8 + off[j];
      return off[j + 1] - off[j];
    }
    default:
      *data = NULL; return -2;
  }
  /* END layout knowledge */
}
