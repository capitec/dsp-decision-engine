/* shim.c -- the C shim between numba kernels and vendored nanoarrow 0.9.0.
 *
 * Every value handed out is a nanoarrow field or accessor result; nothing
 * here decodes an Arrow layout by hand. Plain C99 with no Python, loaded
 * with ctypes: numba kernels call sm_gather_row and sm_get_string through
 * raw addresses passed as arguments, because a pointer argument disk-caches
 * and a captured symbol does not.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"

/* decider.engine.ir.decls.FeatureKind -- program data, never renumbered. */
#define SM_KIND_F64 0
#define SM_KIND_I64 1
#define SM_KIND_BOOL 2
#define SM_KIND_CODE 3
#define SM_KIND_STR 4

/* ---- one column, resolved once per batch --------------------------------
 * Filled by sm_resolve_col from nanoarrow's decoded ArrowArrayView; read by
 * sm_columns for the whole batch, or sm_gather_row one row at a time.
 * `fill_*` is the value written when the validity bit is clear (a MISSING_AS fill rides here); the
 * defaults are NaN / 0 / -1 (CODE).                                     */
struct SmColDesc {
  const uint8_t* validity;            /* buffer_views[0]; NULL when absent */
  const void* data;                   /* buffer_views[1] */
  const struct ArrowArrayView* view;  /* the child view (strings go through the accessor) */
  int64_t offset;                     /* view->offset: a sliced frame carries it on each child */
  double fill_f64;
  int64_t fill_i64;
  int32_t kind;                       /* SM_KIND_* */
  int32_t width;                      /* bytes per element for F64/I64/CODE */
  int32_t slot;                       /* index into the kind's row buffer */
  int32_t is_signed;
};

struct SmRowPlan {
  int32_t ncols;
  int32_t _pad;
  const struct SmColDesc* cols;
  double* f64;
  int64_t* i64;
  uint8_t* b8;
  int32_t* i32;
  int64_t* span;                      /* 2 per STR slot: (address, length); length -1 = null */
  uint8_t* valid;                     /* per column, in plan order */
};

/* ---- sizes, so Python allocates every struct opaquely ------------------ */
size_t sm_sizeof_array_view(void) { return sizeof(struct ArrowArrayView); }
size_t sm_sizeof_schema(void) { return sizeof(struct ArrowSchema); }
size_t sm_sizeof_array(void) { return sizeof(struct ArrowArray); }
size_t sm_sizeof_error(void) { return sizeof(struct ArrowError); }
size_t sm_sizeof_coldesc(void) { return sizeof(struct SmColDesc); }
size_t sm_sizeof_plan(void) { return sizeof(struct SmRowPlan); }
const char* sm_nanoarrow_version(void) { return NANOARROW_VERSION; }
const char* sm_error_message(const struct ArrowError* err) { return err->message; }

/* ---- the whole polars -> nanoarrow handshake in ONE call ----------------
 * Pull the schema and the first chunk from the ArrowArrayStream, init the
 * view from the schema and set the first array (SetArrayMinimal: no buffer
 * walk). Returns 0 for a single-chunk stream, 1 when the stream had a
 * second chunk (left in `array2`, which the caller refuses), or a negative
 * code with the message in `err`.                                        */
int sm_import_frame(struct ArrowArrayStream* stream, struct ArrowSchema* schema,
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
  rc = ArrowArrayViewInitFromSchema(view, schema, err);
  if (rc != NANOARROW_OK) return -5;
  rc = ArrowArrayViewSetArrayMinimal(view, array, err);
  if (rc != NANOARROW_OK) return -6;
  return array2->release != NULL ? 1 : 0;
}

/* ---- view accessors ---------------------------------------------------- */
int sm_view_storage_type(const struct ArrowArrayView* v) { return (int)v->storage_type; }
int64_t sm_view_length(const struct ArrowArrayView* v) { return v->length; }
int64_t sm_view_offset(const struct ArrowArrayView* v) { return v->offset; }
int64_t sm_view_null_count(const struct ArrowArrayView* v) {
  return v->null_count >= 0 ? v->null_count : ArrowArrayViewComputeNullCount(v);
}
const struct ArrowArrayView* sm_view_child(const struct ArrowArrayView* v, int64_t k) {
  return (k >= 0 && k < v->n_children) ? v->children[k] : NULL;
}
const struct ArrowArrayView* sm_view_dictionary(const struct ArrowArrayView* v) {
  return v->dictionary;
}
int sm_view_n_variadic_buffers(const struct ArrowArrayView* v) { return v->n_variadic_buffers; }
int sm_view_has_validity(const struct ArrowArrayView* v) {
  return v->buffer_views[0].data.as_uint8 != NULL;
}

void sm_array_release(struct ArrowArray* array) {
  if (array->release != NULL) ArrowArrayRelease(array);
}
/* Everything sm_import_frame took, in one call. */
void sm_release(struct ArrowSchema* schema, struct ArrowArray* array, struct ArrowArray* array2,
                struct ArrowArrayView* view) {
  ArrowArrayViewReset(view);
  sm_array_release(array);
  sm_array_release(array2);
  if (schema->release != NULL) ArrowSchemaRelease(schema);
}
/* Bitwise move, as the C data interface allows: `dst` owns the array and `src` is released. */
void sm_array_move(struct ArrowArray* src, struct ArrowArray* dst) { ArrowArrayMove(src, dst); }

/* ---- schema accessors (names and types for the plan and for messages) -- */
/* Human-readable type of child k ("string_view", "dictionary(...)", ...). */
int64_t sm_schema_child_to_string(const struct ArrowSchema* s, int64_t k, char* out, int64_t n) {
  if (k < 0 || k >= s->n_children) return -1;
  return ArrowSchemaToString(s->children[k], out, n, 0);
}

/* One string, one call: nanoarrow's own accessor, which handles utf8 ("u"),
 * large_utf8 ("U") and utf8_view ("vu") behind one call. Returns the byte
 * length, or -1 for a null; writes the byte pointer to *data.            */
int64_t sm_get_string(const struct ArrowArrayView* v, int64_t i, const uint8_t** data) {
  if (ArrowArrayViewIsNull(v, i)) {
    *data = NULL;
    return -1;
  }
  struct ArrowStringView sv = ArrowArrayViewGetStringUnsafe(v, i);
  *data = (const uint8_t*)sv.data;
  return sv.size_bytes;
}

/* ---- resolve: one ColDesc per column, once per batch --------------------
 * Which Arrow storage types land in which kind. Returns 0, or -1 when the child's Arrow type does not match the
 * requested kind (the Python side names the column and the Arrow type). */
static int sm_int_width(enum ArrowType t, int32_t* width, int32_t* is_signed) {
  switch (t) {
    case NANOARROW_TYPE_INT64: case NANOARROW_TYPE_TIMESTAMP: case NANOARROW_TYPE_DURATION:
    case NANOARROW_TYPE_DATE64: case NANOARROW_TYPE_TIME64:
      *width = 8; *is_signed = 1; return 1;
    case NANOARROW_TYPE_INT32: case NANOARROW_TYPE_DATE32: case NANOARROW_TYPE_TIME32:
      *width = 4; *is_signed = 1; return 1;
    case NANOARROW_TYPE_INT16: *width = 2; *is_signed = 1; return 1;
    case NANOARROW_TYPE_INT8:  *width = 1; *is_signed = 1; return 1;
    case NANOARROW_TYPE_UINT64: *width = 8; *is_signed = 0; return 1;
    case NANOARROW_TYPE_UINT32: *width = 4; *is_signed = 0; return 1;
    case NANOARROW_TYPE_UINT16: *width = 2; *is_signed = 0; return 1;
    case NANOARROW_TYPE_UINT8:  *width = 1; *is_signed = 0; return 1;
    default: return 0;
  }
}

static int sm_resolve_col(const struct ArrowArrayView* v, int32_t kind, int32_t slot,
                   double fill_f64, int64_t fill_i64, struct SmColDesc* d) {
  d->view = v;
  d->offset = v->offset;
  d->kind = kind;
  d->slot = slot;
  d->validity = v->buffer_views[0].data.as_uint8;  /* nanoarrow leaves NULL when absent */
  d->data = v->buffer_views[1].data.data;
  d->fill_f64 = fill_f64;
  d->fill_i64 = fill_i64;
  d->width = 0;
  d->is_signed = 1;
  int encoded = v->dictionary != NULL;  /* Categorical/Enum: storage_type is the INDEX type */
  switch (kind) {
    case SM_KIND_F64:
      if (encoded) return -1;
      if (v->storage_type == NANOARROW_TYPE_DOUBLE) { d->width = 8; return 0; }
      if (v->storage_type == NANOARROW_TYPE_FLOAT) { d->width = 4; return 0; }
      return -1;
    case SM_KIND_I64:
      if (encoded) return -1;
      return sm_int_width(v->storage_type, &d->width, &d->is_signed) ? 0 : -1;
    case SM_KIND_CODE:
      /* a dictionary index, or an already-encoded integer column */
      return sm_int_width(v->storage_type, &d->width, &d->is_signed) ? 0 : -1;
    case SM_KIND_BOOL:
      if (encoded) return -1;
      return v->storage_type == NANOARROW_TYPE_BOOL ? 0 : -1;
    case SM_KIND_STR:
      if (encoded) return -1;  /* a dictionary column is read as CODE */
      switch (v->storage_type) {
        case NANOARROW_TYPE_STRING: case NANOARROW_TYPE_LARGE_STRING: case NANOARROW_TYPE_STRING_VIEW:
          return 0;
        default:
          return -1;
      }
    default:
      return -1;
  }
}

/* One call per batch: resolve every declared column of the frame view into
 * the plan's ColDesc array, and (when `addrs` is non-NULL) the flat address
 * table a numba kernel can load from without a per-row C call:
 *   addrs[c*4+0] = data address, already advanced by offset*width
 *                  (bit-indexed BOOL data is NOT advanced: index offset+i)
 *   addrs[c*4+1] = validity bitmap address, 0 when absent (index offset+i)
 *   addrs[c*4+2] = offset
 *   addrs[c*4+3] = the child view address (for sm_get_string)
 * Returns 0, or -(c+1) for the first column whose Arrow type does not match
 * its declared kind.                                                       */
int sm_resolve_all(const struct ArrowArrayView* frame, const int32_t* kinds, const int32_t* slots,
                   const int32_t* child_idx, const double* fill_f64, const int64_t* fill_i64,
                   int32_t ncols, struct SmRowPlan* plan, uint64_t* addrs) {
  struct SmColDesc* cols = (struct SmColDesc*)plan->cols;
  for (int32_t c = 0; c < ncols; c++) {
    if (child_idx[c] < 0 || child_idx[c] >= frame->n_children) return -(c + 1);
    const struct ArrowArrayView* v = frame->children[child_idx[c]];
    double ff = fill_f64 ? fill_f64[c] : NAN;
    int64_t fi = fill_i64 ? fill_i64[c] : (kinds[c] == SM_KIND_CODE ? -1 : 0);
    if (sm_resolve_col(v, kinds[c], slots[c], ff, fi, &cols[c]) != 0) return -(c + 1);
    if (addrs) {
      const struct SmColDesc* d = &cols[c];
      uint64_t adv = (d->kind == SM_KIND_BOOL || d->kind == SM_KIND_STR)
                         ? 0 : (uint64_t)d->offset * (uint64_t)d->width;
      addrs[c * 4 + 0] = (uint64_t)(uintptr_t)d->data + adv;
      addrs[c * 4 + 1] = (uint64_t)(uintptr_t)d->validity;
      addrs[c * 4 + 2] = (uint64_t)d->offset;
      addrs[c * 4 + 3] = (uint64_t)(uintptr_t)d->view;
    }
  }
  plan->ncols = ncols;
  return 0;
}

/* ---- the per-row gather: ONE C call per row ----------------------------
 * `data[offset + i]` per column, the dispatch on width resolved per column
 * per batch above. A null never reads its value: the slot gets the fill.  */
static inline int64_t sm_load_int(const void* data, int32_t width, int32_t is_signed, int64_t j) {
  switch (width) {
    case 8: return is_signed ? ((const int64_t*)data)[j] : (int64_t)((const uint64_t*)data)[j];
    case 4: return is_signed ? ((const int32_t*)data)[j] : (int64_t)((const uint32_t*)data)[j];
    case 2: return is_signed ? ((const int16_t*)data)[j] : (int64_t)((const uint16_t*)data)[j];
    default: return is_signed ? ((const int8_t*)data)[j] : (int64_t)((const uint8_t*)data)[j];
  }
}

void sm_gather_row(const struct SmRowPlan* p, int64_t i) {
  for (int32_t c = 0; c < p->ncols; c++) {
    const struct SmColDesc* d = &p->cols[c];
    int64_t j = d->offset + i;
    int isnull = d->validity != NULL && !ArrowBitGet(d->validity, j);
    p->valid[c] = (uint8_t)!isnull;
    switch (d->kind) {
      case SM_KIND_F64:
        p->f64[d->slot] = isnull ? d->fill_f64
                          : (d->width == 8 ? ((const double*)d->data)[j]
                                           : (double)((const float*)d->data)[j]);
        break;
      case SM_KIND_I64:
        p->i64[d->slot] = isnull ? d->fill_i64 : sm_load_int(d->data, d->width, d->is_signed, j);
        break;
      case SM_KIND_BOOL:
        p->b8[d->slot] = isnull ? (uint8_t)(d->fill_i64 != 0)
                                : (uint8_t)ArrowBitGet((const uint8_t*)d->data, j);
        break;
      case SM_KIND_CODE:
        p->i32[d->slot] = isnull ? (int32_t)d->fill_i64
                                 : (int32_t)sm_load_int(d->data, d->width, d->is_signed, j);
        break;
      case SM_KIND_STR:
        if (isnull) {
          p->span[2 * d->slot] = 0;
          p->span[2 * d->slot + 1] = -1;
        } else {
          struct ArrowStringView sv = ArrowArrayViewGetStringUnsafe(d->view, i);
          p->span[2 * d->slot] = (int64_t)(intptr_t)sv.data;
          p->span[2 * d->slot + 1] = sv.size_bytes;
        }
        break;
      default:
        break;
    }
  }
}

/* ---- the whole batch, one column at a time: ONE C call per batch --------
 * `out` is one buffer: an int64 header of three rows of `ncols` (null
 * count, byte offset of the column's values in `out`, in-place address),
 * then the F64, I64, STR, CODE and BOOL columns (`n` values each; a STR
 * value is an (address, length) pair of int64s), then one validity row of
 * `n` bytes per column. A null takes the fill, the same values
 * sm_gather_row writes row by row; only a column with nulls gets its
 * validity row. With `borrow` set, a column the kernel can read in place
 * (doubles as F64, int64 as I64, int32 as CODE, no nulls) is not copied:
 * its in-place address is set instead (0 for a copied column).          */
void sm_columns(const struct SmRowPlan* p, int64_t n, uint8_t* buf, int32_t borrow) {
  static const int64_t size[5] = {8, 8, 1, 4, 16};  /* by SM_KIND_* */
  static const int order[5] = {SM_KIND_F64, SM_KIND_I64, SM_KIND_STR, SM_KIND_CODE, SM_KIND_BOOL};
  int32_t ncols = p->ncols;
  int64_t count[5] = {0, 0, 0, 0, 0}, start[5];
  for (int32_t c = 0; c < ncols; c++) count[p->cols[c].kind]++;
  int64_t at = 24 * (int64_t)ncols;
  for (int k = 0; k < 5; k++) {
    start[order[k]] = at;
    at += count[order[k]] * n * size[order[k]];
  }
  uint8_t* valid = buf + at;
  int64_t* nulls = (int64_t*)buf;
  int64_t* where = nulls + ncols;
  uint64_t* in_place = (uint64_t*)(where + ncols);
  for (int32_t c = 0; c < ncols; c++) {
    const struct SmColDesc* d = &p->cols[c];
    int64_t nn = d->validity == NULL ? 0 : sm_view_null_count(d->view);
    const uint8_t* bits = nn ? d->validity : NULL;
    int64_t o = d->offset;
    nulls[c] = nn;
    where[c] = start[d->kind] + (int64_t)d->slot * n * size[d->kind];
    in_place[c] = 0;
    void* dst = buf + where[c];
    if (borrow && nn == 0 && d->is_signed &&
        ((d->kind == SM_KIND_F64 && d->width == 8) || (d->kind == SM_KIND_I64 && d->width == 8) ||
         (d->kind == SM_KIND_CODE && d->width == 4))) {
      in_place[c] = (uint64_t)(uintptr_t)((const uint8_t*)d->data + o * d->width);
      continue;
    }
    if (bits != NULL) {
      uint8_t* vc = valid + (int64_t)c * n;
      for (int64_t i = 0; i < n; i++) vc[i] = (uint8_t)ArrowBitGet(bits, o + i);
    }
#define SM_NULL(i) (bits != NULL && !ArrowBitGet(bits, o + (i)))
    switch (d->kind) {
      case SM_KIND_F64: {
        double* out = (double*)dst;
        if (d->width == 8 && bits == NULL) {
          memcpy(out, (const double*)d->data + o, (size_t)n * sizeof(double));
        } else if (d->width == 8) {
          const double* x = (const double*)d->data + o;
          for (int64_t i = 0; i < n; i++) out[i] = SM_NULL(i) ? d->fill_f64 : x[i];
        } else {
          const float* x = (const float*)d->data + o;
          for (int64_t i = 0; i < n; i++) out[i] = SM_NULL(i) ? d->fill_f64 : (double)x[i];
        }
        break;
      }
      case SM_KIND_I64: {
        int64_t* out = (int64_t*)dst;
        if (bits == NULL && d->width == 8 && d->is_signed)
          memcpy(out, (const int64_t*)d->data + o, (size_t)n * sizeof(int64_t));
        else
          for (int64_t i = 0; i < n; i++)
            out[i] = SM_NULL(i) ? d->fill_i64 : sm_load_int(d->data, d->width, d->is_signed, o + i);
        break;
      }
      case SM_KIND_BOOL: {
        uint8_t* out = (uint8_t*)dst;
        for (int64_t i = 0; i < n; i++)
          out[i] = SM_NULL(i) ? (uint8_t)(d->fill_i64 != 0)
                              : (uint8_t)ArrowBitGet((const uint8_t*)d->data, o + i);
        break;
      }
      case SM_KIND_CODE: {
        int32_t* out = (int32_t*)dst;
        for (int64_t i = 0; i < n; i++)
          out[i] = SM_NULL(i) ? (int32_t)d->fill_i64
                              : (int32_t)sm_load_int(d->data, d->width, d->is_signed, o + i);
        break;
      }
      case SM_KIND_STR: {
        int64_t* out = (int64_t*)dst;
        for (int64_t i = 0; i < n; i++) {
          if (SM_NULL(i)) {
            out[2 * i] = 0;
            out[2 * i + 1] = -1;
          } else {
            struct ArrowStringView sv = ArrowArrayViewGetStringUnsafe(d->view, i);
            out[2 * i] = (int64_t)(intptr_t)sv.data;
            out[2 * i + 1] = sv.size_bytes;
          }
        }
        break;
      }
      default:
        break;
    }
#undef SM_NULL
  }
}
