/* rowshim.c -- throwaway probe for BOUNDARY-REWORK.md.
 *
 * Question: can ONE nanoarrow import of a whole polars DataFrame (the `+s`
 * struct stream) plus ONE C call per row that gathers a typed row through
 * nanoarrow's unsafe accessors replace decider2's per-column numpy
 * extraction and numba `_fill_array` gather?  Zero Arrow-layout knowledge
 * lives here: every element read is a nanoarrow accessor.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "nanoarrow/nanoarrow.h"

size_t rs_sizeof_view(void)   { return sizeof(struct ArrowArrayView); }
size_t rs_sizeof_schema(void) { return sizeof(struct ArrowSchema); }
size_t rs_sizeof_array(void)  { return sizeof(struct ArrowArray); }
size_t rs_sizeof_error(void)  { return sizeof(struct ArrowError); }
const char* rs_error_message(const struct ArrowError* e) { return e->message; }

/* Import a whole frame: schema + first chunk -> view. Returns n_children, or
 * -k on failure, or -1000-n_children if the stream had a SECOND chunk (the
 * caller then falls back to per-chunk handling; not exercised here). */
int rs_import_frame(struct ArrowArrayStream* stream, struct ArrowSchema* schema,
                    struct ArrowArray* array, struct ArrowArray* array2,
                    struct ArrowArrayView* view, struct ArrowError* err) {
  if (stream->get_schema(stream, schema) != 0) { ArrowErrorSet(err, "get_schema"); return -1; }
  if (stream->get_next(stream, array) != 0)    { ArrowErrorSet(err, "get_next");   return -2; }
  if (array->release == NULL)                  { ArrowErrorSet(err, "no chunks");  return -3; }
  if (stream->get_next(stream, array2) != 0)   { ArrowErrorSet(err, "get_next2");  return -4; }
  if (ArrowArrayViewInitFromSchema(view, schema, err) != NANOARROW_OK) return -5;
  if (ArrowArrayViewSetArrayMinimal(view, array, err) != NANOARROW_OK) return -6;
  if (array2->release != NULL) return -1000 - (int)view->n_children;
  return (int)view->n_children;
}
/* Same, but SetArray (which resolves buffer sizes, e.g. for the checked path). */
int rs_set_array_full(struct ArrowArrayView* view, const struct ArrowArray* array, struct ArrowError* err) {
  return ArrowArrayViewSetArray(view, array, err) == NANOARROW_OK ? 0 : -1;
}
int rs_validate(struct ArrowArrayView* view, int level, struct ArrowError* err) {
  return ArrowArrayViewValidate(view, (enum ArrowValidationLevel)level, err);
}
const struct ArrowArrayView* rs_child(const struct ArrowArrayView* view, int k) { return view->children[k]; }
int rs_storage_type(const struct ArrowArrayView* v) { return (int)v->storage_type; }
int64_t rs_length(const struct ArrowArrayView* v) { return v->length; }
const struct ArrowArrayView* rs_dictionary(const struct ArrowArrayView* v) { return v->dictionary; }
void rs_view_reset(struct ArrowArrayView* v) { ArrowArrayViewReset(v); }
void rs_array_release(struct ArrowArray* a) { if (a->release) ArrowArrayRelease(a); }
void rs_schema_release(struct ArrowSchema* s) { if (s->release) ArrowSchemaRelease(s); }

/* ---- the per-row gather ------------------------------------------------
 * A RowPlan is built once per (schema, call) in Python; the kernel calls
 * rs_gather_row(plan, i) once per row.  kinds: 0=f64 1=i64 2=bool 3=str
 * (spans: addr,len; len -1 = null) 4=code (dictionary index as int64).
 * valid[col] = 1 if not null.  Numeric nulls: f64 -> NaN, i64 -> 0 (the
 * caller reads valid[]).                                                  */
struct RowPlan {
  int32_t ncols;
  const struct ArrowArrayView** views;
  const int8_t* kinds;
  const int32_t* slots;
  double* f64;
  int64_t* i64;
  uint8_t* b8;
  int64_t* spans;
  uint8_t* valid;
};
size_t rs_sizeof_plan(void) { return sizeof(struct RowPlan); }

void rs_gather_row(const struct RowPlan* p, int64_t i) {
  for (int32_t c = 0; c < p->ncols; c++) {
    const struct ArrowArrayView* v = p->views[c];
    int32_t s = p->slots[c];
    int isnull = ArrowArrayViewIsNull(v, i);
    p->valid[c] = (uint8_t)!isnull;
    switch (p->kinds[c]) {
      case 0: p->f64[s] = isnull ? NAN : ArrowArrayViewGetDoubleUnsafe(v, i); break;
      case 1: p->i64[s] = isnull ? 0 : ArrowArrayViewGetIntUnsafe(v, i); break;
      case 2: p->b8[s]  = isnull ? 0 : (uint8_t)ArrowArrayViewGetIntUnsafe(v, i); break;
      case 3: {
        if (isnull) { p->spans[2*s] = 0; p->spans[2*s+1] = -1; }
        else {
          struct ArrowStringView sv = ArrowArrayViewGetStringUnsafe(v, i);
          p->spans[2*s] = (int64_t)(intptr_t)sv.data; p->spans[2*s+1] = sv.size_bytes;
        }
        break;
      }
      case 4: p->i64[s] = isnull ? -1 : ArrowArrayViewGetIntUnsafe(v, i); break; /* dictionary index */
      default: break;
    }
  }
}

/* One string, one call (the lazy STR-node shape from kernel_b). */
int64_t rs_get_string(const struct ArrowArrayView* v, int64_t i, const uint8_t** data) {
  if (ArrowArrayViewIsNull(v, i)) { *data = NULL; return -1; }
  struct ArrowStringView sv = ArrowArrayViewGetStringUnsafe(v, i);
  *data = (const uint8_t*)sv.data; return sv.size_bytes;
}
double rs_get_f64(const struct ArrowArrayView* v, int64_t i) { return ArrowArrayViewGetDoubleUnsafe(v, i); }
int64_t rs_get_i64(const struct ArrowArrayView* v, int64_t i) { return ArrowArrayViewGetIntUnsafe(v, i); }
int rs_is_null(const struct ArrowArrayView* v, int64_t i) { return ArrowArrayViewIsNull(v, i); }

/* ---- V2: column descriptors resolved ONCE per batch from nanoarrow's decoded
 * view (buffer_views, offset, storage type), then a per-row gather that does a
 * plain load per column.  The dispatch on storage type happens per COLUMN per
 * batch, not per element.  Still zero hand-decoding: every pointer below comes
 * out of nanoarrow's ArrowArrayView; the only per-element arithmetic is
 * `base[offset + i]`, which is what nanoarrow's own accessors do.           */
struct ColDesc {
  const uint8_t* validity;   /* NULL when the column has no validity buffer */
  const void* data;          /* buffer_views[1].data */
  const struct ArrowArrayView* view;  /* for strings: the accessor */
  int64_t offset;
  int32_t kind;              /* 0 f64, 1 i64 (any int width via width), 2 bool, 3 str, 4 dict code */
  int32_t width;             /* bytes per element for kinds 0/1/4 */
  int32_t slot;
  int32_t is_signed;
};
struct RowPlan2 {
  int32_t ncols;
  struct ColDesc* cols;
  double* f64; int64_t* i64; uint8_t* b8; int64_t* spans; uint8_t* valid;
};
size_t rs_sizeof_coldesc(void) { return sizeof(struct ColDesc); }
size_t rs_sizeof_plan2(void) { return sizeof(struct RowPlan2); }

/* Fill one ColDesc from a child view. Returns 0, or -1 for a kind/type mismatch. */
int rs_resolve_col(const struct ArrowArrayView* v, int32_t kind, int32_t slot, struct ColDesc* d) {
  d->view = v; d->offset = v->offset; d->kind = kind; d->slot = slot;
  d->validity = v->buffer_views[0].data.as_uint8;   /* nanoarrow leaves NULL when absent */
  d->data = v->buffer_views[1].data.data;
  d->width = 0; d->is_signed = 1;
  switch (v->storage_type) {
    case NANOARROW_TYPE_DOUBLE: d->width = 8; if (kind != 0) return -1; break;
    case NANOARROW_TYPE_FLOAT:  d->width = 4; if (kind != 0) return -1; break;
    case NANOARROW_TYPE_INT64: case NANOARROW_TYPE_TIMESTAMP: case NANOARROW_TYPE_DURATION: case NANOARROW_TYPE_DATE64:
      d->width = 8; if (kind != 1 && kind != 4) return -1; break;
    case NANOARROW_TYPE_INT32: case NANOARROW_TYPE_DATE32: case NANOARROW_TYPE_TIME32: d->width = 4; if (kind != 1 && kind != 4) return -1; break;
    case NANOARROW_TYPE_INT16: d->width = 2; if (kind != 1 && kind != 4) return -1; break;
    case NANOARROW_TYPE_INT8:  d->width = 1; if (kind != 1 && kind != 4) return -1; break;
    case NANOARROW_TYPE_UINT64: d->width = 8; d->is_signed = 0; if (kind != 1 && kind != 4) return -1; break;
    case NANOARROW_TYPE_UINT32: d->width = 4; d->is_signed = 0; if (kind != 1 && kind != 4) return -1; break;
    case NANOARROW_TYPE_UINT16: d->width = 2; d->is_signed = 0; if (kind != 1 && kind != 4) return -1; break;
    case NANOARROW_TYPE_UINT8:  d->width = 1; d->is_signed = 0; if (kind != 1 && kind != 4) return -1; break;
    case NANOARROW_TYPE_BOOL: if (kind != 2) return -1; break;
    case NANOARROW_TYPE_STRING: case NANOARROW_TYPE_LARGE_STRING: case NANOARROW_TYPE_STRING_VIEW:
      if (kind != 3) return -1; break;
    default: return -1;
  }
  return 0;
}

static inline int64_t rs_load_int(const void* data, int32_t width, int32_t is_signed, int64_t j) {
  switch (width) {
    case 8: return is_signed ? ((const int64_t*)data)[j] : (int64_t)((const uint64_t*)data)[j];
    case 4: return is_signed ? ((const int32_t*)data)[j] : (int64_t)((const uint32_t*)data)[j];
    case 2: return is_signed ? ((const int16_t*)data)[j] : (int64_t)((const uint16_t*)data)[j];
    default: return is_signed ? ((const int8_t*)data)[j] : (int64_t)((const uint8_t*)data)[j];
  }
}

void rs_gather_row2(const struct RowPlan2* p, int64_t i) {
  for (int32_t c = 0; c < p->ncols; c++) {
    const struct ColDesc* d = &p->cols[c];
    int64_t j = d->offset + i;
    int isnull = d->validity != NULL && !ArrowBitGet(d->validity, j);
    p->valid[c] = (uint8_t)!isnull;
    switch (d->kind) {
      case 0: p->f64[d->slot] = isnull ? NAN : (d->width == 8 ? ((const double*)d->data)[j] : (double)((const float*)d->data)[j]); break;
      case 1: case 4: p->i64[d->slot] = isnull ? (d->kind == 4 ? -1 : 0) : rs_load_int(d->data, d->width, d->is_signed, j); break;
      case 2: p->b8[d->slot] = isnull ? 0 : (uint8_t)ArrowBitGet((const uint8_t*)d->data, j); break;
      case 3: {
        if (isnull) { p->spans[2*d->slot] = 0; p->spans[2*d->slot+1] = -1; }
        else { struct ArrowStringView sv = ArrowArrayViewGetStringUnsafe(d->view, i);
               p->spans[2*d->slot] = (int64_t)(intptr_t)sv.data; p->spans[2*d->slot+1] = sv.size_bytes; }
        break;
      }
      default: break;
    }
  }
}

/* V3 helper: hand numba the resolved addresses so it can load without any
 * per-row C call.  out[c*4+0]=data address already advanced by offset*width,
 * [1]=validity address (0 if none), [2]=offset (for bit-indexed buffers),
 * [3]=view address (strings).  Bool/validity bits stay bit-indexed. */
void rs_plan2_addrs(const struct RowPlan2* p, uint64_t* out) {
  for (int32_t c = 0; c < p->ncols; c++) {
    const struct ColDesc* d = &p->cols[c];
    out[c*4+0] = (uint64_t)(uintptr_t)d->data + (d->kind == 2 ? 0 : (uint64_t)d->offset * (uint64_t)d->width);
    out[c*4+1] = (uint64_t)(uintptr_t)d->validity;
    out[c*4+2] = (uint64_t)d->offset;
    out[c*4+3] = (uint64_t)(uintptr_t)d->view;
  }
}

/* One call: resolve every column of a frame view into a RowPlan2 and the
 * V3 address table. kinds/slots are per column; returns 0 or -(k+1) for the
 * first column whose Arrow type does not match its declared kind. */
int rs_resolve_all(const struct ArrowArrayView* frame, const int32_t* kinds, const int32_t* slots,
                   const int32_t* child_idx, int32_t ncols, struct RowPlan2* p, uint64_t* addrs) {
  for (int32_t c = 0; c < ncols; c++) {
    const struct ArrowArrayView* v = frame->children[child_idx[c]];
    if (rs_resolve_col(v, kinds[c], slots[c], &p->cols[c]) != 0) return -(c + 1);
  }
  p->ncols = ncols;
  if (addrs) rs_plan2_addrs(p, addrs);
  return 0;
}
