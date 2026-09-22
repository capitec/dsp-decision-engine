/* strmatch.c -- the C side of "regex evaluated lazily at the node".
 *
 * A tiny shim over PCRE2 (8-bit, JIT) plus four hand-rolled literal
 * matchers (exact / prefix / suffix / substring), exposing exactly three
 * entry points a numba kernel can reach through a raw function pointer:
 *
 *   sm_compile(pat, len, kind, flags, &err, &erroff) -> id >= 0, or -1
 *   sm_match_id(id, subject, len)                    -> 1 / 0 / negative
 *   sm_free(id)
 *
 * C owns the table of compiled patterns (the registry); a kernel only ever
 * passes a small integer id, bounds-checked here. No raw pointer crosses.
 *
 * Nothing here allocates or touches Python on the per-row path: sm_match
 * is a (handle, pointer, length) call, the same shape as the Rust
 * `regex_is_match` in ../rust-cabi-in-kernel, so the two strands are
 * comparable call-for-call.
 *
 * PCRE2 has no headers on this box (only libpcre2-8.so.0), so the handful
 * of prototypes and constants we use are declared here by hand, straight
 * from pcre2.h. Only the 8-bit code-unit width is used.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>

/* ---- PCRE2 8-bit prototypes (from pcre2.h; no header installed) ---- */
typedef struct pcre2_real_code_8 pcre2_code_8;
typedef struct pcre2_real_match_data_8 pcre2_match_data_8;
typedef struct pcre2_real_general_context_8 pcre2_general_context_8;
typedef struct pcre2_real_compile_context_8 pcre2_compile_context_8;
typedef struct pcre2_real_match_context_8 pcre2_match_context_8;

extern pcre2_code_8 *pcre2_compile_8(const uint8_t *, size_t, uint32_t, int *, size_t *,
                                     pcre2_compile_context_8 *);
extern int pcre2_jit_compile_8(pcre2_code_8 *, uint32_t);
extern pcre2_match_data_8 *pcre2_match_data_create_from_pattern_8(const pcre2_code_8 *,
                                                                  pcre2_general_context_8 *);
extern int pcre2_match_8(const pcre2_code_8 *, const uint8_t *, size_t, size_t, uint32_t,
                         pcre2_match_data_8 *, pcre2_match_context_8 *);
extern int pcre2_jit_match_8(const pcre2_code_8 *, const uint8_t *, size_t, size_t, uint32_t,
                             pcre2_match_data_8 *, pcre2_match_context_8 *);
extern void pcre2_match_data_free_8(pcre2_match_data_8 *);
extern void pcre2_code_free_8(pcre2_code_8 *);
extern int pcre2_get_error_message_8(int, uint8_t *, size_t);

#define PCRE2_UTF            0x00080000u
#define PCRE2_NO_UTF_CHECK   0x40000000u
#define PCRE2_JIT_COMPLETE   0x00000001u
#define PCRE2_ERROR_NOMATCH  (-1)

/* ---- matcher kinds and flags (mirrored in strmatch.py) ---- */
enum { SM_REGEX = 0, SM_EXACT = 1, SM_PREFIX = 2, SM_SUFFIX = 3, SM_SUBSTRING = 4,
       SM_POSIX = 5 /* glibc regexec, the §T control; copies to a NUL-terminated buffer */ };
enum { SM_FLAG_UTF = 1, SM_FLAG_NO_JIT = 2 };

#define SM_MAGIC 0x534D4154u  /* "SMAT": a cheap sanity check on the handle */

typedef struct {
    uint32_t magic;
    int32_t kind;
    int32_t flags;
    int32_t pat_len;
    uint8_t *pat;                 /* literal kinds: the pattern bytes        */
    pcre2_code_8 *code;           /* regex kind: compiled (JIT if possible)  */
    pcre2_match_data_8 *md;       /* regex kind: one match block per handle  */
    int jit;                      /* regex kind: pcre2_jit_match usable?     */
    regex_t posix;                /* SM_POSIX: glibc compiled regex           */
    int posix_ok;
} sm_handle;

/* ---- The registry: C OWNS the table of live patterns. A kernel never
 * holds a pointer -- it passes a small integer id which is bounds-checked
 * here against the registry before anything is dereferenced. This is what
 * turns "garbage handle -> SIGSEGV" into "garbage id -> -1". ---- */
#define SM_MAX_HANDLES 4096
static sm_handle *g_registry[SM_MAX_HANDLES];
static int32_t g_n_registry = 0;

/* One shared match context carrying the match limit. PCRE2 is a
 * backtracking engine: a pathological pattern (`^(a+)+$`) is exponential
 * and the DEFAULT limit of 10,000,000 steps costs ~40 ms per row. We cap
 * it hard; a limit hit is reported as -4 and the row becomes ERR_LEAF. */
extern pcre2_match_context_8 *pcre2_match_context_create_8(pcre2_general_context_8 *);
extern int pcre2_set_match_limit_8(pcre2_match_context_8 *, uint32_t);
extern int pcre2_set_depth_limit_8(pcre2_match_context_8 *, uint32_t);
static pcre2_match_context_8 *g_mctx = NULL;
static uint32_t g_match_limit = 50000;

static void ensure_mctx(void)
{
    if (!g_mctx) g_mctx = pcre2_match_context_create_8(NULL);
    if (g_mctx) { pcre2_set_match_limit_8(g_mctx, g_match_limit); pcre2_set_depth_limit_8(g_mctx, g_match_limit); }
}

void sm_set_match_limit(uint32_t limit) { g_match_limit = limit; ensure_mctx(); }
uint32_t sm_get_match_limit(void) { return g_match_limit; }

/* Compile a pattern. Returns NULL and sets *err (PCRE2 error code, or
 * -100 for a bad kind, -101 for a NULL pattern, -102 for OOM) on failure.
 * Never raises, never aborts: an invalid regex is a NULL return, full stop. */
static sm_handle *compile_handle(const uint8_t *pat, int32_t len, int32_t kind, int32_t flags,
                                 int32_t *err, int64_t *erroff)
{
    if (err) *err = 0;
    if (erroff) *erroff = 0;
    if (pat == NULL && len > 0) { if (err) *err = -101; return NULL; }
    if (len < 0)                 { if (err) *err = -101; return NULL; }
    if (kind < SM_REGEX || kind > SM_POSIX) { if (err) *err = -100; return NULL; }

    sm_handle *h = calloc(1, sizeof *h);
    if (!h) { if (err) *err = -102; return NULL; }
    h->magic = SM_MAGIC; h->kind = kind; h->flags = flags; h->pat_len = len;
    h->pat = malloc(len > 0 ? (size_t)len : 1);
    if (!h->pat) { free(h); if (err) *err = -102; return NULL; }
    if (len > 0) memcpy(h->pat, pat, (size_t)len);

    if (kind == SM_REGEX) {
        int ec = 0; size_t eo = 0;
        uint32_t copts = (flags & SM_FLAG_UTF) ? PCRE2_UTF : 0u;
        h->code = pcre2_compile_8(pat, (size_t)len, copts, &ec, &eo, NULL);
        if (!h->code) {
            if (err) *err = ec;
            if (erroff) *erroff = (int64_t)eo;
            free(h->pat); free(h);
            return NULL;
        }
        h->md = pcre2_match_data_create_from_pattern_8(h->code, NULL);
        if (!h->md) { pcre2_code_free_8(h->code); free(h->pat); free(h); if (err) *err = -102; return NULL; }
        h->jit = 0;
        if (!(flags & SM_FLAG_NO_JIT)) {
            if (pcre2_jit_compile_8(h->code, PCRE2_JIT_COMPLETE) == 0) h->jit = 1;
        }
    }
    if (kind == SM_POSIX) {
        char *z = malloc((size_t)len + 1);
        if (!z) { free(h->pat); free(h); if (err) *err = -102; return NULL; }
        memcpy(z, pat, (size_t)len); z[len] = 0;
        int rc = regcomp(&h->posix, z, REG_EXTENDED | REG_NOSUB);
        free(z);
        if (rc != 0) { if (err) *err = -200 - rc; free(h->pat); free(h); return NULL; }
        h->posix_ok = 1;
    }
    return h;
}

/* Public: compile and REGISTER. Returns the id (>= 0) or -1 on failure
 * (with *err set; -103 = registry full). */
int64_t sm_compile(const uint8_t *pat, int32_t len, int32_t kind, int32_t flags,
                   int32_t *err, int64_t *erroff)
{
    ensure_mctx();
    sm_handle *h = compile_handle(pat, len, kind, flags, err, erroff);
    if (!h) return -1;
    int32_t id = -1;
    for (int32_t i = 0; i < g_n_registry; i++) if (g_registry[i] == NULL) { id = i; break; }
    if (id < 0) {
        if (g_n_registry >= SM_MAX_HANDLES) { if (err) *err = -103; /* free below */
            if (h->md) pcre2_match_data_free_8(h->md); if (h->code) pcre2_code_free_8(h->code);
            if (h->posix_ok) regfree(&h->posix); free(h->pat); free(h); return -1; }
        id = g_n_registry++;
    }
    g_registry[id] = h;
    return id;
}

int32_t sm_error_message(int32_t err, uint8_t *buf, int32_t buflen)
{
    if (err == -100) { strncpy((char *)buf, "bad matcher kind", (size_t)buflen); return 0; }
    if (err == -101) { strncpy((char *)buf, "null/negative pattern", (size_t)buflen); return 0; }
    if (err == -102) { strncpy((char *)buf, "out of memory", (size_t)buflen); return 0; }
    if (err == -103) { strncpy((char *)buf, "pattern registry full", (size_t)buflen); return 0; }
    if (err <= -200) { strncpy((char *)buf, "POSIX regcomp failed", (size_t)buflen); return 0; }
    return pcre2_get_error_message_8(err, buf, (size_t)buflen);
}

/* The per-row call. 1 = match, 0 = no match, <0 = error (never a crash):
 *   -1 bad/NULL handle   -2 negative length   -3 NULL subject with len>0
 *   -4 PCRE2 runtime error (e.g. match limit)
 * NOTE: the regex handle carries ONE match-data block, so a handle must not
 * be shared across threads (a prange kernel would need a per-thread handle).
 */
static inline int32_t match_impl(const sm_handle *h, const uint8_t *subject, int64_t len)
{
    if (len < 0) return -2;
    if (subject == NULL && len > 0) return -3;
    const size_t n = (size_t)len, p = (size_t)h->pat_len;
    switch (h->kind) {
    case SM_EXACT:     return n == p && memcmp(subject, h->pat, p) == 0;
    case SM_PREFIX:    return n >= p && memcmp(subject, h->pat, p) == 0;
    case SM_SUFFIX:    return n >= p && memcmp(subject + (n - p), h->pat, p) == 0;
    case SM_SUBSTRING: return p == 0 || (n >= p && memmem(subject, n, h->pat, p) != NULL);
    case SM_REGEX: {
        uint32_t mopts = (h->flags & SM_FLAG_UTF) ? PCRE2_NO_UTF_CHECK : 0u;
        int rc = h->jit
            ? pcre2_jit_match_8(h->code, subject, n, 0, mopts, h->md, g_mctx)
            : pcre2_match_8(h->code, subject, n, 0, mopts, h->md, g_mctx);
        if (rc >= 0) return 1;
        if (rc == PCRE2_ERROR_NOMATCH) return 0;
        return -4;
    }
    case SM_POSIX: {
        /* regexec needs a NUL terminator; Arrow buffers have none. Copy to
         * a stack buffer (heap if long) -- this copy is part of what §T paid. */
        char stackbuf[256]; char *z = stackbuf;
        if (n + 1 > sizeof stackbuf) { z = malloc(n + 1); if (!z) return -4; }
        memcpy(z, subject, n); z[n] = 0;
        int rc = regexec(&h->posix, z, 0, NULL, 0);
        if (z != stackbuf) free(z);
        return rc == 0 ? 1 : (rc == REG_NOMATCH ? 0 : -4);
    }
    default: return -1;
    }
}

/* THE per-row entry point a kernel calls: (id, subject, len). Any int64
 * is safe to pass as `id`. */
int32_t sm_match_id(int64_t id, const uint8_t *subject, int64_t len)
{
    if (id < 0 || id >= g_n_registry) return -1;
    const sm_handle *h = g_registry[id];
    if (h == NULL) return -1;
    return match_impl(h, subject, len);
}

/* Raw-pointer variant, kept ONLY so bench_regex.py can show the registry
 * indirection costs nothing measurable. A kernel must never use it. */
int32_t sm_match(void *handle, const uint8_t *subject, int64_t len)
{
    const sm_handle *h = (const sm_handle *)handle;
    if (h == NULL || h->magic != SM_MAGIC) return -1;
    return match_impl(h, subject, len);
}

void *sm_handle_ptr(int64_t id)
{
    if (id < 0 || id >= g_n_registry) return NULL;
    return g_registry[id];
}

int32_t sm_is_jit(int64_t id)
{
    const sm_handle *h = (const sm_handle *)sm_handle_ptr(id);
    if (h == NULL) return -1;
    return h->kind == SM_REGEX ? h->jit : 0;
}

void sm_free(int64_t id)
{
    sm_handle *h = (sm_handle *)sm_handle_ptr(id);
    if (h == NULL) return;
    g_registry[id] = NULL;
    if (h->md) pcre2_match_data_free_8(h->md);
    if (h->code) pcre2_code_free_8(h->code);
    if (h->posix_ok) regfree(&h->posix);
    free(h->pat);
    h->magic = 0;
    free(h);
}

/* Call-overhead probe: the same trivial shape §V measured (int32,int32)->int32. */
int32_t sm_trivial(int32_t a, int32_t b) { return a + b; }
