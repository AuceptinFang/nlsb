/* 线程池驱动：每个线程从共享计数器取下一个文件，跑完 读文件 → 解码 → 预处理 → 特征 → 评分。
 * 图片大小差异很大（几 KB 的 GIF 到 12 MP 的照片），所以用动态分配而不是静态切分。 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "nl.h"

void nl_decode_thread_cleanup(void);  /* decode.c：释放线程局部的 libdeflate 解压器 */

typedef struct {
    const char *const *paths;
    int n;
    double min_yellow;
    const double *params;  /* npresets × NL_NPARAM */
    int npresets;
    double *feats;         /* n × NL_NFEAT */
    double *scores;        /* n × npresets */
    int *status;           /* NL_OK / NL_UNSUPPORTED / NL_ERROR / 3=读文件失败 */
    atomic_int next;
    int *order;            /* 派发顺序：按文件大小从大到小（最长任务优先，避免尾部空等） */
} job_t;

typedef struct { off_t size; int idx; } sized_t;

static int by_size_desc(const void *a, const void *b) {
    const sized_t *x = a, *y = b;
    if (x->size != y->size) return x->size < y->size ? 1 : -1;
    return x->idx - y->idx;
}

static int *size_order(const char *const *paths, int n) {
    sized_t *s = malloc((size_t)n * sizeof *s);
    int *order = malloc((size_t)n * sizeof *order);
    if (!s || !order) { free(s); free(order); return NULL; }
    for (int i = 0; i < n; i++) {
        struct stat st;
        s[i].size = stat(paths[i], &st) == 0 ? st.st_size : 0;
        s[i].idx = i;
    }
    qsort(s, (size_t)n, sizeof *s, by_size_desc);
    for (int i = 0; i < n; i++) order[i] = s[i].idx;
    free(s);
    return order;
}

static int read_file(const char *path, uint8_t **buf, size_t *cap, size_t *len) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t need = (size_t)st.st_size;
    if (need > *cap) {
        uint8_t *p = realloc(*buf, need);
        if (!p) { close(fd); return -1; }
        *buf = p;
        *cap = need;
    }
    size_t got = 0;
    while (got < need) {
        ssize_t r = read(fd, *buf + got, need - got);
        if (r <= 0) { close(fd); return -1; }
        got += (size_t)r;
    }
    close(fd);
    *len = got;
    return 0;
}

static void *worker(void *arg) {
    job_t *j = arg;
    uint8_t *buf = NULL;
    size_t cap = 0, len = 0;
    for (;;) {
        int k = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed);
        if (k >= j->n) break;
        int i = j->order ? j->order[k] : k;
        if (read_file(j->paths[i], &buf, &cap, &len) != 0) { j->status[i] = 3; continue; }
        nl_img img = {0};
        int st = nl_decode(buf, len, &img);
        if (st != NL_OK) { free(img.px); j->status[i] = st; continue; }
        nl_prep prep = {0};
        st = nl_prepare(&img, &prep);
        free(img.px);
        if (st != NL_OK) { nl_prep_free(&prep); j->status[i] = st; continue; }
        int skipped = 0;
        double *f = j->feats + (size_t)i * NL_NFEAT;
        st = nl_features(&prep, j->min_yellow, f, &skipped);
        nl_prep_free(&prep);
        if (st != NL_OK) { j->status[i] = st; continue; }
        for (int k = 0; k < j->npresets; k++)
            j->scores[(size_t)i * j->npresets + k] = nl_score(f, j->params + (size_t)k * NL_NPARAM);
        j->status[i] = NL_OK;
        if (cap > (64u << 20)) { free(buf); buf = NULL; cap = 0; }  /* 偶尔的超大文件不常驻 */
    }
    free(buf);
    nl_decode_thread_cleanup();
    return NULL;
}

int nl_run_files(const char *const *paths, int n, int threads, double min_yellow,
                 const double *params, int npresets, double *feats, double *scores, int *status) {
    job_t j = {paths, n, min_yellow, params, npresets, feats, scores, status, 0, NULL};
    if (threads < 1) threads = 1;
    if (threads == 1) { worker(&j); return 0; }
    const char *sched = getenv("NL_SCHED");  /* "fifo" = 按输入顺序派发，用于对比 */
    j.order = (sched && strcmp(sched, "fifo") == 0) ? NULL : size_order(paths, n);
    pthread_t *t = calloc((size_t)threads, sizeof *t);
    if (!t) { free(j.order); return -1; }
    int started = 0;
    for (; started < threads; started++)
        if (pthread_create(&t[started], NULL, worker, &j) != 0) break;
    if (started == 0) worker(&j);
    for (int k = 0; k < started; k++) pthread_join(t[k], NULL);
    free(t);
    free(j.order);
    return 0;
}
