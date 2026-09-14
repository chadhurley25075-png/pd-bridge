/* pd_rdma.c — capture transport for pd-bridge over RoCEv2. R1 and R2 of docs/RDMA.md.
 *
 * One source, three roles, built on the DGX Spark (rdma-core) and on the Mac (MelonDMA libibverbs-compat):
 *
 *   R1 pull   pd_rdma serve  --root ~/pd_capture --bind 10.0.0.2 [--port 18515] [--dev rocep1s0f1] [--gid-index 3]
 *             pd_rdma client --host 10.0.0.2 [--port 18515] [--dev mlx5_0] [--arena-mib 128]
 *   R2 push   pd_rdma recvd  --root ~/pd_pull --bind 10.0.0.1 [--port 18516] [--dev mlx5_0] [--arena-mib 128]
 *             (the sender is libpd_rdma_tx.so inside the vLLM connector, see pd_rdma_tx.c)
 *
 * TCP carries the handshake and a line protocol over 10GbE — the Mac side of the RDMA link has no macOS network
 * interface (the DEXT owns the port). Bulk bytes never touch TCP: they are RDMA-written into an arena the receiving
 * side registered once, then named on the control channel.
 *
 * R1 wire   C->S PULL <tag>
 *           S->C DATA <name> <offset> <len> <total>      (payload already in the client arena)
 *           C->S OK            ...   S->C END <files> <bytes> <read_us> <wire_us>  |  ERR <reason>
 * R2 wire   S->C (recvd publishes the arena, sends READY)
 *           C->S BLK <tag> <index> <L> <H> <B> <D>     arena = k then v, bf16 [L,H,B,D]; written as blk_<index>.safetensors
 *           C->S FILE <tag> <name> <len>               arena = file bytes (manifest.json)
 *           C->S DONE <tag> <status>                   writes <root>/<tag>/DONE last
 *           S->C OK | ERR <reason>                     after each
 *
 * Mac: the provider needs MELONDMA_LOCAL_IP, MELONDMA_LOCAL_MAC, MELONDMA_REMOTE_MAC and the binary must carry the
 * DriverKit userclient-access entitlement (the Makefile signs it).
 */
#include "pd_rdma_common.h"

#include <dirent.h>

/* ------------------------------------------------------------------ R1 server (Spark) */
static int name_order(const struct dirent **a, const struct dirent **b)
{
    int ma = !strcmp((*a)->d_name, "manifest.json"), mb = !strcmp((*b)->d_name, "manifest.json");
    if (ma != mb) return mb - ma;   /* manifest first */
    return strcmp((*a)->d_name, (*b)->d_name);
}

static int wanted(const struct dirent *d)
{
    size_t n = strlen(d->d_name);
    if (d->d_name[0] == '.' || !strcmp(d->d_name, "DONE")) return 0;
    if (n > 4 && !strcmp(d->d_name + n - 4, ".tmp")) return 0;
    return pd_valid_token(d->d_name, 1);
}

static void serve_pull(struct pd_rdma *r, int fd, struct pd_lineio *io, const char *root, const char *tag,
                       const struct pd_hello *peer)
{
    char dir[4096], path[4096 + 256 + 2], line[PD_LINE_MAX];   /* d_name can be 255 bytes before the filter */
    if (!pd_valid_token(tag, 0)) { pd_send_line(fd, "ERR bad-tag"); return; }
    snprintf(dir, sizeof(dir), "%s/%s", root, tag);
    struct dirent **names = NULL;
    int n = scandir(dir, &names, wanted, name_order);
    if (n < 0) { pd_send_line(fd, "ERR no-such-tag"); return; }
    size_t chunk_max = peer->arena_len < r->buf_len ? peer->arena_len : r->buf_len;
    uint64_t bytes = 0;
    int64_t read_us = 0, wire_us = 0;
    int files = 0, ok = 1;
    for (int i = 0; i < n && ok; i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, names[i]->d_name);
        int in = open(path, O_RDONLY);
        struct stat st;
        if (in < 0 || fstat(in, &st) || !S_ISREG(st.st_mode)) {
            if (in >= 0) close(in);
            continue;
        }
        uint64_t total = (uint64_t)st.st_size, off = 0;
        do {
            size_t len = (size_t)(total - off < chunk_max ? total - off : chunk_max);
            int64_t t0 = pd_now_us();
            for (size_t got = 0; got < len;) {
                ssize_t k = pread(in, r->buf + got, len - got, (off_t)(off + got));
                if (k < 0 && errno == EINTR) continue;
                if (k <= 0) { ok = 0; break; }
                got += (size_t)k;
            }
            int64_t t1 = pd_now_us();
            if (!ok || (len && pd_rdma_write(r, len, peer->arena_addr, peer->rkey))) { ok = 0; break; }
            int64_t t2 = pd_now_us();
            read_us += t1 - t0;
            wire_us += t2 - t1;
            if (pd_send_line(fd, "DATA %s %llu %zu %llu", names[i]->d_name, (unsigned long long)off, len,
                             (unsigned long long)total) ||
                pd_read_line(io, line, sizeof(line)) || strcmp(line, "OK")) { ok = 0; break; }
            off += len;
            bytes += len;
        } while (off < total);
        close(in);
        files++;
    }
    for (int i = 0; i < n; i++) free(names[i]);
    free(names);
    if (ok) pd_send_line(fd, "END %d %llu %lld %lld", files, (unsigned long long)bytes, (long long)read_us, (long long)wire_us);
    else pd_send_line(fd, "ERR transfer-failed");
}

static int serve(const char *root, const char *bind_host, int port, const char *dev, int gid_index, size_t buf_len,
                 uint32_t mtu)
{
    struct pd_rdma r;
    if (pd_rdma_open(&r, dev, gid_index, buf_len, 0)) return 1;
    int lfd = pd_tcp_listen(bind_host, port);
    if (lfd < 0) return 1;
    fprintf(stderr, "pd_rdma serve: root=%s control=%s:%d dev=%s gid_index=%d buf=%zu MiB\n", root, bind_host, port,
            dev ? dev : "<first>", gid_index, buf_len >> 20);
    for (;;) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; return pd_errno("accept"), 1; }
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        struct pd_hello peer;
        struct pd_lineio io = { .fd = fd };
        char line[PD_LINE_MAX], tag[PD_NAME_MAX];
        if (pd_qp_create(&r) == 0 && pd_handshake(&r, fd, mtu, 1, 0, &peer) == 0 && peer.arena_addr && peer.rkey &&
            peer.arena_len >= (1u << 20) && pd_send_line(fd, "READY") == 0) {
            fprintf(stderr, "pd_rdma serve: client connected, arena %llu MiB\n", (unsigned long long)(peer.arena_len >> 20));
            while (pd_read_line(&io, line, sizeof(line)) == 0) {
                if (sscanf(line, "PULL %127s", tag) == 1) serve_pull(&r, fd, &io, root, tag, &peer);
                else if (!strcmp(line, "QUIT")) break;
                else pd_send_line(fd, "ERR unknown-command");
            }
        } else {
            fprintf(stderr, "pd_rdma serve: handshake failed\n");
        }
        close(fd);
        pd_qp_destroy(&r);
        fprintf(stderr, "pd_rdma serve: client gone\n");
    }
}

/* ------------------------------------------------------------------ R1 client (Mac) */
static void client_pull(struct pd_rdma *r, int fd, struct pd_lineio *io, const char *tag, const char *outdir)
{
    char line[PD_LINE_MAX], name[PD_NAME_MAX], path[4096], tmp[4200];
    int64_t t0 = pd_now_us(), write_us = 0;
    int out = -1;
    if (mkdir(outdir, 0755) && errno != EEXIST) {
        printf("{\"tag\":\"%s\",\"ok\":false,\"error\":\"mkdir %s\"}\n", tag, strerror(errno));
        return;
    }
    if (pd_send_line(fd, "PULL %s", tag)) { printf("{\"tag\":\"%s\",\"ok\":false,\"error\":\"control-send\"}\n", tag); return; }
    for (;;) {
        if (pd_read_line(io, line, sizeof(line))) { printf("{\"tag\":\"%s\",\"ok\":false,\"error\":\"control-lost\"}\n", tag); return; }
        unsigned long long off, total, bytes;
        size_t len;
        int files;
        long long read_us, wire_us;
        if (sscanf(line, "DATA %127s %llu %zu %llu", name, &off, &len, &total) == 4) {
            if (!pd_valid_token(name, 1) || len > r->buf_len) {
                printf("{\"tag\":\"%s\",\"ok\":false,\"error\":\"bad DATA line\"}\n", tag);
                return;
            }
            snprintf(path, sizeof(path), "%s/%s", outdir, name);
            snprintf(tmp, sizeof(tmp), "%s.tmp", path);
            int64_t w0 = pd_now_us();
            if (off == 0) {
                if (out >= 0) close(out);
                out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            }
            int bad = out < 0 || pd_write_full(out, r->buf, len);
            if (!bad && off + len == total) {
                bad = close(out) != 0 || rename(tmp, path) != 0;
                out = -1;
            }
            write_us += pd_now_us() - w0;
            if (bad) { printf("{\"tag\":\"%s\",\"ok\":false,\"error\":\"write %s\"}\n", tag, strerror(errno)); return; }
            if (pd_send_line(fd, "OK")) { printf("{\"tag\":\"%s\",\"ok\":false,\"error\":\"control-send\"}\n", tag); return; }
        } else if (sscanf(line, "END %d %llu %lld %lld", &files, &bytes, &read_us, &wire_us) == 4) {
            double total_s = (pd_now_us() - t0) / 1e6;
            printf("{\"tag\":\"%s\",\"ok\":true,\"files\":%d,\"bytes\":%llu,\"t_total_s\":%.3f,\"t_server_read_s\":%.3f,"
                   "\"t_wire_s\":%.3f,\"t_client_write_s\":%.3f,\"wire_gb_per_s\":%.3f}\n",
                   tag, files, bytes, total_s, read_us / 1e6, wire_us / 1e6, write_us / 1e6,
                   wire_us > 0 ? bytes / 1e9 / (wire_us / 1e6) : 0.0);
            return;
        } else {
            printf("{\"tag\":\"%s\",\"ok\":false,\"error\":\"%s\"}\n", tag, strncmp(line, "ERR ", 4) ? "unexpected reply" : line + 4);
            return;
        }
    }
}

static int client(const char *host, int port, const char *dev, int gid_index, size_t arena_len, uint32_t mtu)
{
    struct pd_rdma r;
    if (pd_rdma_open(&r, dev, gid_index, arena_len, 1) || pd_qp_create(&r)) return 1;
    int fd = pd_tcp_connect(host, port);
    if (fd < 0) return 1;
    struct pd_hello peer;
    struct pd_lineio io = { .fd = fd };
    char line[PD_LINE_MAX], cmd[PD_LINE_MAX], tag[PD_NAME_MAX], outdir[2048];
    if (pd_handshake(&r, fd, mtu, 0, 1, &peer) || pd_read_line(&io, line, sizeof(line)) || strcmp(line, "READY"))
        return pd_err("handshake with %s:%d failed", host, port), 1;
    printf("{\"ready\":true,\"arena_mib\":%zu}\n", arena_len >> 20);
    fflush(stdout);
    while (fgets(cmd, sizeof(cmd), stdin)) {
        if (sscanf(cmd, "PULL %127s %2047s", tag, outdir) == 2) client_pull(&r, fd, &io, tag, outdir);
        else if (!strncmp(cmd, "QUIT", 4)) break;
        else printf("{\"ok\":false,\"error\":\"usage: PULL <tag> <outdir>\"}\n");
        fflush(stdout);
    }
    pd_send_line(fd, "QUIT");
    close(fd);
    return 0;
}

/* ------------------------------------------------------------------ R2 receiver (Mac) */
static int ensure_dir(const char *path)
{
    return (mkdir(path, 0755) && errno != EEXIST) ? pd_errno("mkdir") : 0;
}

static int write_file_atomic(const char *path, const void *a, size_t alen, const void *b, size_t blen,
                             const void *c, size_t clen)
{
    char tmp[4400];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    int out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) return pd_errno("open");
    int bad = (alen && pd_write_full(out, a, alen)) || (blen && pd_write_full(out, b, blen)) ||
              (clen && pd_write_full(out, c, clen));
    bad = close(out) != 0 || bad;
    if (bad || rename(tmp, path)) { unlink(tmp); return pd_errno("write/rename"); }
    return 0;
}

/* The block as a safetensors file with tensors k and v, exactly what studio/pd_assemble_kv.py reads. */
static int write_block(const char *dir, int index, const uint8_t *arena, int L, int H, int B, int D)
{
    uint64_t each = (uint64_t)L * H * B * D * 2;   /* bf16 */
    char path[4300], hdr[512];
    snprintf(path, sizeof(path), "%s/blk_%06d.safetensors", dir, index);
    int h = snprintf(hdr, sizeof(hdr),
                     "{\"k\":{\"dtype\":\"BF16\",\"shape\":[%d,%d,%d,%d],\"data_offsets\":[0,%llu]},"
                     "\"v\":{\"dtype\":\"BF16\",\"shape\":[%d,%d,%d,%d],\"data_offsets\":[%llu,%llu]}}",
                     L, H, B, D, (unsigned long long)each, L, H, B, D, (unsigned long long)each,
                     (unsigned long long)(2 * each));
    if (h < 0 || h >= (int)sizeof(hdr) - 8) return pd_err("block header overflow");
    while (h % 8) hdr[h++] = ' ';   /* safetensors pads the header to 8 bytes */
    uint8_t len8[8];
    for (int i = 0; i < 8; i++) len8[i] = (uint8_t)((uint64_t)h >> (8 * i));   /* little-endian header length */
    return write_file_atomic(path, len8, 8, hdr, (size_t)h, arena, (size_t)(2 * each));
}

static int is_hex64(const char *s)
{
    if (strlen(s) != 64) return 0;
    for (; *s; s++)
        if (!((*s >= '0' && *s <= '9') || (*s >= 'a' && *s <= 'f'))) return 0;
    return 1;
}

/* ------------------------------------------------------------------ R4 async block writes (Mac)
 * The ack used to wait for the block's file write (~20 ms of ~43 ms per 64 MiB block, 2.3 s of writes per 8 GB), so the
 * last prefill step's 32-block burst reached DONE ~1.3 s after the engine. OBLK now copies the arena into a pre-touched
 * pool buffer, queues it and acks; one writer thread lands the files in order. FILE and DONE drain the queue first, so
 * the tail, the manifest and DONE still land after every block, and a failed background write is reported as ERR on
 * the next command. PD_RDMA_ASYNC_WRITES=0 restores the synchronous path (A/B control). */
#include <pthread.h>

#define PD_WQ_DEPTH 4
struct pd_wjob { char path[4096 + 2 * PD_NAME_MAX + 4]; size_t len; uint8_t *buf; };
struct pd_writer {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct pd_wjob jobs[PD_WQ_DEPTH];
    uint8_t *free_bufs[PD_WQ_DEPTH];
    int head, count, nfree, busy, failed;
    int64_t write_us;
    pthread_t th;
};

static void *pd_writer_main(void *arg)
{
    struct pd_writer *w = arg;
    pthread_mutex_lock(&w->mu);
    for (;;) {
        while (!w->count) pthread_cond_wait(&w->cv, &w->mu);
        struct pd_wjob job = w->jobs[w->head];
        w->head = (w->head + 1) % PD_WQ_DEPTH;
        w->count--;
        w->busy = 1;
        pthread_mutex_unlock(&w->mu);
        int64_t t0 = pd_now_us();
        int rc = write_file_atomic(job.path, job.buf, job.len, NULL, 0, NULL, 0);
        int64_t dt = pd_now_us() - t0;
        pthread_mutex_lock(&w->mu);
        w->write_us += dt;
        if (rc) w->failed = 1;
        w->free_bufs[w->nfree++] = job.buf;
        w->busy = 0;
        pthread_cond_broadcast(&w->cv);
    }
    return NULL;
}

static int pd_writer_start(struct pd_writer *w, size_t buf_len)
{
    memset(w, 0, sizeof(*w));
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cv, NULL);
    for (int i = 0; i < PD_WQ_DEPTH; i++) {
        uint8_t *b = malloc(buf_len);
        if (!b) return pd_err("writer buffer allocation failed");
        memset(b, 0, buf_len);                        /* pre-touch: a fresh page fault per block costs what the write did */
        w->free_bufs[w->nfree++] = b;
    }
    if (pthread_create(&w->th, NULL, pd_writer_main, w)) return pd_err("writer thread failed to start");
    return 0;
}

/* Queue one file; blocks while every pool buffer is in flight (backpressure onto the sender's ack). */
static int pd_writer_put(struct pd_writer *w, const char *path, const uint8_t *src, size_t len)
{
    pthread_mutex_lock(&w->mu);
    while (!w->nfree && !w->failed) pthread_cond_wait(&w->cv, &w->mu);
    if (w->failed) { pthread_mutex_unlock(&w->mu); return pd_err("an earlier background write failed"); }
    uint8_t *buf = w->free_bufs[--w->nfree];
    pthread_mutex_unlock(&w->mu);
    memcpy(buf, src, len);
    pthread_mutex_lock(&w->mu);
    struct pd_wjob *j = &w->jobs[(w->head + w->count) % PD_WQ_DEPTH];
    snprintf(j->path, sizeof(j->path), "%s", path);
    j->len = len;
    j->buf = buf;
    w->count++;
    pthread_cond_broadcast(&w->cv);
    pthread_mutex_unlock(&w->mu);
    return 0;
}

/* Wait until every queued file has landed; returns -1 (and clears the flag) if any of them failed. */
static int pd_writer_drain(struct pd_writer *w, int64_t *write_us)
{
    pthread_mutex_lock(&w->mu);
    while (w->count || w->busy) pthread_cond_wait(&w->cv, &w->mu);
    int failed = w->failed;
    w->failed = 0;
    if (write_us) *write_us += w->write_us;
    w->write_us = 0;
    pthread_mutex_unlock(&w->mu);
    return failed ? pd_err("a background block write failed") : 0;
}

static int recvd(const char *root, const char *omlx_cache, const char *bind_host, int port, const char *dev,
                 int gid_index, size_t arena_len, uint32_t mtu)
{
    struct pd_rdma r;
    if (pd_rdma_open(&r, dev, gid_index, arena_len, 1)) return 1;
    const char *aw = getenv("PD_RDMA_ASYNC_WRITES");
    int async_writes = !aw || strcmp(aw, "0");
    struct pd_writer w;
    if (async_writes && pd_writer_start(&w, r.buf_len)) return 1;
    int lfd = pd_tcp_listen(bind_host, port);
    if (lfd < 0) return 1;
    fprintf(stderr, "pd_rdma recvd: root=%s control=%s:%d arena=%zu MiB async_writes=%d\n", root, bind_host, port, arena_len >> 20,
            async_writes);
    for (;;) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; return pd_errno("accept"), 1; }
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        struct pd_hello peer;
        struct pd_lineio io = { .fd = fd };
        if (pd_qp_create(&r) || pd_handshake(&r, fd, mtu, 1, 1, &peer) || pd_send_line(fd, "READY")) {
            fprintf(stderr, "pd_rdma recvd: handshake failed\n");
            close(fd);
            pd_qp_destroy(&r);
            continue;
        }
        fprintf(stderr, "pd_rdma recvd: sender connected\n");
        char line[PD_LINE_MAX], tag[PD_NAME_MAX], name[PD_NAME_MAX], status[64], dir[4096 + PD_NAME_MAX + 2],
             path[4096 + 2 * PD_NAME_MAX + 4];
        int64_t write_us = 0;
        while (pd_read_line(&io, line, sizeof(line)) == 0) {
            int index, L, H, B, D;
            unsigned long long len;
            int rc;
            if (sscanf(line, "BLK %127s %d %d %d %d %d", tag, &index, &L, &H, &B, &D) == 6) {
                if (!pd_valid_token(tag, 0) || index < 0 || L <= 0 || H <= 0 || B <= 0 || D <= 0 ||
                    (uint64_t)L * H * B * D * 4 > r.buf_len) { pd_send_line(fd, "ERR bad-block"); continue; }
                snprintf(dir, sizeof(dir), "%s/%s", root, tag);
                int64_t t0 = pd_now_us();
                rc = ensure_dir(dir) || write_block(dir, index, r.buf, L, H, B, D);
                write_us += pd_now_us() - t0;
            } else if (sscanf(line, "FILE %127s %127s %llu", tag, name, &len) == 3) {
                if (!pd_valid_token(tag, 0) || !pd_valid_token(name, 1) || len > r.buf_len) { pd_send_line(fd, "ERR bad-file"); continue; }
                snprintf(dir, sizeof(dir), "%s/%s", root, tag);
                snprintf(path, sizeof(path), "%s/%s", dir, name);
                rc = (async_writes && pd_writer_drain(&w, &write_us)) || ensure_dir(dir) ||
                     write_file_atomic(path, r.buf, (size_t)len, NULL, 0, NULL, 0);
            } else if (sscanf(line, "OBLK %127s %llu", name, &len) == 2) {
                /* R4: a complete oMLX block file, landed where the decoder's disk-index fallback looks for it */
                if (!omlx_cache || !is_hex64(name) || len > r.buf_len) { pd_send_line(fd, "ERR bad-oblk"); continue; }
                snprintf(dir, sizeof(dir), "%s/%c", omlx_cache, name[0]);
                snprintf(path, sizeof(path), "%s/%s.safetensors", dir, name);
                if (async_writes) {
                    rc = ensure_dir(dir) || pd_writer_put(&w, path, r.buf, (size_t)len);
                } else {
                    int64_t t0 = pd_now_us();
                    rc = ensure_dir(dir) || write_file_atomic(path, r.buf, (size_t)len, NULL, 0, NULL, 0);
                    write_us += pd_now_us() - t0;
                }
            } else if (sscanf(line, "DONE %127s %63s", tag, status) == 2) {
                if (!pd_valid_token(tag, 0)) { pd_send_line(fd, "ERR bad-tag"); continue; }
                snprintf(dir, sizeof(dir), "%s/%s", root, tag);
                snprintf(path, sizeof(path), "%s/DONE", dir);
                rc = (async_writes && pd_writer_drain(&w, &write_us)) || ensure_dir(dir) ||
                     write_file_atomic(path, status, strlen(status), NULL, 0, NULL, 0);
                printf("{\"tag\":\"%s\",\"done\":\"%s\",\"t_write_s\":%.3f}\n", tag, status, write_us / 1e6);
                fflush(stdout);
                write_us = 0;
            } else if (!strcmp(line, "QUIT")) {
                break;
            } else {
                pd_send_line(fd, "ERR unknown-command");
                continue;
            }
            if (pd_send_line(fd, rc ? "ERR write-failed" : "OK")) break;
        }
        if (async_writes && pd_writer_drain(&w, NULL))   /* land what the last sender queued before the next one */
            fprintf(stderr, "pd_rdma recvd: background writes of the last sender failed\n");
        close(fd);
        pd_qp_destroy(&r);
        fprintf(stderr, "pd_rdma recvd: sender gone\n");
    }
}

/* ------------------------------------------------------------------ main */
static const char *opt(int argc, char **argv, const char *name, const char *def)
{
    for (int i = 2; i + 1 < argc; i++)
        if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}

int main(int argc, char **argv)
{
    if (argc < 2 || (strcmp(argv[1], "serve") && strcmp(argv[1], "client") && strcmp(argv[1], "recvd"))) {
        fprintf(stderr, "usage: pd_rdma serve  --root DIR --bind IP [--port 18515] [--dev NAME] [--gid-index N] [--buf-mib 128] [--mtu 4096]\n"
                        "       pd_rdma client --host IP [--port 18515] [--dev NAME] [--gid-index N] [--arena-mib 128] [--mtu 4096]\n"
                        "       pd_rdma recvd  --root DIR --bind IP [--port 18516] [--dev NAME] [--gid-index N] [--arena-mib 128] [--mtu 4096]\n");
        return 2;
    }
    const char *dev = opt(argc, argv, "--dev", NULL);
    int gid_index = atoi(opt(argc, argv, "--gid-index", "0"));
    uint32_t mtu = (uint32_t)atoi(opt(argc, argv, "--mtu", "4096"));
    if (!strcmp(argv[1], "serve")) {
        const char *root = opt(argc, argv, "--root", NULL), *bind_host = opt(argc, argv, "--bind", NULL);
        if (!root || !bind_host) { fprintf(stderr, "pd_rdma serve: --root and --bind are required\n"); return 2; }
        size_t mib = (size_t)atoi(opt(argc, argv, "--buf-mib", "128"));
        return serve(root, bind_host, atoi(opt(argc, argv, "--port", "18515")), dev, gid_index, mib << 20, mtu);
    }
    if (!strcmp(argv[1], "recvd")) {
        const char *root = opt(argc, argv, "--root", NULL), *bind_host = opt(argc, argv, "--bind", NULL);
        if (!root || !bind_host) { fprintf(stderr, "pd_rdma recvd: --root and --bind are required\n"); return 2; }
        size_t mib = (size_t)atoi(opt(argc, argv, "--arena-mib", "128"));
        const char *omlx_cache = opt(argc, argv, "--omlx-cache", NULL);   /* R4: OBLK lands here; without it OBLK is refused */
        return recvd(root, omlx_cache, bind_host, atoi(opt(argc, argv, "--port", "18516")), dev, gid_index, mib << 20, mtu);
    }
    const char *host = opt(argc, argv, "--host", NULL);
    if (!host) { fprintf(stderr, "pd_rdma client: --host is required\n"); return 2; }
    size_t mib = (size_t)atoi(opt(argc, argv, "--arena-mib", "128"));
    return client(host, atoi(opt(argc, argv, "--port", "18515")), dev, gid_index, mib << 20, mtu);
}
