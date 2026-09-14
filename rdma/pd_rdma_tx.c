/* pd_rdma_tx.c — libpd_rdma_tx.so, the R2 sender loaded by spark/pd_kv_connector.py inside the vLLM worker.
 *
 * One RC QP to `pd_rdma recvd` on the Mac and one registered buffer. The connector copies each finished decoder
 * block straight from the GPU into pd_tx_buf() and calls pd_tx_block(); the bytes cross in one RDMA WRITE and the
 * Mac writes the block file while prefill continues. Nothing here exits the process: every call returns 0 on success
 * and -1 on failure, and a failed transfer marks the handle dead so the connector can drop it and reconnect.
 */
#include "pd_rdma_common.h"

struct pd_tx {
    struct pd_rdma r;
    int fd;
    struct pd_lineio io;
    struct pd_hello peer;
    int dead;
    int64_t wire_us;
    uint64_t bytes;
};

void pd_tx_close(void *h);

void *pd_tx_open(const char *host, int port, const char *dev, int gid_index, int buf_mib, int mtu)
{
    struct pd_tx *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->fd = -1;
    char line[PD_LINE_MAX];
    if (pd_rdma_open(&t->r, dev, gid_index, (size_t)buf_mib << 20, 0) || pd_qp_create(&t->r)) goto bad;
    t->fd = pd_tcp_connect(host, port);
    if (t->fd < 0) goto bad;
    t->io.fd = t->fd;
    if (pd_handshake(&t->r, t->fd, (uint32_t)mtu, 0, 0, &t->peer)) goto bad;
    if (pd_read_line(&t->io, line, sizeof(line)) || strcmp(line, "READY")) { pd_err("receiver did not send READY"); goto bad; }
    if (!t->peer.arena_addr || !t->peer.rkey) { pd_err("receiver published no arena"); goto bad; }
    return t;
bad:
    pd_tx_close(t);
    return NULL;
}

void *pd_tx_buf(void *h) { return h ? ((struct pd_tx *)h)->r.buf : NULL; }

size_t pd_tx_capacity(void *h)
{
    struct pd_tx *t = h;
    if (!t) return 0;
    return t->peer.arena_len < t->r.buf_len ? (size_t)t->peer.arena_len : t->r.buf_len;
}

static int tx_ok(struct pd_tx *t)
{
    char line[PD_LINE_MAX];
    if (pd_read_line(&t->io, line, sizeof(line))) { t->dead = 1; return pd_err("receiver control channel lost"); }
    if (strcmp(line, "OK")) { t->dead = 1; return pd_err("receiver replied: %s", line); }
    return 0;
}

static int tx_write(struct pd_tx *t, size_t len)
{
    if (t->dead) return -1;
    if (len > pd_tx_capacity(t)) return pd_err("payload %zu exceeds the arena", len);
    int64_t t0 = pd_now_us();
    if (len && pd_rdma_write(&t->r, len, t->peer.arena_addr, t->peer.rkey)) { t->dead = 1; return -1; }
    t->wire_us += pd_now_us() - t0;
    t->bytes += len;
    return 0;
}

/* pd_tx_buf() holds k then v, bf16, each [L, H, B, D]. */
int pd_tx_block(void *h, const char *tag, int index, int L, int H, int B, int D)
{
    struct pd_tx *t = h;
    if (!t || !pd_valid_token(tag, 0) || L <= 0 || H <= 0 || B <= 0 || D <= 0) return -1;
    size_t len = (size_t)L * H * B * D * 4;
    if (tx_write(t, len)) return -1;
    if (pd_send_line(t->fd, "BLK %s %d %d %d %d %d", tag, index, L, H, B, D)) { t->dead = 1; return -1; }
    return tx_ok(t);
}

/* pd_tx_buf() holds the file bytes. */
int pd_tx_file(void *h, const char *tag, const char *name, size_t len)
{
    struct pd_tx *t = h;
    if (!t || !pd_valid_token(tag, 0) || !pd_valid_token(name, 1)) return -1;
    if (tx_write(t, len)) return -1;
    if (pd_send_line(t->fd, "FILE %s %s %zu", tag, name, len)) { t->dead = 1; return -1; }
    return tx_ok(t);
}

/* R4: pd_tx_buf() holds a complete oMLX block file (header + layer tensors); the receiver lands it at
 * <omlx-cache>/<hash[0]>/<hash>.safetensors, where the decoder's disk-index fallback finds it. */
int pd_tx_oblk(void *h, const char *hash_hex, size_t len)
{
    struct pd_tx *t = h;
    if (!t || strlen(hash_hex) != 64 || !pd_valid_token(hash_hex, 0)) return -1;
    if (tx_write(t, len)) return -1;
    if (pd_send_line(t->fd, "OBLK %s %zu", hash_hex, len)) { t->dead = 1; return -1; }
    return tx_ok(t);
}

int pd_tx_done(void *h, const char *tag, const char *status)
{
    struct pd_tx *t = h;
    if (!t || t->dead || !pd_valid_token(tag, 0) || !pd_valid_token(status, 0)) return -1;
    if (pd_send_line(t->fd, "DONE %s %s", tag, status)) { t->dead = 1; return -1; }
    return tx_ok(t);
}

int pd_tx_dead(void *h) { return h ? ((struct pd_tx *)h)->dead : 1; }

double pd_tx_wire_s(void *h) { return h ? ((struct pd_tx *)h)->wire_us / 1e6 : 0.0; }

void pd_tx_close(void *h)
{
    struct pd_tx *t = h;
    if (!t) return;
    if (t->fd >= 0) {
        if (!t->dead) pd_send_line(t->fd, "QUIT");
        close(t->fd);
    }
    pd_rdma_close(&t->r);
    free(t);
}
