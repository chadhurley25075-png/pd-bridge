/* pd_rdma_common.h — RoCEv2 plumbing shared by pd_rdma (serve / client / recvd) and libpd_rdma_tx (the vLLM
 * connector's sender). See docs/RDMA.md, R1 and R2.
 *
 * Every function reports failure by return value and never exits: the sender runs inside the vLLM engine process,
 * where exit() would take the engine down with a failed RDMA write. The CLI decides what is fatal.
 */
#ifndef PD_RDMA_COMMON_H
#define PD_RDMA_COMMON_H

#ifndef __APPLE__
/* glibc gates scandir/pread/getaddrinfo behind feature macros; on Darwin _POSIX_C_SOURCE would instead hide the
 * BSD types <netinet/tcp.h> needs, so it is defined for Linux only. */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PD_MAGIC "PDRDMA1"          /* 7 chars + NUL = 8 bytes on the wire */
#define PD_HELLO_BYTES 56
#define PD_WRID_WRITE 0x5044575200000000ULL
#define PD_LINE_MAX 1024
#define PD_NAME_MAX 128

/* ------------------------------------------------------------------ utilities */
static inline int64_t pd_now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000 + t.tv_nsec / 1000;
}

static inline int pd_err(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("pd_rdma: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    return -1;
}

static inline int pd_errno(const char *what)
{
    return pd_err("%s failed (errno %d: %s)", what, errno, strerror(errno));
}

static inline void pd_put32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static inline void pd_put64(uint8_t *p, uint64_t v) { pd_put32(p, (uint32_t)(v >> 32)); pd_put32(p + 4, (uint32_t)v); }
static inline uint32_t pd_get32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static inline uint64_t pd_get64(const uint8_t *p) { return (uint64_t)pd_get32(p) << 32 | pd_get32(p + 4); }

static inline int pd_write_full(int fd, const void *buf, size_t len)
{
    const uint8_t *p = buf;
    while (len) {
        ssize_t n = write(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static inline int pd_read_full(int fd, void *buf, size_t len)
{
    uint8_t *p = buf;
    while (len) {
        ssize_t n = read(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

struct pd_lineio { int fd; size_t len; char buf[4096]; };

static inline int pd_read_line(struct pd_lineio *io, char *out, size_t cap)
{
    for (;;) {
        char *nl = memchr(io->buf, '\n', io->len);
        if (nl) {
            size_t n = (size_t)(nl - io->buf);
            if (n >= cap) return -1;
            memcpy(out, io->buf, n);
            out[n] = 0;
            io->len -= n + 1;
            memmove(io->buf, nl + 1, io->len);
            return 0;
        }
        if (io->len == sizeof(io->buf)) return -1;
        ssize_t r = read(io->fd, io->buf + io->len, sizeof(io->buf) - io->len);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        io->len += (size_t)r;
    }
}

static inline int pd_send_line(int fd, const char *fmt, ...)
{
    char line[PD_LINE_MAX];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line) - 1, fmt, ap);
    va_end(ap);
    if (n < 0 || n >= (int)sizeof(line) - 1) return -1;
    line[n++] = '\n';
    return pd_write_full(fd, line, (size_t)n);
}

/* Tags and file names travel on the wire and become paths: letters, digits, _ -, and '.' where allowed. */
static inline int pd_valid_token(const char *s, int allow_dot)
{
    if (!*s || strlen(s) >= PD_NAME_MAX || s[0] == '.') return 0;
    for (; *s; s++)
        if (!((*s >= 'A' && *s <= 'Z') || (*s >= 'a' && *s <= 'z') || (*s >= '0' && *s <= '9') ||
              *s == '_' || *s == '-' || (allow_dot && *s == '.'))) return 0;
    return 1;
}

/* ------------------------------------------------------------------ TCP control channel */
static inline int pd_tcp_connect(const char *host, int port)
{
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char port_s[8];
    snprintf(port_s, sizeof(port_s), "%d", port);
    if (getaddrinfo(host, port_s, &hints, &res) || !res) return pd_err("cannot resolve %s", host);
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd >= 0 && connect(fd, res->ai_addr, res->ai_addrlen)) { close(fd); fd = -1; }
    freeaddrinfo(res);
    if (fd < 0) return pd_err("cannot reach %s:%d", host, port);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

static inline int pd_tcp_listen(const char *bind_host, int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return pd_errno("socket");
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, bind_host, &addr.sin_addr) != 1) { close(fd); return pd_err("bad bind address %s", bind_host); }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) || listen(fd, 4)) { close(fd); return pd_errno("bind/listen"); }
    return fd;
}

/* ------------------------------------------------------------------ RDMA */
struct pd_hello {
    uint32_t qpn, psn, mtu, rkey;
    uint8_t gid[16];
    uint64_t arena_addr, arena_len;
};

static inline void pd_hello_pack(const struct pd_hello *h, uint8_t *b)
{
    memset(b, 0, PD_HELLO_BYTES);
    memcpy(b, PD_MAGIC, 8);
    pd_put32(b + 8, h->qpn);
    pd_put32(b + 12, h->psn);
    memcpy(b + 16, h->gid, 16);
    pd_put64(b + 32, h->arena_addr);
    pd_put32(b + 40, h->rkey);
    pd_put32(b + 44, h->mtu);
    pd_put64(b + 48, h->arena_len);
}

static inline int pd_hello_unpack(const uint8_t *b, struct pd_hello *h)
{
    if (memcmp(b, PD_MAGIC, 8)) return -1;
    h->qpn = pd_get32(b + 8);
    h->psn = pd_get32(b + 12);
    memcpy(h->gid, b + 16, 16);
    h->arena_addr = pd_get64(b + 32);
    h->rkey = pd_get32(b + 40);
    h->mtu = pd_get32(b + 44);
    h->arena_len = pd_get64(b + 48);
    return 0;
}

struct pd_rdma {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    uint8_t *buf;
    size_t buf_len;
    int port, gid_index;
    union ibv_gid gid;
    uint32_t psn;
    uint64_t seq;
};

#ifdef __APPLE__
static inline int pd_parse_mac(const char *s, uint8_t out[6])
{
    unsigned v[6];
    if (!s || sscanf(s, "%x:%x:%x:%x:%x:%x", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) return -1;
    for (int i = 0; i < 6; i++) {
        if (v[i] > 0xff) return -1;
        out[i] = (uint8_t)v[i];
    }
    return 0;
}

/* The DEXT has no netif/ARP for this port: the caller supplies the local GID and both MACs explicitly. */
static inline int pd_roce_configure(struct pd_rdma *r)
{
    struct ibv_mlx5_roce_config c;
    memset(&c, 0, sizeof(c));
    const char *ip = getenv("MELONDMA_LOCAL_IP");
    if (!ip || pd_parse_mac(getenv("MELONDMA_LOCAL_MAC"), c.local_mac) ||
        pd_parse_mac(getenv("MELONDMA_REMOTE_MAC"), c.peer_mac))
        return pd_err("set MELONDMA_LOCAL_IP, MELONDMA_LOCAL_MAC and MELONDMA_REMOTE_MAC");
    c.local_gid.raw[10] = c.local_gid.raw[11] = 0xff;   /* IPv4-mapped GID, l3_type 0 */
    if (inet_pton(AF_INET, ip, c.local_gid.raw + 12) != 1) return pd_err("MELONDMA_LOCAL_IP=%s is not IPv4", ip);
    c.hop_limit = 1;
    int rc = ibv_mlx5_configure_roce(r->ctx, &c);
    if (rc) { errno = rc; return pd_errno("ibv_mlx5_configure_roce"); }
    r->gid = c.local_gid;
    return 0;
}
#endif

static inline void pd_rdma_close(struct pd_rdma *r)
{
    if (r->qp) ibv_destroy_qp(r->qp);
    if (r->mr) ibv_dereg_mr(r->mr);
    if (r->cq) ibv_destroy_cq(r->cq);
    if (r->pd) ibv_dealloc_pd(r->pd);
    if (r->ctx) ibv_close_device(r->ctx);
    free(r->buf);
    memset(r, 0, sizeof(*r));
}

/* Device, PD, CQ and ONE registered buffer for the life of the process (the DEXT caps a client at 512 MiB pinned). */
static inline int pd_rdma_open(struct pd_rdma *r, const char *dev_name, int gid_index, size_t buf_len, int remote_write)
{
    memset(r, 0, sizeof(*r));
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    struct ibv_device *dev = NULL;
    for (int i = 0; list && i < n; i++)
        if (!dev_name || !strcmp(ibv_get_device_name(list[i]), dev_name)) { dev = list[i]; break; }
    if (!dev) {
        if (list) ibv_free_device_list(list);
        return pd_err("no RDMA device matches %s", dev_name ? dev_name : "<first>");
    }
    r->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (!r->ctx) return pd_errno("ibv_open_device");
    r->port = 1;
    r->gid_index = gid_index;
#ifdef __APPLE__
    if (pd_roce_configure(r)) { pd_rdma_close(r); return -1; }
#else
    if (ibv_query_gid(r->ctx, (uint8_t)r->port, gid_index, &r->gid)) { pd_rdma_close(r); return pd_errno("ibv_query_gid"); }
#endif
    r->pd = ibv_alloc_pd(r->ctx);
    r->cq = r->pd ? ibv_create_cq(r->ctx, 64, NULL, NULL, 0) : NULL;
    if (!r->pd || !r->cq) { pd_rdma_close(r); return pd_errno("ibv_alloc_pd/ibv_create_cq"); }
    void *mem = NULL;
    if (posix_memalign(&mem, 4096, buf_len)) { pd_rdma_close(r); return pd_errno("posix_memalign"); }
    memset(mem, 0, buf_len);   /* touch every page before pinning */
    r->buf = mem;
    r->buf_len = buf_len;
    int access = IBV_ACCESS_LOCAL_WRITE | (remote_write ? IBV_ACCESS_REMOTE_WRITE : 0);
    r->mr = ibv_reg_mr(r->pd, r->buf, buf_len, access);
    if (!r->mr) { pd_rdma_close(r); return pd_errno("ibv_reg_mr (check `ulimit -l`)"); }
    return 0;
}

static inline void pd_qp_destroy(struct pd_rdma *r)
{
    if (r->qp) ibv_destroy_qp(r->qp);
    r->qp = NULL;
}

static inline int pd_qp_create(struct pd_rdma *r)
{
    struct ibv_qp_init_attr ia;
    memset(&ia, 0, sizeof(ia));
    ia.send_cq = ia.recv_cq = r->cq;
    ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = ia.cap.max_recv_wr = 16;
    ia.cap.max_send_sge = ia.cap.max_recv_sge = 1;
    r->qp = ibv_create_qp(r->pd, &ia);
    if (!r->qp) return pd_errno("ibv_create_qp");
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = (uint8_t)r->port;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        pd_qp_destroy(r);
        return pd_errno("ibv_modify_qp RESET->INIT");
    }
    r->psn = ((uint32_t)getpid() * 2654435761u + (uint32_t)pd_now_us()) & 0xffffff;
    r->seq = 0;
    return 0;
}

static inline enum ibv_mtu pd_mtu_enum(uint32_t bytes)
{
    return bytes >= 4096 ? IBV_MTU_4096 : bytes >= 2048 ? IBV_MTU_2048 : IBV_MTU_1024;
}

/* Same transitions and attributes as melon_group.cpp, which is proven on this Mac <-> Spark pair. */
static inline int pd_qp_connect(struct pd_rdma *r, const struct pd_hello *peer, uint32_t mtu)
{
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = pd_mtu_enum(mtu);
    a.dest_qp_num = peer->qpn;
    a.rq_psn = peer->psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1;
    a.ah_attr.port_num = (uint8_t)r->port;
    memcpy(a.ah_attr.grh.dgid.raw, peer->gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index;
    a.ah_attr.grh.hop_limit = 1;
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                     IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        return pd_errno("ibv_modify_qp INIT->RTR");
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = r->psn;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.max_rd_atomic = 1;
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                     IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC))
        return pd_errno("ibv_modify_qp RTR->RTS");
    return 0;
}

/* One WRITE of buf[0:len] into the peer arena, one signaled completion, busy-polled (bulk path). */
static inline int pd_rdma_write(struct pd_rdma *r, size_t len, uint64_t raddr, uint32_t rkey)
{
    struct ibv_sge sge;
    sge.addr = (uintptr_t)r->buf;
    sge.length = (uint32_t)len;
    sge.lkey = r->mr->lkey;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = PD_WRID_WRITE | (++r->seq & 0xffffffffULL);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey = rkey;
    struct ibv_send_wr *bad = NULL;
    if (ibv_post_send(r->qp, &wr, &bad)) return pd_err("ibv_post_send failed");
    struct ibv_wc wc;
    for (;;) {
        int n = ibv_poll_cq(r->cq, 1, &wc);
        if (n < 0) return pd_err("ibv_poll_cq failed");
        if (n == 0) continue;
        if (wc.status != IBV_WC_SUCCESS)
            return pd_err("CQE %s (vendor_err=%u)", ibv_wc_status_str(wc.status), wc.vendor_err);
        if (wc.wr_id == wr.wr_id) return 0;
    }
}

/* send_first: the accepting side speaks first. publish_arena: this side receives WRITEs into r->buf. */
static inline int pd_handshake(struct pd_rdma *r, int fd, uint32_t my_mtu, int send_first, int publish_arena,
                               struct pd_hello *peer)
{
    struct pd_hello me;
    memset(&me, 0, sizeof(me));
    me.qpn = r->qp->qp_num;
    me.psn = r->psn;
    me.mtu = my_mtu;
    memcpy(me.gid, r->gid.raw, 16);
    if (publish_arena) {
        me.arena_addr = (uintptr_t)r->buf;
        me.rkey = r->mr->rkey;
        me.arena_len = r->buf_len;
    }
    uint8_t mine[PD_HELLO_BYTES], theirs[PD_HELLO_BYTES];
    pd_hello_pack(&me, mine);
    if (send_first) {
        if (pd_write_full(fd, mine, sizeof(mine)) || pd_read_full(fd, theirs, sizeof(theirs))) return pd_err("hello I/O failed");
    } else {
        if (pd_read_full(fd, theirs, sizeof(theirs)) || pd_write_full(fd, mine, sizeof(mine))) return pd_err("hello I/O failed");
    }
    if (pd_hello_unpack(theirs, peer)) return pd_err("bad hello magic");
    return pd_qp_connect(r, peer, peer->mtu < my_mtu ? peer->mtu : my_mtu);
}

#endif /* PD_RDMA_COMMON_H */
