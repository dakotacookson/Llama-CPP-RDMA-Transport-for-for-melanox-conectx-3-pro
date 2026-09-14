#include "transport.h"
#include "ggml-impl.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <array>
#  include <time.h>
#  ifndef _WIN32
#    include <poll.h>
#  endif
#  ifdef GGML_RPC_RDMA_APPLE
#    include "transport-apple.h"
#  endif
#endif // GGML_RPC_RDMA

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)

#ifdef GGML_RPC_RDMA
static constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
static constexpr size_t RDMA_CHUNK    = 1024 * 1024;  // 1 MiB: 4x fewer round trips per bulk tensor   // 256 KiB per send/recv (fits default 8 MiB memlock)
static constexpr int    RDMA_RX_DEPTH = 24;
static constexpr int    RDMA_SEND_WINDOW = 8;  // bounded pipeline depth (<= RX_DEPTH)            // pre-posted recv ring: 24 × 256 KiB = 6 MiB

struct rdma_conn {
    struct ibv_context * ctx = nullptr;
    struct ibv_pd * pd  = nullptr;
    struct ibv_cq * scq = nullptr;   // send completions
    struct ibv_cq * rcq = nullptr;   // recv completions
    struct ibv_qp * qp  = nullptr;

    void          * tx_buf[RDMA_SEND_WINDOW] = {};
    struct ibv_mr * tx_mr[RDMA_SEND_WINDOW]  = {};

    void          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    struct ibv_mr * rx_mr  = nullptr;
    int             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(int i) const {
        return static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    bool post_rx(int i) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        struct ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        return ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        for (auto & m : tx_mr) if (m) ibv_dereg_mr(m);
        if (rx_mr) ibv_dereg_mr(rx_mr);
        for (auto & b : tx_buf) if (b) free(b);
        free(rx_buf);
        if (qp)  ibv_destroy_qp(qp);
        if (scq) ibv_destroy_cq(scq);
        if (rcq) ibv_destroy_cq(rcq);
        if (pd)  ibv_dealloc_pd(pd);
        if (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
struct rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    int      gid_idx = 0;
    enum ibv_mtu path_mtu = IBV_MTU_1024;
};

struct rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(sizeof(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

struct socket_t::impl {
    impl(sockfd_t fd) : use_rdma(false), fd(fd) {}
    ~impl();
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    bool flush();
    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  else
    bool rdma_probe();
    bool rdma_send(const void * data, size_t size);
    bool rdma_recv(void * data, size_t size);
    bool tcp_peer_closed();
    bool rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid);
    bool rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc);

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    bool     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
    if (fd != INVALID_SOCKET) closesocket(this->fd);
#else
    if (fd >= 0) close(this->fd);
#endif
}

#ifdef GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        return std::nullopt;
    }
    rdma_gid_t target = {};
    if (addr.ss_family == AF_INET) {
        const auto * a = reinterpret_cast<const sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        return target;
    }
    if (addr.ss_family == AF_INET6) {
        const auto * a = reinterpret_cast<const sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        return target;
    }
    return std::nullopt;
}

// Format an RDMA GID (16 bytes) as a colon-separated hex string for diagnostics.
static std::string rdma_gid_to_string(const uint8_t * gid) {
    char buf[3 * RDMA_GID_SIZE]; // 16 * 2 hex digits + 15 separators + NUL
    char * p = buf;
    for (size_t i = 0; i < RDMA_GID_SIZE; i++) {
        p += sprintf(p, "%02x%s", gid[i], i + 1 < RDMA_GID_SIZE ? ":" : "");
    }
    return std::string(buf);
}

static const char * rdma_gid_type_str(uint32_t type) {
    switch (type) {
        case IBV_GID_TYPE_IB:       return "IB";
        case IBV_GID_TYPE_ROCE_V1:  return "RoCEv1";
        case IBV_GID_TYPE_ROCE_V2:  return "RoCEv2";
        default:                    return "unknown";
    }
}

#ifndef GGML_RPC_RDMA_APPLE

bool socket_t::impl::tcp_peer_closed() {
    if (fd < 0) return false;
#ifndef _WIN32
    struct pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    int r = poll(&pfd, 1, 0);
    return r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#else
    return false;
#endif
}

bool socket_t::impl::rdma_probe() {
    const char * dev_env = std::getenv("GGML_RDMA_DEV");
    const char * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    if (!target_gid) {
        return false;
    }
    GGML_LOG_INFO("RDMA probe: target GID (local addr) = %s\n", rdma_gid_to_string(target_gid->data()));
    if (dev_env || gid_env) {
        GGML_LOG_INFO("RDMA probe: env GGML_RDMA_DEV=%s GGML_RDMA_GID=%s\n",
                      dev_env ? dev_env : "(unset)", gid_env ? gid_env : "(unset)");
    }

    int num_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&num_devs);
    if (!devs || num_devs == 0) return false;

    ibv_context * ibctx = nullptr;
    const char * matched_dev = nullptr;
    int gid_idx = gid_env ? atoi(gid_env) : -1;
    int gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB
    uint8_t ib_port = 0;

    for (int d = 0; d < num_devs; d++) {
        const char * dn = ibv_get_device_name(devs[d]);
        if (dev_env && strcmp(dev_env, dn) != 0) continue;

        ibv_context * ctx = ibv_open_device(devs[d]);
        if (!ctx) continue;

        // multi-port cards (ConnectX-3 Pro): scan ports 1..2 for the ACTIVE one
        ibv_port_attr pa;
        for (uint8_t prt = 1; prt <= 2; prt++) {
            if (ibv_query_port(ctx, prt, &pa) != 0) continue;
            if (pa.state == IBV_PORT_ACTIVE) {
                ib_port = prt;
                break;
            }
        }
        if (!ib_port) { ibv_close_device(ctx); continue; }

        int found_gid = gid_idx;
        int found_version = IBV_GID_TYPE_IB;
        if (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            int v2_idx = -1;
            for (int i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                if (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) continue;
                const bool matches = memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) == 0;
                GGML_LOG_INFO("RDMA probe: device %s port %u GID[%d] type=%s gid=%s%s\n",
                              dn, ib_port, i, rdma_gid_type_str(entry.gid_type),
                              rdma_gid_to_string(entry.gid.raw),
                              matches ? " [MATCH]" : "");
                if (!matches) continue;
                if (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                }
            }
            if (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } else {
                // RoCEv2-only: no v1 fallback (raw-Ethernet path type mismatches
                // the QP/AH semantics and silently drops packets on routed paths)
                GGML_LOG_INFO("RDMA probe: device %s port %u: no RoCEv2 GID matches local addr, skipping\n",
                              dn, rdma_gid_to_string(target_gid->data()));
                ibv_close_device(ctx);
                continue;
            }
        } else {
            // GGML_RDMA_GID hint: honor it if it's a valid RoCEv2 entry; otherwise
            // (stale index after MTU/GID-table shifts, wrong type) fall through to
            // auto-select — which is RoCEv2-only anyway. Never silently use a
            // RoCEv1 entry: mismatched encapsulation drops packets silently.
            ibv_gid_entry entry = {};
            if (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0 &&
                entry.gid_type == IBV_GID_TYPE_ROCE_V2) {
                found_version = IBV_GID_TYPE_ROCE_V2;
            } else {
                GGML_LOG_INFO("RDMA probe: GGML_RDMA_GID=%d not a valid RoCEv2 entry on "
                              "%s port %u — auto-selecting RoCEv2 GID by local address\n",
                              found_gid, dn, ib_port);
                found_gid = -1;  // force the auto-select branch below
                int v2_idx = -1;
                for (int i = 0; i < pa.gid_tbl_len; i++) {
                    ibv_gid_entry e = {};
                    if (ibv_query_gid_ex(ctx, ib_port, i, &e, 0) != 0) continue;
                    if (e.gid_type == IBV_GID_TYPE_ROCE_V2) {
                        GGML_LOG_INFO("RDMA probe: candidate GID[%d] %s\n", i,
                                      rdma_gid_to_string(e.gid.raw).c_str());
                        if (memcmp(e.gid.raw, target_gid->data(), RDMA_GID_SIZE) == 0) {
                            v2_idx = i;
                            break;
                        }
                    }
                }
                if (v2_idx >= 0) {
                    found_gid = v2_idx;
                    found_version = IBV_GID_TYPE_ROCE_V2;
                } else {
                    GGML_LOG_INFO("RDMA probe: device %s port %u: no RoCEv2 GID matches local addr, skipping\n",
                                  dn, rdma_gid_to_string(target_gid->data()));
                    ibv_close_device(ctx);
                    continue;
                }
            }
        }
        if (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            break;
        }
        GGML_LOG_INFO("RDMA probe: device %s port %u: no GID matching local addr %s, skipping\n",
                      dn, ib_port, rdma_gid_to_string(target_gid->data()));
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    if (!ibctx) return false;
    // hard guarantee: RoCEv2 only, both sides, always — a mismatch here would build
    // an AH/QP of the wrong encapsulation and silently drop packets
    if (gid_version != IBV_GID_TYPE_ROCE_V2) {
        GGML_LOG_ERROR("RDMA probe: selected GID %d on %s is not RoCEv2 (type=%d) — aborting probe\n",
                       gid_idx, matched_dev, gid_version);
        ibv_close_device(ibctx);
        return false;
    }

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    if (!rdma->pd) return false;

    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, nullptr, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, nullptr, 0);
    if (!rdma->scq || !rdma->rcq) return false;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    if (!rdma->qp) return false;
    rdma->max_inline = qia.cap.max_inline_data;

    for (int i = 0; i < RDMA_SEND_WINDOW; i++) {
        rdma->tx_buf[i] = aligned_alloc(4096, RDMA_CHUNK);
        if (!rdma->tx_buf[i]) return false;
        rdma->tx_mr[i] = ibv_reg_mr(rdma->pd, rdma->tx_buf[i], RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
        if (!rdma->tx_mr[i]) return false;
    }
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    if (!rdma->rx_buf) return false;

    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!rdma->rx_mr) return false;

    ibv_gid local_gid;
    if (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) return false;

    rdma_local.qpn = rdma->qp->qp_num;
    rdma_local.psn = rdma->qp->qp_num & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    const char * ver_str = "";
    if (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } else if (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    GGML_LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u mtu=%d (discovered from HCA)\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline,
                  128 << rdma_local.path_mtu);
    return true;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
bool socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid) {
    // RESET -> INIT
    {
        struct ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_num        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            GGML_LOG_ERROR("RDMA activate: INIT failed: %s (port=%u)\n",
                           strerror(errno), rdma_local.ib_port);
            return false;
        }
    }

    for (int i = 0; i < RDMA_RX_DEPTH; i++) {
        if (!rdma->post_rx(i)) return false;
    }

    // INIT -> RTR
    {
        struct ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_num        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_num   = rdma_local.ib_port;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            GGML_LOG_ERROR("RDMA activate: RTR failed: %s (port=%u gid_idx=%d dgid=%s "
                           "dest_qpn=%u rq_psn=%u mtu=%d)\n",
                           strerror(errno), rdma_local.ib_port, rdma_local.gid_idx,
                           rdma_gid_to_string(remote_gid).c_str(), remote_qpn, remote_psn,
                           128 << rdma_local.path_mtu);
            return false;
        }
    }

    // RTR -> RTS
    {
        struct ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            GGML_LOG_ERROR("RDMA activate: RTS failed: %s (sq_psn=%u)\n",
                           strerror(errno), rdma_local.psn);
            return false;
        }
    }

    GGML_LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    return true;
}

bool socket_t::impl::rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc) {
    for (uint64_t s = 0; ; s++) {
        int n = ibv_poll_cq(cq, 1, wc);
        if (n > 0) {
            if (wc->status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            return wc->status == IBV_WC_SUCCESS;
        }
        if (n < 0) return false;
        if ((s & 0xFFFFF) == 0 && s > 0) {
            if (tcp_peer_closed()) {
                return false;
            }
        }
    }
}

bool socket_t::impl::rdma_send(const void * data, size_t size) {
    rdma_conn * c = rdma.get();
    const uint8_t * src = (const uint8_t *)data;
    size_t rem = size;
    int outstanding = 0;
    uint64_t buf_idx = 0;
    // bounded pipeline: up to RDMA_SEND_WINDOW sends in flight. RC delivery is
    // ordered, so the peer's recv ring always sees chunks in post order; the recv
    // ring (RDMA_RX_DEPTH=24) is deeper than the window (8), so RNR is impossible.
    // Small sends stay inline+lockstep-ish: the window only opens up the wait for
    // bulk transfers — semantics unchanged (every send completes; buffer reused
    // only after its own completion).
    while (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        struct ibv_sge sge = {};
        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        const bool bulk = chunk > c->max_inline;
        if (bulk) {
            // window: before overwriting buffer[buf_idx % WINDOW] (in flight since
            // WINDOW sends ago), drain completions until it (and all older) arrived
            while (outstanding >= RDMA_SEND_WINDOW) {
                struct ibv_wc wcd;
                int n = ibv_poll_cq(c->scq, 1, &wcd);
                if (n > 0) {
                    if (wcd.status != IBV_WC_SUCCESS) {
                        GGML_LOG_ERROR("RDMA send CQ error (window drain): status=%d\n", wcd.status);
                        return false;
                    }
                    outstanding--;
                } else if (n < 0) {
                    return false;
                } else {
                    struct timespec ts = {0, 1000};  // 1µs spin
                    nanosleep(&ts, nullptr);
                }
            }
            const int bi = (int)(buf_idx % RDMA_SEND_WINDOW);
            memcpy(c->tx_buf[bi], src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf[bi];
            sge.length = chunk;
            sge.lkey   = c->tx_mr[bi]->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        } else {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        }

        if (ibv_post_send(c->qp, &wr, &bad) != 0) return false;
        outstanding++;           // every send is signaled — count it
        if (bulk) buf_idx++;     // ring slot advances only for bulk (buffered) sends

        src += chunk;
        rem -= chunk;
    }
    // drain the tail
    struct ibv_wc wc;
    while (outstanding > 0) {
        int n = ibv_poll_cq(c->scq, 1, &wc);
        if (n > 0) {
            if (wc.status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RDMA send CQ error (tail): status=%d\n", wc.status);
                return false;
            }
            outstanding--;
        } else if (n < 0) {
            return false;
        } else {
            struct timespec ts = {0, 1000};
            nanosleep(&ts, nullptr);
        }
    }
    return true;
}

bool socket_t::impl::rdma_recv(void * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    while (rem > 0) {
        struct ibv_wc wc;
        if (!rdma_poll(c->rcq, &wc)) return false;

        int slot = (int)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        if (!c->post_rx(slot)) return false;

        dst += got;
        rem -= got;
    }
    return true;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

bool socket_t::impl::send_data(const void * data, size_t size) {
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->send(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) {
        return rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (const char *)data + bytes_sent, size_to_send, 0);
        if (n < 0) {
            GGML_LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            return false;
        }
        bytes_sent += (size_t)n;
    }
    return true;
}

bool socket_t::impl::recv_data(void * data, size_t size) {
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->recv(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    if (use_rdma) {
        return rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (char *)data + bytes_recv, size_to_recv, 0);
        if (n < 0) {
            GGML_LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            return false;
        }
        if (n == 0) {
            LOG_DBG("recv returned 0 (peer closed?)\n");
            return false;
        }
        bytes_recv += (size_t)n;
    }
    return true;
}

void socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    if (std::getenv("GGML_RPC_NO_RDMA")) {
        return;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    if (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
#  else
    rdma_local = {};
    if (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, sizeof(rc));
    } else {
        rdma.reset();
    }
#  endif
#endif // GGML_RPC_RDMA
}

void socket_t::impl::update_caps(const uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    bool remote_rdma = false;
    for (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    if (!rdma || !remote_rdma) {
        rdma.reset();
        return;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    bool activated = rdma->activate(remote_caps);
#  else
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, sizeof(rc));
    bool activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    if (activated) {
        use_rdma = true;
    } else {
        GGML_LOG_ERROR("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#else
    (void)remote_caps;
#endif // GGML_RPC_RDMA
}

bool socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    if (use_rdma) {
        return rdma->flush();
    }
#endif
    return true;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = default;

bool socket_t::send_data(const void * data, size_t size) {
    return pimpl->send_data(data, size);
}

bool socket_t::recv_data(void * data, size_t size) {
    return pimpl->recv_data(data, size);
}

bool socket_t::flush() {
    return pimpl->flush();
}

void socket_t::get_caps(uint8_t * local_caps) {
    return pimpl->get_caps(local_caps);
}

void socket_t::update_caps(const uint8_t * remote_caps) {
    return pimpl->update_caps(remote_caps);
}

static bool is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    return sockfd != INVALID_SOCKET;
#else
    return sockfd >= 0;
#endif
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, NULL, NULL);
    if (!is_valid_fd(client_socket_fd)) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        GGML_LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        GGML_LOG_ERROR("Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        GGML_LOG_ERROR("Invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (!is_valid_fd(sockfd)) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        GGML_LOG_ERROR("Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        GGML_LOG_ERROR("Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (::connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    return socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
static std::mutex g_rpc_transport_mu;
static bool g_rpc_transport_wsa_started = false;
#endif

bool rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (g_rpc_transport_wsa_started) {
        return true;
    }
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        return false;
    }
    g_rpc_transport_wsa_started = true;
    return true;
#else
    return true;
#endif
}

void rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    if (!g_rpc_transport_wsa_started) {
        return;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = false;
#endif
}
