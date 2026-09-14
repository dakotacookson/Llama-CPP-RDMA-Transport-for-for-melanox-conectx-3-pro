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
#  include <fcntl.h>
#endif
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <rdma/rdma_cma.h>
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
static constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
static struct rdma_event_channel * cm_create_channel_nb();
static constexpr int    RDMA_RX_DEPTH = 1024;
static constexpr int    TX_RING_DEPTH  = 16;            // pipelined send buffers          // pre-posted recv ring: 1024 × 256 KiB = 256 MiB

// Adaptive send pacing (credit-flow-control stand-in): if the peer's recv
// drain lags our send progress by more than half the ring depth, back off
// exponentially (10us → 100ms) so the peer's RQ never empties regardless of
// NIC speed, PCIe width, or consumer stall length. Resets on catch-up.
static uint32_t g_pacing_delay_us = 0;
static constexpr uint32_t PACING_MIN_US = 10;
static constexpr uint32_t PACING_MAX_US = 100 * 1000;
// (must absorb a full large tensor streaming at line rate while the app is
//  busy in CUDA work between drains; 24 slots = 6 MiB wedged on 5.9 MiB
//  tensors with RNR retry-counter-exceeded)

struct rdma_conn {
    struct ibv_context * ctx = nullptr;
    struct ibv_pd * pd  = nullptr;
    struct ibv_cq * scq = nullptr;   // send completions
    struct ibv_cq * rcq = nullptr;   // recv completions
    struct ibv_qp * qp  = nullptr;
    // false when the context (and QP) lifetime belongs to a rdma_cm_id:
    // rdma_destroy_id closes the context and rdma_destroy_qp frees the QP,
    // so the destructor must skip both.
    bool ctx_owned = true;

    // TX ring: multiple buffers so rdma_send can pipeline posts without
    // waiting for each completion (single buffer = lockstep, kills prefill)
    // tx_outstanding: sends posted but not yet completed (for ring-wrap safety)
    uint64_t tx_outstanding = 0;
    // legacy single-buffer (non-CM raw path)
    void          * tx_buf = nullptr;
    struct ibv_mr * tx_mr  = nullptr;
    void          * tx_bufs[TX_RING_DEPTH] = {};
    struct ibv_mr * tx_mrs[TX_RING_DEPTH] = {};
    int             tx_head = 0;
    void          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    struct ibv_mr * rx_mr  = nullptr;
    int             rx_head = 0;

    uint32_t        max_inline = 0;

    // lifetime counters for RNR diagnosis (logged on transport errors)
    uint64_t        n_send_posted = 0;
    uint64_t        n_send_done = 0;
    uint64_t        n_recv_done = 0;
    uint64_t        n_recv_reposted = 0;

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
        for (int i = 0; i < TX_RING_DEPTH; i++) {
            if (tx_mrs[i]) ibv_dereg_mr(tx_mrs[i]);
            if (tx_bufs[i]) free(tx_bufs[i]);
        }
        if (rx_mr) ibv_dereg_mr(rx_mr);
        free(rx_buf);
        // QP destruction: in CM mode (ctx_owned=false) the QP belongs to the
        // cm_id and rdma_destroy_qp(cm_id) in the impl destructor owns it.
        // Destroying it here as well would be a double-free (and the partial-
        // failure path leaves id->qp dangling while this struct is being
        // torn down first). Skip when CM owns the connection.
        if (qp && ctx_owned) ibv_destroy_qp(qp);
        if (scq) ibv_destroy_cq(scq);
        if (rcq) ibv_destroy_cq(rcq);
        if (pd)  ibv_dealloc_pd(pd);
        if (ctx && ctx_owned) ibv_close_device(ctx);
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
    sockaddr_storage src_addr = {};  // local addr of the TCP socket (for rdma_cm binding)
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
    // native rdma_cm connection state (GGML_RDMA_USE_CM=1): the QP is owned by
    // the cm_id (rdma_destroy_qp), NOT by rdma->qp — the destructor must null
    // rdma->qp first to avoid a double destroy. cm_ch is owned only when
    // cm_owns_ch (client side); accepted server connections share the
    // listener's channel.
    struct rdma_cm_id *        cm_id = nullptr;
    struct rdma_event_channel *cm_ch = nullptr;
    bool                       cm_owns_ch = false;
    bool                       cm_connected = false;
#  endif
#endif // GGML_RPC_RDMA
    bool     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE) && !defined(_WIN32)
    if (cm_id) {
        // QP was created via rdma_create_qp on the cm_id: destroy through the
        // CM so the id stays consistent, and detach it from rdma_conn first.
        // (id->qp is null when setup failed before QP creation — e.g. addr
        // resolve failure — and rdma_destroy_qp would crash on it.)
        if (cm_connected) rdma_disconnect(cm_id);
        if (cm_id->qp) rdma_destroy_qp(cm_id);
        if (rdma) rdma->qp = nullptr;
        rdma_destroy_id(cm_id);
        cm_id = nullptr;
    }
    if (cm_ch && cm_owns_ch) {
        rdma_destroy_event_channel(cm_ch);
        cm_ch = nullptr;
    }
#endif
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

static const char * rdma_gid_type_str(enum ibv_gid_type type) {
    switch (type) {
        case IBV_GID_TYPE_IB:       return "IB";
        case IBV_GID_TYPE_ROCE_V1:  return "RoCEv1";
        case IBV_GID_TYPE_ROCE_V2:  return "RoCEv2";
        default:                    return "unknown";
    }
}

#ifndef GGML_RPC_RDMA_APPLE

bool socket_t::impl::tcp_peer_closed() {
    // CM-mode connections have no TCP socket (fd=-1). Detect a dead peer by
    // checking the CM event channel for DISCONNECTED/TIMEDOUT events instead.
#if defined(GGML_RPC_RDMA) || defined(GGML_RPC_RDMA_APPLE)
    if (cm_connected && cm_ch && cm_id) {
        // non-blocking: pull any pending CM event
        struct rdma_cm_event * ev = nullptr;
        if (rdma_get_cm_event(cm_ch, &ev) == 0) {
            enum rdma_cm_event_type evtype = ev->event;
            rdma_ack_cm_event(ev);
            // DISCONNECTED / DEVICE_REMOVAL / TIMED_OUT = peer is gone
            if (evtype == RDMA_CM_EVENT_DISCONNECTED ||
                evtype == RDMA_CM_EVENT_DEVICE_REMOVAL ||
                evtype == RDMA_CM_EVENT_TIMEWAIT_EXIT) {
                GGML_LOG_ERROR("RDMA cm: peer disconnected (event=%d)\n", (int)evtype);
                return true;
            }
            // non-fatal events (ESTABLISHED etc.) — ignore
            return false;
        }
        // no pending event — but rdma_get_cm_event on a non-blocking fd returns
        // immediately; EAGAIN means no event, peer still connected
        return false;
    }
#endif
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
    std::string target_gid_str = rdma_gid_to_string(target_gid->data());
    GGML_LOG_INFO("RDMA probe: target GID (local addr) = %s\n", target_gid_str.c_str());
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

    for (int d = 0; d < num_devs; d++) {
        const char * dn = ibv_get_device_name(devs[d]);
        if (dev_env && strcmp(dev_env, dn) != 0) continue;

        ibv_context * ctx = ibv_open_device(devs[d]);
        if (!ctx) continue;

        // Find the ACTIVE port on this device (multi-port cards: the RoCE link may
        // be on port 2; GID tables exist for all addresses on every port even when
        // a port is DOWN, so hardcoding port 1 matches GIDs that can never work).
        uint8_t ib_port = 0;
        struct ibv_port_attr pa = {};
        for (uint8_t prt = 1; prt <= 2; prt++) {
            struct ibv_port_attr pap;
            if (ibv_query_port(ctx, prt, &pap)) continue;
            if (pap.state == IBV_PORT_ACTIVE) {
                ib_port = prt;
                pa = pap;
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
            int v1_idx = -1;
            for (int i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                if (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) continue;
                const bool matches = memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) == 0;
                std::string gid_type_str = rdma_gid_type_str((enum ibv_gid_type)entry.gid_type);
                std::string gid_str = rdma_gid_to_string(entry.gid.raw);
                GGML_LOG_INFO("RDMA probe: device %s port %u GID[%d] type=%s gid=%s%s\n",
                              dn, ib_port, i, gid_type_str.c_str(), gid_str.c_str(),
                              matches ? " [MATCH]" : "");
                if (!matches) continue;
                if (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } else if (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            if (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } else if (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } else {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            if (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        if (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            rdma_local.ib_port = ib_port;
            break;
        }
        GGML_LOG_INFO("RDMA probe: device %s port %u: no GID matching local addr %s, skipping\n",
                      dn, ib_port, target_gid_str.c_str());
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    if (!ibctx) return false;

    rdma_local.gid_idx = gid_idx;
    // capture the TCP socket's local address for later rdma_cm binding
    socklen_t src_len = sizeof(rdma_local.src_addr);
    getsockname(fd, reinterpret_cast<sockaddr *>(&rdma_local.src_addr), &src_len);

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
    qia.cap.max_send_wr     = TX_RING_DEPTH + 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    if (!rdma->qp) return false;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    if (!rdma->tx_buf || !rdma->rx_buf) return false;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!rdma->tx_mr || !rdma->rx_mr) return false;

    ibv_gid local_gid;
    if (ibv_query_gid(ibctx, rdma_local.ib_port, gid_idx, &local_gid) != 0) return false;

    rdma_local.qpn = rdma->qp->qp_num;
    rdma_local.psn = rdma->qp->qp_num & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    const char * ver_str = "";
    if (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } else if (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    GGML_LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
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
            return false;
        }
    }

    for (int i = 0; i < RDMA_RX_DEPTH; i++) {
        if (!rdma->post_rx(i)) return false;
    }

    // Transport selection: TCP is handled by the caller via GGML_RPC_NO_RDMA.
    // GGML_RDMA_USE_CM=1 -> rdma_cm path resolution for QP (mlx4/ConnectX-3).
    // Unset/0 -> original upstream raw ibv_modify_qp path (mlx5/E810).
    const char * use_cm_env = std::getenv("GGML_RDMA_USE_CM");
    const bool use_cm = (use_cm_env && use_cm_env[0] != '\0' && use_cm_env[0] != '0');
    if (!use_cm) {
        // INIT -> RTR (raw, upstream behavior)
        struct ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_num        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 12;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_num   = rdma_local.ib_port;
        if (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            return false;
        }
        // RTR -> RTS
        struct ibv_qp_attr rts = {};
        rts.qp_state     = IBV_QPS_RTS;
        rts.timeout      = 14;
        rts.retry_cnt    = 7;
        rts.rnr_retry    = 7;
        rts.sq_psn       = rdma_local.psn;
        rts.max_rd_atomic = 1;
        if (ibv_modify_qp(rdma->qp, &rts,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            return false;
        }
        GGML_LOG_INFO("RDMA activated (raw): qpn=%u->%u mtu=%d rx_depth=%d\n",
                      rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
        return true;
    }

    // RTR via rdma_cm path resolution (GGML_RDMA_USE_CM=1).
    //
    // Raw ibv_modify_qp(RTR) with an AH built from the remote GID works on NICs whose
    // kernel driver resolves RoCE paths in-core (mlx5, ice/E810) but FAILS on mlx4_ib
    // (ConnectX-3): the driver's internal path lookup returns ENETUNREACH/EINVAL for any
    // cross-host dgid, even with live ARP entries and correct routes. rdma_cm
    // (rdma_resolve_route) resolves the path through the same machinery perftest uses,
    // so we ask it for a route to the remote GID's address and copy the resolved AH
    // attributes into the QP. The remote GID is an IPv4-mapped RoCEv2 address of the
    // form ::ffff:a.b.c.d (matching the TCP socket's peer), so we resolve the peer's
    // IPv4 address over the same RDMA device/port our local GID came from.
    {
        // Derive the remote address for rdma_cm resolution. Use AF_INET for a
        // v4-mapped GID (::ffff:a.b.c.d — the RoCEv2 form of an IPv4 address)
        // and AF_INET6 for a native v6 GID. Resolving a v4-mapped v6 GID via
        // AF_INET6 rdma_resolve_addr fails (the CMA can't route it), while the
        // same address via AF_INET resolves cleanly.
        sockaddr_storage remote_addr = {};
        socklen_t remote_addr_len = 0;
        if (remote_gid[10] == 0xff && remote_gid[11] == 0xff) {
            sockaddr_in * sin = reinterpret_cast<sockaddr_in *>(&remote_addr);
            sin->sin_family = AF_INET;
            memcpy(&sin->sin_addr, remote_gid + 12, 4);
            remote_addr_len = sizeof(sockaddr_in);
        } else {
            sockaddr_in6 * sin6 = reinterpret_cast<sockaddr_in6 *>(&remote_addr);
            sin6->sin6_family = AF_INET6;
            memcpy(&sin6->sin6_addr, remote_gid, RDMA_GID_SIZE);
            remote_addr_len = sizeof(sockaddr_in6);
        }
        struct rdma_event_channel * cm_channel = cm_create_channel_nb();
        struct rdma_cm_id * cm_id = nullptr;
        struct ibv_qp_attr a = {};
        bool resolved = false;
        if (cm_channel) {
            if (rdma_create_id(cm_channel, &cm_id, nullptr, RDMA_PS_TCP) == 0) {
                // Bind to our RDMA device/port so the route resolves over the RoCE link.
                // Passing src=nullptr lets the CMA pick the source address/device itself
                // (same as perftest's rdma_cm usage) — a manual bind to the TCP socket's
                // local addr made resolve_addr fail with EINVAL on mlx4.
                int brc = 0;
                int rarc = rdma_resolve_addr(cm_id, nullptr,
                                      reinterpret_cast<sockaddr *>(&remote_addr),
                                      2000 /* ms */);
                GGML_LOG_DEBUG("RDMA cm: bind=skipped resolve_addr=%s\n",
                              rarc == 0 ? "ok" : (rarc < 0 ? strerror(errno) : "async"));
                if (rarc == 0) {
                    // Wait for RDMA_CM_EVENT_ADDR_RESOLVED — poll the channel fd with a
                    // timeout instead of blocking forever in rdma_get_cm_event.
                    auto cm_wait = [&](struct rdma_event_channel * ch, struct rdma_cm_event ** out, int timeout_ms) -> int {
                        struct pollfd pfd = { ch->fd, POLLIN, 0 };
                        int pr = poll(&pfd, 1, timeout_ms);
                        if (pr <= 0) return -1;
                        return rdma_get_cm_event(ch, out);
                    };
                    // wait for RDMA_CM_EVENT_ADDR_RESOLVED
                    struct rdma_cm_event * ev = nullptr;
                    bool addr_ev = false;
                    for (int w = 0; w < 100 && !addr_ev; w++) {
                        if (cm_wait(cm_channel, &ev, 50) == 0) {
                            if (ev->event == RDMA_CM_EVENT_ADDR_RESOLVED) addr_ev = true;
                            else { GGML_LOG_INFO("RDMA cm: unexpected addr event=%d\n", (int)ev->event); }
                            rdma_ack_cm_event(ev);
                        } else {
                            break;
                        }
                    }
                    GGML_LOG_DEBUG("RDMA cm: addr resolved=%d\n", (int)addr_ev);
                    if (addr_ev && rdma_resolve_route(cm_id, 2000) == 0) {
                        // wait for route
                        struct rdma_cm_event * rev = nullptr;
                        for (int w = 0; w < 100 && !resolved; w++) {
                            if (cm_wait(cm_channel, &rev, 50) == 0) {
                                GGML_LOG_DEBUG("RDMA cm: route event=%d\n", (int)rev->event);
                                if (rev->event == RDMA_CM_EVENT_ROUTE_RESOLVED) {
                                    resolved = true;
                                    struct rdma_route * route = &cm_id->route;
                                    GGML_LOG_DEBUG("RDMA cm: num_paths=%d\n", (int)route->num_paths);
                                    if (route->num_paths > 0) {
                                        struct ibv_sa_path_rec * path = &route->path_rec[0];
                                        // Find the local port + gid index whose GID matches the
                                        // path's sgid (cm resolves to the REAL source GID, which
                                        // may be a different index than the probe's match).
                                        // PREFER a RoCEv2-type GID entry: on mlx4 dual-mode cards
                                        // each address has both a v1 (IB-style path: dlid +
                                        // flow_label) and a v2 (Ethernet RoCE) entry. Matching
                                        // the v1 entry yields an IB path record that fails RTR
                                        // with EINVAL on an Ethernet RoCEv2 link.
                                        uint8_t cm_port = 0;
                                        int cm_gid_idx = -1;
                                        for (uint8_t prt = 1; prt <= 2; prt++) {
                                            struct ibv_port_attr pa2;
                                            if (ibv_query_port(rdma->ctx, prt, &pa2)) continue;
                                            if (pa2.state != IBV_PORT_ACTIVE) continue;  // skip DOWN ports (GIDs exist even when down)
                                            for (int g = 0; g < pa2.gid_tbl_len; g++) {
                                                union ibv_gid lg;
                                                if (ibv_query_gid(rdma->ctx, prt, g, &lg)) continue;
                                                if (memcmp(lg.raw, path->sgid.raw, RDMA_GID_SIZE) != 0) continue;
                                                // Prefer a RoCEv2-type source GID: the CM route
                                                // may resolve to a v1-type entry whose IB-style
                                                // path (dlid/flow) cannot carry RoCEv2 traffic.
                                                // Fall back to the first match if no v2 entry.
                                                char typepath[96];
                                                snprintf(typepath, sizeof(typepath),
                                                         "/sys/class/infiniband/%s/ports/%u/gid_attrs/types/%d",
                                                         ibv_get_device_name(rdma->ctx->device), prt, g);
                                                FILE * tf = fopen(typepath, "r");
                                                int is_v2 = 0;
                                                if (tf) {
                                                    char tbuf[32] = {0};
                                                    if (fgets(tbuf, sizeof(tbuf), tf)) is_v2 = strstr(tbuf, "v2") != NULL;
                                                    fclose(tf);
                                                }
                                                if (cm_gid_idx < 0) { cm_port = prt; cm_gid_idx = g; }
                                                if (is_v2) { cm_port = prt; cm_gid_idx = g; break; }
                                            }
                                            if (cm_gid_idx >= 0) break;
                                        }
                                        GGML_LOG_INFO("RDMA cm: sgid resolved to port=%u gid_idx=%d\n", cm_port, cm_gid_idx);
                                        if (cm_gid_idx < 0) {
                                            GGML_LOG_ERROR("RDMA cm: resolved sgid not found on device\n");
                                            rdma_ack_cm_event(rev);
                                            continue;
                                        }
                                        struct ibv_qp_attr rtr = {};
                                        rtr.qp_state           = IBV_QPS_RTR;
                                        rtr.path_mtu           = (enum ibv_mtu)path->mtu;
                                        rtr.dest_qp_num        = remote_qpn;
                                        rtr.rq_psn             = remote_psn;
                                        rtr.max_dest_rd_atomic = 1;
                                        rtr.min_rnr_timer      = 12;
                                        rtr.ah_attr.is_global  = 1;
                                        memcpy(rtr.ah_attr.grh.dgid.raw, path->dgid.raw, RDMA_GID_SIZE);
                                        // RoCEv2 over Ethernet must not carry IB-style fields —
                                        // a non-zero dlid/flow_label makes mlx4 emit IB packets
                                        // that go nowhere on a RoCE fabric (shows up as
                                        // 'transport retry counter exceeded' / silent drops).
                                        rtr.ah_attr.grh.flow_label   = 0;
                                        rtr.ah_attr.grh.hop_limit    = 1;
                                        rtr.ah_attr.grh.traffic_class = path->traffic_class;
                                        rtr.ah_attr.grh.sgid_index   = cm_gid_idx;
                                        rtr.ah_attr.dlid             = 0;   // RoCE: no LIDs
                                        rtr.ah_attr.sl               = 0;
                                        rtr.ah_attr.src_path_bits    = 0;
                                        rtr.ah_attr.static_rate      = path->rate;
                                        rtr.ah_attr.port_num         = cm_port;
                                        GGML_LOG_INFO("RDMA cm path: mtu=%d dgid=%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x\n",
                                                      (int)rtr.path_mtu,
                                                      rtr.ah_attr.grh.dgid.raw[0], rtr.ah_attr.grh.dgid.raw[1], rtr.ah_attr.grh.dgid.raw[2], rtr.ah_attr.grh.dgid.raw[3],
                                                      rtr.ah_attr.grh.dgid.raw[4], rtr.ah_attr.grh.dgid.raw[5], rtr.ah_attr.grh.dgid.raw[6], rtr.ah_attr.grh.dgid.raw[7],
                                                      rtr.ah_attr.grh.dgid.raw[8], rtr.ah_attr.grh.dgid.raw[9], rtr.ah_attr.grh.dgid.raw[10], rtr.ah_attr.grh.dgid.raw[11],
                                                      rtr.ah_attr.grh.dgid.raw[12], rtr.ah_attr.grh.dgid.raw[13], rtr.ah_attr.grh.dgid.raw[14], rtr.ah_attr.grh.dgid.raw[15]);
                                        if (ibv_modify_qp(rdma->qp, &rtr,
                                                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0) {
                                            GGML_LOG_DEBUG("RDMA cm: RTR ok, transitioning to RTS [dgid=%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x sgid_idx=%d port=%u dlid=%d mtu=%d flow=%u]\n",
                                                           rtr.ah_attr.grh.dgid.raw[0], rtr.ah_attr.grh.dgid.raw[1], rtr.ah_attr.grh.dgid.raw[2], rtr.ah_attr.grh.dgid.raw[3],
                                                           rtr.ah_attr.grh.dgid.raw[4], rtr.ah_attr.grh.dgid.raw[5], rtr.ah_attr.grh.dgid.raw[6], rtr.ah_attr.grh.dgid.raw[7],
                                                           rtr.ah_attr.grh.dgid.raw[8], rtr.ah_attr.grh.dgid.raw[9], rtr.ah_attr.grh.dgid.raw[10], rtr.ah_attr.grh.dgid.raw[11],
                                                           rtr.ah_attr.grh.dgid.raw[12], rtr.ah_attr.grh.dgid.raw[13], rtr.ah_attr.grh.dgid.raw[14], rtr.ah_attr.grh.dgid.raw[15],
                                                           rtr.ah_attr.grh.sgid_index, rtr.ah_attr.port_num, rtr.ah_attr.dlid,
                                                           (int)rtr.path_mtu, rtr.ah_attr.grh.flow_label);
                                            struct ibv_qp_attr rts = {};
                                            rts.qp_state     = IBV_QPS_RTS;
                                            rts.timeout      = 14;
                                            rts.retry_cnt    = 7;
                                            rts.rnr_retry    = 7;
                                            rts.sq_psn       = rdma_local.psn;
                                            rts.max_rd_atomic = 1;
                                            if (ibv_modify_qp(rdma->qp, &rts,
                                                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                                    IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
                                                GGML_LOG_ERROR("RDMA cm: RTS failed: %s (errno=%d)\n", strerror(errno), errno);
                                                resolved = false;
                                            }
                                        } else {
                                            GGML_LOG_ERROR("RDMA cm: RTR failed: %s (errno=%d) [mtu=%d dgid=%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x sgid_idx=%d port=%u dlid=%d tc=%d hop=%d flow=%u sl=%d rate=%d]\n",
                                                           strerror(errno), errno,
                                                           (int)rtr.path_mtu,
                                                           rtr.ah_attr.grh.dgid.raw[0], rtr.ah_attr.grh.dgid.raw[1], rtr.ah_attr.grh.dgid.raw[2], rtr.ah_attr.grh.dgid.raw[3],
                                                           rtr.ah_attr.grh.dgid.raw[4], rtr.ah_attr.grh.dgid.raw[5], rtr.ah_attr.grh.dgid.raw[6], rtr.ah_attr.grh.dgid.raw[7],
                                                           rtr.ah_attr.grh.dgid.raw[8], rtr.ah_attr.grh.dgid.raw[9], rtr.ah_attr.grh.dgid.raw[10], rtr.ah_attr.grh.dgid.raw[11],
                                                           rtr.ah_attr.grh.dgid.raw[12], rtr.ah_attr.grh.dgid.raw[13], rtr.ah_attr.grh.dgid.raw[14], rtr.ah_attr.grh.dgid.raw[15],
                                                           rtr.ah_attr.grh.sgid_index, rtr.ah_attr.port_num, rtr.ah_attr.dlid,
                                                           (int)rtr.ah_attr.grh.traffic_class, (int)rtr.ah_attr.grh.hop_limit,
                                                           rtr.ah_attr.grh.flow_label, (int)rtr.ah_attr.sl, (int)rtr.ah_attr.static_rate);
                                            resolved = false;
                                        }
                                    } else {
                                        resolved = false;
                                    }
                                }
                                rdma_ack_cm_event(rev);
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }
        if (cm_id) rdma_destroy_id(cm_id);
        if (cm_channel) rdma_destroy_event_channel(cm_channel);
        if (!resolved) {
            GGML_LOG_ERROR("RDMA activate failed (rdma_cm path resolution), staying on TCP\n");
            return false;
        }
    }

    GGML_LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    return true;
}

bool socket_t::impl::rdma_poll(struct ibv_cq * cq, struct ibv_wc * wc) {
    // Long-wait semantics: model loads from slow NAS can take 30-60+ min and the
    // client legitimately stalls in D-state on storage reads while the server
    // waits. Default timeout 30 min (survives any real load, still frees a
    // genuinely dead peer). Override with GGML_RDMA_POLL_TIMEOUT (seconds).
    // While waiting: log a warning so freezes are VISIBLE instead of silent.
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int poll_timeout_s = 1800;  // 30 min default
    {
        const char * pt = std::getenv("GGML_RDMA_POLL_TIMEOUT");
        if (pt) poll_timeout_s = atoi(pt);  // 0 = infinite
    }
    bool warned_idle = false;
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
            struct timespec tn;
            clock_gettime(CLOCK_MONOTONIC, &tn);
            double elapsed = (tn.tv_sec - t0.tv_sec) + (tn.tv_nsec - t0.tv_nsec) / 1e9;
            if (tcp_peer_closed()) {
                return false;
            }
            // visibility: warn once per stall so freezes are diagnosable
            if (elapsed > 60 && !warned_idle) {
                GGML_LOG_ERROR("RDMA poll idle %.0fs (peer busy/slow storage — waiting)\n", elapsed);
                warned_idle = true;
            }
            if (poll_timeout_s > 0 && elapsed > poll_timeout_s) {
                GGML_LOG_ERROR("RDMA poll timeout (%.0fs): peer dead, dropping connection\n", elapsed);
                return false;
            }
        }
    }
}

bool socket_t::impl::rdma_send(const void * data, size_t size) {
    rdma_conn * c = rdma.get();
    const uint8_t * src = (const uint8_t *)data;
    size_t rem = size;
    int pipelined = 0;  // chunks posted without waiting for completion
    uint64_t spins = 0;

    while (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        struct ibv_sge sge = {};
        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        if (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            // inline: data copied into the WR at post time, no buffer needed
        } else {
            // pipeline: use the next tx_buf in the ring (each chunk gets its own
            // buffer, so multiple chunks can be in flight simultaneously)
            memcpy(c->tx_bufs[c->tx_head], src, chunk);
            sge.addr   = (uintptr_t)c->tx_bufs[c->tx_head];
            sge.length = chunk;
            sge.lkey   = c->tx_mrs[c->tx_head]->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
            c->tx_head = (c->tx_head + 1) % TX_RING_DEPTH;
        }

        // ring-wrap safety: if all TX_RING_DEPTH buffers are outstanding,
        // drain completions until at least one tx_buf is free before
        // overwriting it (otherwise we corrupt in-flight data)
        if (c->tx_outstanding >= TX_RING_DEPTH) {
            struct ibv_wc wc_drain;
            int n_drain;
            while (c->tx_outstanding > 0) {
                n_drain = ibv_poll_cq(c->scq, 1, &wc_drain);
                if (n_drain > 0) {
                    c->n_send_done++;
                    c->tx_outstanding--;
                    if (wc_drain.status != IBV_WC_SUCCESS) {
                        GGML_LOG_ERROR("RDMA send CQ error (drain): status=%d\n", wc_drain.status);
                        return false;
                    }
                } else if (n_drain < 0) {
                    return false;
                } else {
                    struct timespec ts = {0, 1000}; // 1µs spin
                    nanosleep(&ts, nullptr);
                }
            }
            pipelined = 0;
        }

        if (ibv_post_send(c->qp, &wr, &bad) != 0) return false;
        c->n_send_posted++;
        c->tx_outstanding++;
        pipelined++;  // count for ring-wrap check (checked at top of next iteration)
        src += chunk;
        rem -= chunk;

        // poll the SCQ periodically to drain completions (non-blocking, don't
        // wait — we're pipelining). Also poll RCQ to repost incoming recvs.
        struct ibv_wc wc;
        int n;
        while ((n = ibv_poll_cq(c->scq, 1, &wc)) > 0) {
            c->n_send_done++;
            if (wc.status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RDMA send CQ error: status=%d (%s)\n",
                    wc.status, ibv_wc_status_str(wc.status));
                return false;
            }
        }
        if (n < 0) return false;
        struct ibv_wc rwc;
        while ((n = ibv_poll_cq(c->rcq, 1, &rwc)) > 0) {
            if (rwc.status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RDMA recv CQ error during send: status=%d (%s)\n",
                    rwc.status, ibv_wc_status_str(rwc.status));
                return false;
            }
            struct ibv_sge rsge = {};
            rsge.addr   = (uintptr_t)c->rx_slot((int)rwc.wr_id);
            rsge.length = RDMA_CHUNK;
            rsge.lkey   = c->rx_mr->lkey;
            struct ibv_recv_wr rwr = {}, * rbad = nullptr;
            rwr.wr_id = rwc.wr_id; rwr.sg_list = &rsge; rwr.num_sge = 1;
            if (ibv_post_recv(c->qp, &rwr, &rbad) != 0) return false;
            c->n_recv_done++; c->n_recv_reposted++;
        }
        if (n < 0) return false;

        // adaptive pacing — BULK ONLY: for large streaming, pause periodically
        // to let the peer drain (prevents RNR). Small sends are latency-critical.
        if (chunk == RDMA_CHUNK && c->n_send_posted % 1024 == 0 && size > 1024 * 1024) {
            struct timespec pz = {0, g_pacing_delay_us * 1000};
            nanosleep(&pz, nullptr);
            if (g_pacing_delay_us < PACING_MAX_US) {
                g_pacing_delay_us = g_pacing_delay_us ? g_pacing_delay_us * 2 : PACING_MIN_US;
            }
        } else if (chunk < RDMA_CHUNK / 2) {
            g_pacing_delay_us = PACING_MIN_US;
        }
    }
    // drain remaining completions
    struct ibv_wc wc;
    while (c->n_send_done < c->n_send_posted) {
        int n = ibv_poll_cq(c->scq, 1, &wc);
        if (n > 0) {
            c->n_send_done++;
            c->tx_outstanding--;
            if (wc.status != IBV_WC_SUCCESS) {
                GGML_LOG_ERROR("RDMA send CQ error (drain): status=%d\n", wc.status);
                return false;
            }
        } else if (n < 0) return false;
        else {
            struct ibv_wc rwc;
            int rn = ibv_poll_cq(c->rcq, 1, &rwc);
            if (rn > 0) {
                struct ibv_sge rsge = {};
                rsge.addr   = (uintptr_t)c->rx_slot((int)rwc.wr_id);
                rsge.length = RDMA_CHUNK;
                rsge.lkey   = c->rx_mr->lkey;
                struct ibv_recv_wr rwr = {}, * rbad = nullptr;
                rwr.wr_id = rwc.wr_id; rwr.sg_list = &rsge; rwr.num_sge = 1;
                ibv_post_recv(c->qp, &rwr, &rbad);
                c->n_recv_done++; c->n_recv_reposted++;
            } else if (rn < 0) return false;
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
        if (!rdma_poll(c->rcq, &wc)) {
            GGML_LOG_ERROR("RDMA recv failed: sent=%llu/%llu recvd=%llu reposted=%llu\n",
                           (unsigned long long)c->n_send_posted, (unsigned long long)c->n_send_done,
                           (unsigned long long)c->n_recv_done, (unsigned long long)c->n_recv_reposted);
            return false;
        }
        c->n_recv_done++;
        // peer is successfully sending to us → their send path is healthy →
        // relax our pacing so we don't throttle them unnecessarily
        if (g_pacing_delay_us > PACING_MIN_US) {
            g_pacing_delay_us /= 2;
        }
        if ((c->n_recv_done & 1023) == 0) {
            GGML_LOG_INFO("RDMA progress: sent=%llu/%llu recvd=%llu reposted=%llu\n",
                           (unsigned long long)c->n_send_posted, (unsigned long long)c->n_send_done,
                           (unsigned long long)c->n_recv_done, (unsigned long long)c->n_recv_reposted);
        }

        int slot = (int)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        if (!c->post_rx(slot)) return false;
        c->n_recv_reposted++;

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
#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE) && !defined(_WIN32)
    if (cm_connected) {
        // native-CM connection: the QP is already connected; advertise our
        // QP identity (informational — the peer keeps RDMA regardless).
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, sizeof(rc));
        return;
    }
#endif
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
#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE) && !defined(_WIN32)
    if (cm_connected) {
        // native-CM connection: already connected, nothing to activate.
        // use_rdma stays true.
        (void)remote_caps;
        return;
    }
#endif
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

#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE) && !defined(_WIN32)
// ---------------------------------------------------------------------------
// Native rdma_cm transport (GGML_RDMA_USE_CM=1).
//
// Both peers establish a real RC connection through the RDMA Connection
// Manager — the same machinery perftest/ib_write_bw uses — instead of
// hand-rolling ibv_modify_qp transitions. No TCP socket exists at any point
// on either side; the HELLO handshake runs over the connected QP.
// ---------------------------------------------------------------------------

// create a CM event channel with a NON-BLOCKING fd: rdma_get_cm_event would
// otherwise block forever when no event is pending (deadlocks the poll loop)
static struct rdma_event_channel * cm_create_channel_nb() {
    struct rdma_event_channel * ch = rdma_create_event_channel();
    if (ch) {
        int flags = fcntl(ch->fd, F_GETFL);
        fcntl(ch->fd, F_SETFL, flags | O_NONBLOCK);
    }
    return ch;
}

bool rdma_cm_wanted() {
    if (std::getenv("GGML_RPC_NO_RDMA")) return false;
    const char * e = std::getenv("GGML_RDMA_USE_CM");
    return e && e[0] != '\0' && e[0] != '0';
}

// bounded wait for one CM event (rdma_get_cm_event blocks forever otherwise)
static int cm_wait_event(struct rdma_event_channel * ch, struct rdma_cm_event ** ev, int timeout_ms) {
    struct pollfd pfd = { ch->fd, POLLIN, 0 };
    if (poll(&pfd, 1, timeout_ms) <= 0) return -1;
    return rdma_get_cm_event(ch, ev);
}

// honor GGML_RDMA_DEV pin: fail if the CM picked a different device
static bool cm_check_dev(struct rdma_cm_id * id) {
    const char * dev_env = std::getenv("GGML_RDMA_DEV");
    if (!dev_env || !dev_env[0]) return true;
    const char * dn = id->verbs ? ibv_get_device_name(id->verbs->device) : nullptr;
    if (!dn || strcmp(dn, dev_env) != 0) {
        GGML_LOG_ERROR("RDMA cm: device %s does not match GGML_RDMA_DEV=%s\n",
                       dn ? dn : "(none)", dev_env);
        return false;
    }
    return true;
}

// allocate PD/CQs + buffers/MRs + pre-posted recvs around an rdma_create_qp'd QP
static std::unique_ptr<rdma_conn> cm_make_conn(struct rdma_cm_id * id) {
    auto c = std::make_unique<rdma_conn>();
    c->ctx = id->verbs;
    c->ctx_owned = false;  // owned by the cm_id; rdma_destroy_id closes it
    c->pd = ibv_alloc_pd(c->ctx);
    if (!c->pd) { GGML_LOG_ERROR("RDMA cm: ibv_alloc_pd failed\n"); return nullptr; }
    c->scq = ibv_create_cq(c->ctx, 16, nullptr, nullptr, 0);
    c->rcq = ibv_create_cq(c->ctx, RDMA_RX_DEPTH + 4, nullptr, nullptr, 0);
    if (!c->scq || !c->rcq) { GGML_LOG_ERROR("RDMA cm: ibv_create_cq failed\n"); return nullptr; }
    ibv_qp_init_attr qia = {};
    qia.send_cq = c->scq;
    qia.recv_cq = c->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = TX_RING_DEPTH + 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;
    if (rdma_create_qp(id, c->pd, &qia) != 0) {
        int e1 = errno;
        // transient driver pressure can fail the first attempt; one retry
        struct timespec rq = {0, 100 * 1000 * 1000};
        nanosleep(&rq, nullptr);
        if (rdma_create_qp(id, c->pd, &qia) != 0) {
            GGML_LOG_ERROR("RDMA cm: rdma_create_qp failed: %s (retry: %s)\n",
                           strerror(e1), strerror(errno));
            return nullptr;
        }
        GGML_LOG_INFO("RDMA cm: rdma_create_qp succeeded on retry\n");
    }
    c->qp = id->qp;
    c->max_inline = qia.cap.max_inline_data;
    for (int i = 0; i < TX_RING_DEPTH; i++) {
        c->tx_bufs[i] = aligned_alloc(4096, RDMA_CHUNK);
        if (!c->tx_bufs[i]) { GGML_LOG_ERROR("RDMA cm: tx aligned_alloc failed\n"); return nullptr; }
        c->tx_mrs[i] = ibv_reg_mr(c->pd, c->tx_bufs[i], RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
        if (!c->tx_mrs[i]) { GGML_LOG_ERROR("RDMA cm: tx ibv_reg_mr failed\n"); return nullptr; }
    }
    c->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    if (!c->rx_buf) { GGML_LOG_ERROR("RDMA cm: rx aligned_alloc failed\n"); return nullptr; }
    c->rx_mr = ibv_reg_mr(c->pd, c->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!c->rx_mr) { GGML_LOG_ERROR("RDMA cm: rx ibv_reg_mr failed\n"); return nullptr; }
    for (int i = 0; i < RDMA_RX_DEPTH; i++) {
        if (!c->post_rx(i)) return nullptr;
    }
    return c;
}

socket_ptr socket_t::connect_rdma(const char * host, int port) {
    struct timespec cts;
    clock_gettime(CLOCK_REALTIME, &cts);
    GGML_LOG_DEBUG("RDMA cm: connect attempt to %s:%d at %ld.%03ld\n",
                   host, port, (long)cts.tv_sec, cts.tv_nsec / 1000000);
    auto impl = std::make_unique<socket_t::impl>((sockfd_t)-1);
    struct rdma_event_channel * ch = cm_create_channel_nb();
    if (!ch) {
        GGML_LOG_ERROR("RDMA cm: rdma_create_event_channel failed\n");
        return nullptr;
    }
    impl->cm_ch = ch;
    impl->cm_owns_ch = true;
    struct rdma_cm_id * id = nullptr;
    if (rdma_create_id(ch, &id, nullptr, RDMA_PS_TCP) != 0) {
        GGML_LOG_ERROR("RDMA cm: rdma_create_id failed\n");
        return nullptr;
    }
    impl->cm_id = id;
    // resolve server address (v4 or v6 via getaddrinfo)
    struct addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo * res = nullptr;
    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res) {
        GGML_LOG_ERROR("RDMA cm: cannot resolve host '%s'\n", host);
        return nullptr;
    }
    struct rdma_cm_event * ev = nullptr;
    socket_ptr out = nullptr;
    if (rdma_resolve_addr(id, nullptr, res->ai_addr, 2000) != 0) {
        GGML_LOG_ERROR("RDMA cm: rdma_resolve_addr failed: %s\n", strerror(errno));
        goto done;
    }
    if (cm_wait_event(ch, &ev, 5000) != 0 || ev->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
        GGML_LOG_ERROR("RDMA cm: addr resolve failed (event=%d)\n", ev ? (int)ev->event : -1);
        if (ev) rdma_ack_cm_event(ev);
        goto done;
    }
    rdma_ack_cm_event(ev);
    ev = nullptr;
    if (rdma_resolve_route(id, 2000) != 0) {
        GGML_LOG_ERROR("RDMA cm: rdma_resolve_route failed: %s\n", strerror(errno));
        goto done;
    }
    if (cm_wait_event(ch, &ev, 5000) != 0 || ev->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
        GGML_LOG_ERROR("RDMA cm: route resolve failed (event=%d)\n", ev ? (int)ev->event : -1);
        if (ev) rdma_ack_cm_event(ev);
        goto done;
    }
    rdma_ack_cm_event(ev);
    ev = nullptr;
    if (!cm_check_dev(id)) goto done;
    // local port follows the CM route; find the active port on this device
    for (uint8_t prt = 1; prt <= 2; prt++) {
        struct ibv_port_attr pa;
        if (ibv_query_port(id->verbs, prt, &pa)) continue;
        if (pa.state == IBV_PORT_ACTIVE) { impl->rdma_local.ib_port = prt; break; }
    }
    if (!impl->rdma_local.ib_port) {
        GGML_LOG_ERROR("RDMA cm: no active port on device\n");
        goto done;
    }
    impl->rdma = cm_make_conn(id);
    if (!impl->rdma) {
        GGML_LOG_ERROR("RDMA cm: QP resource setup failed\n");
        goto done;
    }
    {
        struct rdma_conn_param cp = {};
        cp.responder_resources = 1;
        cp.initiator_depth     = 1;
        cp.retry_count         = 7;
        cp.rnr_retry_count     = 7;
        if (rdma_connect(id, &cp) != 0) {
            GGML_LOG_ERROR("RDMA cm: rdma_connect failed: %s\n", strerror(errno));
            goto done;
        }
    }
    if (cm_wait_event(ch, &ev, 10000) != 0 || ev->event != RDMA_CM_EVENT_ESTABLISHED) {
        GGML_LOG_ERROR("RDMA cm: connect failed (event=%d)\n", ev ? (int)ev->event : -1);
        if (ev) rdma_ack_cm_event(ev);
        goto done;
    }
    rdma_ack_cm_event(ev);
    ev = nullptr;
    impl->rdma_local.qpn = impl->rdma->qp->qp_num;
    impl->rdma_local.psn = impl->rdma->qp->qp_num & 0xffffff;
    impl->use_rdma = true;
    impl->cm_connected = true;
    GGML_LOG_INFO("RDMA cm: connected %s:%d qpn=%u\n", host, port, impl->rdma_local.qpn);
    out = socket_ptr(new socket_t(std::move(impl)));
done:
    if (res) freeaddrinfo(res);
    return out;
}

struct rdma_cm_listener {
    struct rdma_event_channel * ch = nullptr;
    struct rdma_cm_id *          id = nullptr;
};

rdma_cm_listener * rdma_cm_listen(const char * host, int port) {
    auto * srv = new (std::nothrow) rdma_cm_listener();
    if (!srv) return nullptr;
    srv->ch = cm_create_channel_nb();
    if (!srv->ch) { delete srv; return nullptr; }
    if (rdma_create_id(srv->ch, &srv->id, nullptr, RDMA_PS_TCP) != 0) {
        rdma_destroy_event_channel(srv->ch);
        delete srv;
        return nullptr;
    }
    struct addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo * res = nullptr;
    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res) {
        GGML_LOG_ERROR("RDMA cm: cannot resolve listen addr '%s'\n", host);
        rdma_destroy_id(srv->id);
        rdma_destroy_event_channel(srv->ch);
        delete srv;
        return nullptr;
    }
    bool bound = false;
    for (struct addrinfo * ai = res; ai; ai = ai->ai_next) {
        if (rdma_bind_addr(srv->id, ai->ai_addr) == 0) { bound = true; break; }
    }
    freeaddrinfo(res);
    if (!bound) {
        GGML_LOG_ERROR("RDMA cm: rdma_bind_addr %s:%d failed: %s\n", host, port, strerror(errno));
        rdma_destroy_id(srv->id);
        rdma_destroy_event_channel(srv->ch);
        delete srv;
        return nullptr;
    }
    if (rdma_listen(srv->id, 16) != 0) {
        GGML_LOG_ERROR("RDMA cm: rdma_listen failed: %s\n", strerror(errno));
        rdma_destroy_id(srv->id);
        rdma_destroy_event_channel(srv->ch);
        delete srv;
        return nullptr;
    }
    GGML_LOG_INFO("RDMA cm: listening on %s:%d\n", host, port);
    return srv;
}

socket_ptr socket_t::accept_rdma(struct rdma_cm_listener * srv) {
    if (!srv) return nullptr;
    // wait for a CONNECT_REQUEST (bounded; server loop retries)
    struct rdma_cm_event * ev = nullptr;
    for (;;) {
        if (cm_wait_event(srv->ch, &ev, 1000) != 0) return nullptr;  // timeout: let caller loop
        if (ev->event == RDMA_CM_EVENT_CONNECT_REQUEST) break;
        GGML_LOG_INFO("RDMA cm: ignoring event %d while accepting\n", (int)ev->event);
        rdma_ack_cm_event(ev);
        ev = nullptr;
    }
    struct rdma_cm_id * child = ev->id;
    rdma_ack_cm_event(ev);
    ev = nullptr;
    // migrate the connection to a private channel so the listener stays clean
    struct rdma_event_channel * ch = cm_create_channel_nb();
    if (!ch) { rdma_reject(child, nullptr, 0); return nullptr; }
    if (rdma_migrate_id(child, ch) != 0) {
        rdma_destroy_event_channel(ch);
        rdma_reject(child, nullptr, 0);
        return nullptr;
    }
    auto impl = std::make_unique<socket_t::impl>((sockfd_t)-1);
    impl->cm_id = child;
    impl->cm_ch = ch;
    impl->cm_owns_ch = true;
    impl->rdma = cm_make_conn(child);
    if (!impl->rdma) {
        GGML_LOG_ERROR("RDMA cm: accept QP setup failed\n");
        return nullptr;  // impl dtor tears down cm_id/channel
    }
    // local port follows the accepted connection's device
    for (uint8_t prt = 1; prt <= 2; prt++) {
        struct ibv_port_attr pa;
        if (ibv_query_port(child->verbs, prt, &pa)) continue;
        if (pa.state == IBV_PORT_ACTIVE) { impl->rdma_local.ib_port = prt; break; }
    }
    struct rdma_conn_param cp = {};
    cp.responder_resources = 1;
    cp.initiator_depth     = 1;
    if (rdma_accept(child, &cp) != 0) {
        GGML_LOG_ERROR("RDMA cm: rdma_accept failed: %s\n", strerror(errno));
        return nullptr;
    }
    if (cm_wait_event(ch, &ev, 10000) != 0 || ev->event != RDMA_CM_EVENT_ESTABLISHED) {
        GGML_LOG_ERROR("RDMA cm: accept failed (event=%d)\n", ev ? (int)ev->event : -1);
        if (ev) rdma_ack_cm_event(ev);
        return nullptr;
    }
    rdma_ack_cm_event(ev);
    impl->rdma_local.qpn = impl->rdma->qp->qp_num;
    impl->rdma_local.psn = impl->rdma->qp->qp_num & 0xffffff;
    impl->use_rdma = true;
    impl->cm_connected = true;
    GGML_LOG_INFO("RDMA cm: accepted connection qpn=%u\n", impl->rdma_local.qpn);
    return socket_ptr(new socket_t(std::move(impl)));
}

socket_ptr rdma_cm_accept(rdma_cm_listener * srv) {
    return socket_t::accept_rdma(srv);
}

void rdma_cm_close(rdma_cm_listener * srv) {
    if (!srv) return;
    if (srv->id) rdma_destroy_id(srv->id);
    if (srv->ch) rdma_destroy_event_channel(srv->ch);
    delete srv;
}
#endif // GGML_RPC_RDMA && !APPLE && !WIN32

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
