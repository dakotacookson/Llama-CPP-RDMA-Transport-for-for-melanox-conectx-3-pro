#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);
#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE) && !defined(_WIN32)
    // Native rdma_cm connection (GGML_RDMA_USE_CM=1): the RC connection is
    // established entirely through the RDMA Connection Manager — no TCP socket
    // exists at all. The HELLO handshake then runs over the connected QP.
    static socket_ptr connect_rdma(const char * host, int port);
    static socket_ptr accept_rdma(struct rdma_cm_listener * srv);
#endif

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();

#if defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE) && !defined(_WIN32)
// Native rdma_cm listener for ggml-rpc-server (GGML_RDMA_USE_CM=1).
// Listens on the SAME host:port the TCP server would use; no TCP socket exists.
struct rdma_cm_listener;
rdma_cm_listener * rdma_cm_listen(const char * host, int port);
// Blocks until a client connects via rdma_connect; returns a socket whose RC QP
// is already connected (use_rdma preset). nullptr on error/timeout.
socket_ptr rdma_cm_accept(rdma_cm_listener * srv);
void rdma_cm_close(rdma_cm_listener * srv);
// True when GGML_RDMA_USE_CM=1 (and not NO_RDMA): both client and server take
// the native-CM path instead of TCP.
bool rdma_cm_wanted();
#endif
