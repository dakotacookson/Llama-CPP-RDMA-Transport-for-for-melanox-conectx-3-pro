// Unit/integration tests for the native rdma_cm RPC transport
// (GGML_RDMA_USE_CM=1, transport.cpp).
//
// - test_env_flags: pure logic, no RDMA hardware needed.
// - test_cm_loopback: full rdma_cm listen -> connect -> RDMA ping-pong
//   against the machine's OWN RoCE IP. Needs real RDMA hardware but runs in
//   seconds (vs minutes for a model load). Skipped gracefully when no device
//   or env requests it.
//
// Usage:
//   test-rpc-rdma-cm                 # env-flag tests only
//   GGML_RDMA_USE_CM=1 test-rpc-rdma-cm 10.0.0.105 [port]
//   (loopback test also honors GGML_RDMA_DEV pinning)

#include "ggml-rpc/transport.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>
#include <atomic>

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
    else { printf("ok %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

static void test_env_flags() {
    printf("== test_env_flags ==\n");
    unsetenv("GGML_RDMA_USE_CM");
    unsetenv("GGML_RPC_NO_RDMA");
    CHECK(rdma_cm_wanted() == false);
    setenv("GGML_RDMA_USE_CM", "1", 1);
    CHECK(rdma_cm_wanted() == true);
    setenv("GGML_RDMA_USE_CM", "0", 1);
    CHECK(rdma_cm_wanted() == false);
    setenv("GGML_RDMA_USE_CM", "", 1);
    CHECK(rdma_cm_wanted() == false);
    // TCP kill-switch always wins
    setenv("GGML_RDMA_USE_CM", "1", 1);
    setenv("GGML_RPC_NO_RDMA", "1", 1);
    CHECK(rdma_cm_wanted() == false);
    unsetenv("GGML_RPC_NO_RDMA");
    unsetenv("GGML_RDMA_USE_CM");
}

static void test_cm_loopback(const char * ip, int port) {
    printf("== test_cm_loopback %s:%d ==\n", ip, port);
    setenv("GGML_RDMA_USE_CM", "1", 1);

    std::string server_msg;
    std::atomic<bool> server_done{false};
    std::atomic<bool> server_ok{false};
    std::atomic<int> served{0};

    std::thread server([&]() {
        rdma_cm_listener * listener = rdma_cm_listen(ip, port);
        if (!listener) {
            server_msg = "listen failed (no RDMA device/route?)";
            server_done = true;
            return;
        }
        // serve TWO sequential connections (rpc-server does this per model load)
        for (int round = 0; round < 2; round++) {
            socket_ptr sock;
            for (int i = 0; i < 60 && !sock; i++) {
                sock = rdma_cm_accept(listener);
            }
            if (!sock) {
                server_msg = "accept failed on round " + std::to_string(round);
                server_done = true;
                rdma_cm_close(listener);
                return;
            }
            char buf[16] = {0};
            if (!sock->recv_data(buf, sizeof(buf)) ||
                memcmp(buf, "rdma-ping-123456", 16) != 0 ||
                !sock->send_data("rdma-pong-123456", 16) || !sock->flush()) {
                server_msg = "ping-pong failed on round " + std::to_string(round);
                server_done = true;
                rdma_cm_close(listener);
                return;
            }
            served++;
        }
        rdma_cm_close(listener);
        server_ok = true;
        server_done = true;
    });

    // give the listener a moment to bind
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    for (int round = 0; round < 2; round++) {
        socket_ptr client = socket_t::connect_rdma(ip, port);
        CHECK(client != nullptr);
        if (!client) break;
        CHECK(client->send_data("rdma-ping-123456", 16));
        CHECK(client->flush());
        char reply[16] = {0};
        CHECK(client->recv_data(reply, sizeof(reply)));
        CHECK(memcmp(reply, "rdma-pong-123456", 16) == 0);
    }
    server.join();
    if (!server_ok) {
        printf("FAIL server side: %s (served %d/2)\n", server_msg.c_str(), (int)served);
        failures++;
    } else {
        printf("ok server side ping-pong x2\n");
    }
    unsetenv("GGML_RDMA_USE_CM");
}

int main(int argc, char ** argv) {
    test_env_flags();
    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "--server" && argc > 3) {
            // server-only: listen on <bind-ip> <port>, serve ONE ping-pong, exit
            const char * ip = argv[2];
            int port = atoi(argv[3]);
            printf("== test_cm_server %s:%d ==\n", ip, port);
            setenv("GGML_RDMA_USE_CM", "1", 1);
            rdma_cm_listener * listener = rdma_cm_listen(ip, port);
            CHECK(listener != nullptr);
            if (listener) {
                // rdma_cm_accept waits ~1s per call; retry for up to 60s so a
                // human has time to start the client on the other box.
                socket_ptr sock;
                for (int i = 0; i < 60 && !sock; i++) {
                    sock = rdma_cm_accept(listener);
                }
                rdma_cm_close(listener);
                CHECK(sock != nullptr);
                if (sock) {
                    char buf[16] = {0};
                    CHECK(sock->recv_data(buf, sizeof(buf)));
                    CHECK(memcmp(buf, "rdma-ping-123456", 16) == 0);
                    CHECK(sock->send_data("rdma-pong-123456", 16));
                    CHECK(sock->flush());
                }
            }
            unsetenv("GGML_RDMA_USE_CM");
        } else if (mode == "--client" && argc > 3) {
            // client-only: connect to <server-ip> <port>, one ping-pong, exit
            const char * ip = argv[2];
            int port = atoi(argv[3]);
            printf("== test_cm_client %s:%d ==\n", ip, port);
            setenv("GGML_RDMA_USE_CM", "1", 1);
            socket_ptr client = socket_t::connect_rdma(ip, port);
            CHECK(client != nullptr);
            if (client) {
                CHECK(client->send_data("rdma-ping-123456", 16));
                CHECK(client->flush());
                char reply[16] = {0};
                CHECK(client->recv_data(reply, sizeof(reply)));
                CHECK(memcmp(reply, "rdma-pong-123456", 16) == 0);
            }
            unsetenv("GGML_RDMA_USE_CM");
        } else if (mode == "--bulk" && argc > 3) {
            // bulk transfer: like real tensor traffic — N x 256KB RDMA writes
            // back-to-back over one connection, then a second connection.
            // usage: --bulk <ip> <port> ; server side: --bulk-server <ip> <port>
            const char * ip = argv[2];
            int port = atoi(argv[3]);
            printf("== test_cm_bulk_client %s:%d ==\n", ip, port);
            setenv("GGML_RDMA_USE_CM", "1", 1);
            static char chunk[256 * 1024];
            for (int i = 0; i < 256 * 1024; i++) chunk[i] = (char)(i & 0xff);
            for (int round = 0; round < 2; round++) {
                socket_ptr client = socket_t::connect_rdma(ip, port);
                CHECK(client != nullptr);
                if (!client) break;
                for (int m = 0; m < 20; m++) {
                    if (!client->send_data(chunk, sizeof(chunk))) break;
                }
                CHECK(client->flush());
                char ack[8] = {0};
                CHECK(client->recv_data(ack, sizeof(ack)));
                CHECK(memcmp(ack, "bulk-ok", 7) == 0);
            }
            unsetenv("GGML_RDMA_USE_CM");
        } else if (mode == "--bulk-server" && argc > 3) {
            const char * ip = argv[2];
            int port = atoi(argv[3]);
            printf("== test_cm_bulk_server %s:%d ==\n", ip, port);
            setenv("GGML_RDMA_USE_CM", "1", 1);
            rdma_cm_listener * listener = rdma_cm_listen(ip, port);
            CHECK(listener != nullptr);
            if (listener) {
                for (int round = 0; round < 2; round++) {
                    socket_ptr sock;
                    for (int i = 0; i < 60 && !sock; i++) sock = rdma_cm_accept(listener);
                    CHECK(sock != nullptr);
                    if (!sock) break;
                    static char rbuf[256 * 1024];
                    bool ok = true;
                    for (int m = 0; m < 20; m++) {
                        if (!sock->recv_data(rbuf, sizeof(rbuf))) { ok = false; break; }
                    }
                    CHECK(ok);
                    // verify pattern of last chunk
                    bool pat = true;
                    for (int i = 0; i < 256 * 1024; i += 4096) {
                        if (rbuf[i] != (char)(i & 0xff)) { pat = false; break; }
                    }
                    CHECK(pat);
                    CHECK(sock->send_data("bulk-ok", 8));
                    CHECK(sock->flush());
                }
                rdma_cm_close(listener);
            }
            unsetenv("GGML_RDMA_USE_CM");
        } else {
            int port = argc > 2 ? atoi(argv[2]) : 19876;
            test_cm_loopback(argv[1], port);
        }
    } else {
        printf("(skip test_cm_loopback: pass <roce-ip> [port] to run it)\n");
        printf("cross-box: --server <bind-ip> <port> on one box, --client <server-ip> <port> on the other\n");
    }
    printf(failures ? "RESULT: %d FAILURES\n" : "RESULT: ALL PASS\n", failures);
    return failures ? 1 : 0;
}
