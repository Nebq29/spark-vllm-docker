/* lanex_core.c — 4-lane TCP exchange-and-add core for Thor TP=2.
 *
 * Pure network I/O in C pthreads (no GIL). Python stages GPU->host buffer,
 * calls lanex_xchg() (directly or via lanex_xchg_async in lanex_async.c) to get
 * the peer's bytes into a second host buffer, then the GPU adds it in.
 *
 * v2 (round 3): per-call peer-wait timing (lanex_last_stats), mutex, error-path fix.
 * v3 (round 5):
 *   - optional MSG_ZEROCOPY sends (lanex_set_zerocopy): the NIC DMAs the send
 *     buffer directly, so the CPU never reads it. Each lanex_xchg waits for all
 *     zerocopy completions before returning, so the buffer can be reused safely.
 *     lanex_zc_stats() reports how many completions the kernel had to fall back
 *     to copying (ZEROCOPY_COPIED).
 *   - a lane whose socket sees no progress for 30 s now fails with ETIMEDOUT
 *     instead of polling forever.
 *
 * v4 (round 7): LANEX_SOCKBUF env sets the socket buffer size (default 32M as
 *   before; 0 = leave it to kernel autotuning); lanex_sockbuf() reports it.
 *
 * Build (with the async shim):
 *   gcc -O2 -shared -fPIC -pthread -o lanex_core.so lanex_core.c lanex_async.c -ldl
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <linux/errqueue.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#ifndef SO_ZEROCOPY
#define SO_ZEROCOPY 60
#endif
#ifndef MSG_ZEROCOPY
#define MSG_ZEROCOPY 0x4000000
#endif
#ifndef SO_EE_ORIGIN_ZEROCOPY
#define SO_EE_ORIGIN_ZEROCOPY 5
#endif
#ifndef SO_EE_CODE_ZEROCOPY_COPIED
#define SO_EE_CODE_ZEROCOPY_COPIED 1
#endif

#define MAXL 8
/* Exchange poll timeout, ms. Was hard-coded 30000; round-15 found a peer
 * rank JIT-compiling a new shape (20-40 s) can stall the other rank's
 * lanex exchange past 30 s, producing a spurious ETIMEDOUT (errno=110)
 * engine death. Configurable via LANEX_TIMEOUT_MS; default 600000 (10 min)
 * because 1M-context prefill chunks and cold JIT compiles can be slow. */
#define POLL_MS_DEFAULT 600000
static int g_poll_ms = 0;
static inline int poll_ms(void) {
  int v = g_poll_ms;
  if (v == 0) {
    const char *e = getenv("LANEX_TIMEOUT_MS");
    v = e ? atoi(e) : POLL_MS_DEFAULT;
    if (v < 1000) v = POLL_MS_DEFAULT;
    g_poll_ms = v;
  }
  return v;
}

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

/* Per-socket MSG_ZEROCOPY bookkeeping (persists across calls, like the socket). */
struct zc_state {
  uint64_t issued;   /* successful zerocopy send() calls (each gets one id) */
  uint64_t done;     /* ids completed */
  uint64_t notifs;   /* completion notifications read */
  uint64_t copied;   /* notifications flagged ZEROCOPY_COPIED (kernel fell back to copy) */
};

struct lane_ctx {
  int fd;
  int zc;
  struct zc_state *z;
  unsigned char *send_ptr;
  unsigned char *recv_ptr;
  size_t len;
  double t_first;  /* time of first received byte, 0 = none yet */
};

/* LANEX_SOCKBUF: SO_SNDBUF/SO_RCVBUF size, e.g. 8M, 4096K, 33554432.
 * Unset = 32M (v1-v3 behaviour). 0 = don't set it (kernel autotuning). */
static long sockbuf_bytes(void) {
  const char *s = getenv("LANEX_SOCKBUF");
  if (!s || !*s) return 32L << 20;
  char *e;
  long v = strtol(s, &e, 10);
  if (*e == 'K' || *e == 'k') v <<= 10;
  else if (*e == 'M' || *e == 'm') v <<= 20;
  return v < 0 ? 32L << 20 : v;
}

static void tune(int fd) {
  int one = 1;
  long b = sockbuf_bytes();
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  if (b > 0) {
    int big = (int)b;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &big, sizeof big);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &big, sizeof big);
  }
}

/* Read every pending zerocopy completion on fd without blocking. */
static int zc_drain(int fd, struct zc_state *z) {
  for (;;) {
    char control[256];
    struct msghdr msg;
    memset(&msg, 0, sizeof msg);
    msg.msg_control = control;
    msg.msg_controllen = sizeof control;
    int r = recvmsg(fd, &msg, MSG_ERRQUEUE | MSG_DONTWAIT);
    if (r < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
      return errno;
    }
    for (struct cmsghdr *cm = CMSG_FIRSTHDR(&msg); cm; cm = CMSG_NXTHDR(&msg, cm)) {
      if (!((cm->cmsg_level == SOL_IP && cm->cmsg_type == IP_RECVERR) ||
            (cm->cmsg_level == SOL_IPV6 && cm->cmsg_type == IPV6_RECVERR)))
        continue;
      struct sock_extended_err *e = (struct sock_extended_err *)CMSG_DATA(cm);
      if (e->ee_origin != SO_EE_ORIGIN_ZEROCOPY) {
        if (e->ee_errno) return e->ee_errno;
        continue;
      }
      uint32_t lo = e->ee_info, hi = e->ee_data;
      z->done += (uint64_t)(hi - lo) + 1;
      z->notifs++;
      if (e->ee_code & SO_EE_CODE_ZEROCOPY_COPIED) z->copied++;
    }
  }
}

static void *lane_worker(void *arg) {
  struct lane_ctx *c = arg;
  size_t sent = 0, got = 0;
  int sflags = MSG_DONTWAIT | MSG_NOSIGNAL | (c->zc ? MSG_ZEROCOPY : 0);
  while (sent < c->len || got < c->len) {
    struct pollfd p = {.fd = c->fd, .events = 0};
    if (sent < c->len) p.events |= POLLOUT;
    if (got < c->len) p.events |= POLLIN;
    int r = poll(&p, 1, poll_ms());
    if (r < 0) { if (errno == EINTR) continue; return (void*)(intptr_t)errno; }
    if (r == 0) return (void*)(intptr_t)ETIMEDOUT;
    if (c->zc && (p.revents & POLLERR)) {
      int e = zc_drain(c->fd, c->z);
      if (e) return (void*)(intptr_t)e;
    }
    if (sent < c->len && (p.revents & POLLOUT)) {
      ssize_t k = send(c->fd, c->send_ptr + sent, c->len - sent, sflags);
      if (k > 0) {
        sent += k;
        if (c->zc) c->z->issued++;
      } else if (k < 0 && errno == ENOBUFS && c->zc) {
        /* too many pinned pages / notification skbs in flight: reap, then retry */
        int e = zc_drain(c->fd, c->z);
        if (e) return (void*)(intptr_t)e;
        if (c->z->done >= c->z->issued) return (void*)(intptr_t)ENOBUFS;
        struct pollfd q = {.fd = c->fd, .events = 0};
        poll(&q, 1, 1);
      } else if (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        return (void*)(intptr_t)errno;
      }
    }
    if (got < c->len && (p.revents & POLLIN)) {
      ssize_t k = recv(c->fd, c->recv_ptr + got, c->len - got, MSG_DONTWAIT);
      if (k > 0) {
        if (got == 0) c->t_first = now_us();
        got += k;
      }
      else if (k == 0) return (void*)(intptr_t)ECONNRESET;
      else if (errno != EAGAIN && errno != EWOULDBLOCK)
        return (void*)(intptr_t)errno;
    }
  }
  /* The send buffer may be overwritten as soon as we return: wait until the NIC
   * has finished with every zerocopy page. */
  while (c->zc && c->z->done < c->z->issued) {
    struct pollfd p = {.fd = c->fd, .events = 0};
    int r = poll(&p, 1, poll_ms());
    if (r < 0) { if (errno == EINTR) continue; return (void*)(intptr_t)errno; }
    if (r == 0) return (void*)(intptr_t)ETIMEDOUT;
    int e = zc_drain(c->fd, c->z);
    if (e) return (void*)(intptr_t)e;
  }
  return NULL;
}

/* Persistent handle: sockets stay open across calls. */
typedef struct {
  int nlanes;
  int fds[MAXL];
  pthread_mutex_t mu;
  double last_wait_us;
  double last_total_us;
  int zc;
  int zc_enabled_on_sockets;
  struct zc_state zs[MAXL];
} lanex_h;

lanex_h *lanex_create(int nlanes, const char **local_ips, const char **peer_ips,
                      int base_port, int is_server) {
  if (nlanes < 1 || nlanes > MAXL) return NULL;
  lanex_h *h = calloc(1, sizeof(lanex_h));
  h->nlanes = nlanes;
  pthread_mutex_init(&h->mu, NULL);
  for (int i = 0; i < nlanes; i++) {
    struct sockaddr_in a = {.sin_family = AF_INET};
    inet_pton(AF_INET, local_ips[i], &a.sin_addr);
    if (is_server) {
      int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
      setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
      a.sin_port = htons(base_port + i);
      if (bind(ls, (void *)&a, sizeof a)) { free(h); return NULL; }
      if (listen(ls, 1)) { free(h); return NULL; }
      int c = accept(ls, NULL, NULL);
      close(ls);
      if (c < 0) { free(h); return NULL; }
      tune(c);
      h->fds[i] = c;
    } else {
      struct sockaddr_in pa = {.sin_family = AF_INET, .sin_port = htons(base_port + i)};
      inet_pton(AF_INET, peer_ips[i], &pa.sin_addr);
      int s = socket(AF_INET, SOCK_STREAM, 0);
      int ok = 0;
      for (int t = 0; t < 900; t++) {
        if (connect(s, (void *)&pa, sizeof pa) == 0) { ok = 1; break; }
        usleep(100000);
      }
      if (!ok) { close(s); free(h); return NULL; }
      tune(s);
      h->fds[i] = s;
    }
  }
  return h;
}

/* Turn MSG_ZEROCOPY sends on (1) or off (0) for later lanex_xchg calls.
 * Returns 0 or the setsockopt errno. Only the sending side changes, so the two
 * ranks may differ. */
int lanex_set_zerocopy(lanex_h *h, int on) {
  pthread_mutex_lock(&h->mu);
  int rc = 0;
  if (on && !h->zc_enabled_on_sockets) {
    int one = 1;
    for (int i = 0; i < h->nlanes; i++)
      if (setsockopt(h->fds[i], SOL_SOCKET, SO_ZEROCOPY, &one, sizeof one)) { rc = errno; break; }
    if (!rc) h->zc_enabled_on_sockets = 1;
  }
  if (!rc) h->zc = on ? 1 : 0;
  pthread_mutex_unlock(&h->mu);
  return rc;
}

/* Effective buffer sizes on lane 0 (the kernel doubles and caps the request). */
void lanex_sockbuf(lanex_h *h, int *snd, int *rcv) {
  socklen_t l = sizeof(int);
  if (snd) getsockopt(h->fds[0], SOL_SOCKET, SO_SNDBUF, snd, &l);
  l = sizeof(int);
  if (rcv) getsockopt(h->fds[0], SOL_SOCKET, SO_RCVBUF, rcv, &l);
}

/* Cumulative zerocopy completion counts over all lanes. */
void lanex_zc_stats(lanex_h *h, uint64_t *notifs, uint64_t *copied) {
  uint64_t n = 0, c = 0;
  for (int i = 0; i < h->nlanes; i++) { n += h->zs[i].notifs; c += h->zs[i].copied; }
  if (notifs) *notifs = n;
  if (copied) *copied = c;
}

/* Exchange nbytes: send `sendbuf`, land peer's bytes in `recvbuf`.
 * Splits across lanes; one pthread per lane. Returns 0 or errno. */
int lanex_xchg(lanex_h *h, unsigned char *sendbuf, unsigned char *recvbuf, size_t nbytes) {
  pthread_t th[MAXL];
  struct lane_ctx ctx[MAXL];
  size_t base = nbytes / h->nlanes, rem = nbytes % h->nlanes, off = 0;
  int started = 0, rc = 0;

  pthread_mutex_lock(&h->mu);
  double t0 = now_us();
  for (int i = 0; i < h->nlanes; i++) {
    size_t ln = base + (i < (int)rem ? 1 : 0);
    ctx[i] = (struct lane_ctx){.fd = h->fds[i],
                              .zc = h->zc,
                              .z = &h->zs[i],
                              .send_ptr = sendbuf + off,
                              .recv_ptr = recvbuf + off,
                              .len = ln,
                              .t_first = 0};
    off += ln;
    int e = pthread_create(&th[i], NULL, lane_worker, &ctx[i]);
    if (e != 0) { rc = e; break; }
    started++;
  }
  for (int i = 0; i < started; i++) {
    void *rr = NULL;
    pthread_join(th[i], &rr);
    if (rr != NULL && rc == 0) rc = (int)(intptr_t)rr;
  }
  double first = 0;
  for (int i = 0; i < started; i++)
    if (ctx[i].t_first > 0 && (first == 0 || ctx[i].t_first < first)) first = ctx[i].t_first;
  double t1 = now_us();
  h->last_wait_us = first > 0 ? first - t0 : t1 - t0;
  h->last_total_us = t1 - t0;
  pthread_mutex_unlock(&h->mu);
  return rc;
}

/* Timing of the most recent lanex_xchg on this handle. Call from the same thread
 * right after lanex_xchg returns. */
void lanex_last_stats(lanex_h *h, double *wait_us, double *total_us) {
  if (wait_us) *wait_us = h->last_wait_us;
  if (total_us) *total_us = h->last_total_us;
}

void lanex_destroy(lanex_h *h) {
  if (!h) return;
  for (int i = 0; i < h->nlanes; i++) close(h->fds[i]);
  pthread_mutex_destroy(&h->mu);
  free(h);
}
