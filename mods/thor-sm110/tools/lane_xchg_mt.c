// lane_xchg_mt.c -- like lane_xchg.c, but one thread per lane.
//
// Same protocol and output as lane_xchg: each iteration BOTH sides send their
// whole buffer (split evenly across the lanes) and receive the peer's whole
// buffer. Here each lane's socket is driven by its own thread, so the result
// shows whether lane_xchg's single thread (one core doing every send/recv copy)
// was the limit at large sizes.
//
// Build:  gcc -O2 -pthread -o lane_xchg_mt lane_xchg_mt.c
// Run (server side first, same flags as lane_xchg; ports default to 48000+i
// so it can't collide with a lane_xchg run):
//   thorc2: ./lane_xchg_mt -r server -l 10.0.0.2,10.0.1.2,10.0.2.2,10.0.3.2 -s 33554432 -n 300
//   thorc1: ./lane_xchg_mt -r client -l 10.0.0.1,10.0.1.1,10.0.2.1,10.0.3.1 \
//                          -p 10.0.0.2,10.0.1.2,10.0.2.2,10.0.3.2 -s 33554432 -n 300
// Optional: -c FIRSTCPU pins lane thread i to CPU FIRSTCPU+i.
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define MAXL 8

static int split(char *s, char **out) {
  int n = 0;
  for (char *t = strtok(s, ","); t && n < MAXL; t = strtok(NULL, ",")) out[n++] = t;
  return n;
}
static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static void die(const char *m) { perror(m); exit(1); }

static void tune(int fd) {
  int one = 1, big = 8 << 20;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &big, sizeof big);
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &big, sizeof big);
}

struct lane {
  int id, fd, cpu;
  char *sbuf, *rbuf;
  size_t chunk;
  int total;                 // warm + iters
  pthread_barrier_t *start, *done;
};

static void *lane_main(void *arg) {
  struct lane *ln = arg;
  if (ln->cpu >= 0) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(ln->cpu, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
  }
  for (int it = 0; it < ln->total; it++) {
    pthread_barrier_wait(ln->start);
    size_t sent = 0, got = 0;
    while (sent < ln->chunk || got < ln->chunk) {
      struct pollfd p = {.fd = ln->fd, .events = 0};
      if (sent < ln->chunk) p.events |= POLLOUT;
      if (got < ln->chunk) p.events |= POLLIN;
      if (poll(&p, 1, 1000) < 0 && errno != EINTR) die("poll");
      if (sent < ln->chunk && (p.revents & (POLLOUT | POLLERR))) {
        ssize_t k = send(ln->fd, ln->sbuf + sent, ln->chunk - sent, MSG_DONTWAIT | MSG_NOSIGNAL);
        if (k > 0) sent += k;
        else if (k < 0 && errno != EAGAIN) die("send");
      }
      if (got < ln->chunk && (p.revents & (POLLIN | POLLERR | POLLHUP))) {
        ssize_t k = recv(ln->fd, ln->rbuf + got, ln->chunk - got, MSG_DONTWAIT);
        if (k > 0) got += k;
        else if (k == 0) { fprintf(stderr, "lane %d: peer closed\n", ln->id); exit(1); }
        else if (errno != EAGAIN) die("recv");
      }
    }
    pthread_barrier_wait(ln->done);
  }
  return NULL;
}

int main(int argc, char **argv) {
  char *role = NULL, *lstr = NULL, *pstr = NULL;
  size_t size = 71680;
  int iters = 5000, warm = 200, port = 48000, cpu0 = -1, c;
  while ((c = getopt(argc, argv, "r:l:p:s:n:w:P:c:")) != -1) {
    switch (c) {
      case 'r': role = optarg; break;
      case 'l': lstr = strdup(optarg); break;
      case 'p': pstr = strdup(optarg); break;
      case 's': size = strtoull(optarg, 0, 10); break;
      case 'n': iters = atoi(optarg); break;
      case 'w': warm = atoi(optarg); break;
      case 'P': port = atoi(optarg); break;
      case 'c': cpu0 = atoi(optarg); break;
    }
  }
  if (!role || !lstr || (strcmp(role, "client") == 0 && !pstr)) {
    fprintf(stderr, "usage: %s -r server|client -l local_ips [-p peer_ips] [-s bytes] [-n iters] [-w warm] [-c firstcpu]\n", argv[0]);
    return 2;
  }
  char *lip[MAXL], *pip[MAXL];
  int L = split(lstr, lip);
  if (pstr && split(pstr, pip) != L) { fprintf(stderr, "-l and -p need the same count\n"); return 2; }
  int server = strcmp(role, "server") == 0;

  int fd[MAXL];
  for (int i = 0; i < L; i++) {
    struct sockaddr_in la = {.sin_family = AF_INET};
    inet_pton(AF_INET, lip[i], &la.sin_addr);
    if (server) {
      int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
      setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
      la.sin_port = htons(port + i);
      if (bind(ls, (void *)&la, sizeof la)) die("bind");
      if (listen(ls, 1)) die("listen");
      fd[i] = accept(ls, NULL, NULL);
      if (fd[i] < 0) die("accept");
      close(ls);
    } else {
      fd[i] = socket(AF_INET, SOCK_STREAM, 0);
      la.sin_port = 0;
      if (bind(fd[i], (void *)&la, sizeof la)) die("bind local");
      struct sockaddr_in pa = {.sin_family = AF_INET, .sin_port = htons(port + i)};
      inet_pton(AF_INET, pip[i], &pa.sin_addr);
      for (int t = 0;; t++) {
        if (connect(fd[i], (void *)&pa, sizeof pa) == 0) break;
        if (t > 100) die("connect");
        usleep(100000);
      }
    }
    tune(fd[i]);
    fcntl(fd[i], F_SETFL, fcntl(fd[i], F_GETFL) | O_NONBLOCK);
  }

  char *sbuf = aligned_alloc(64, (size + 63) & ~63UL), *rbuf = aligned_alloc(64, (size + 63) & ~63UL);
  memset(sbuf, 0, size); memset(rbuf, 0, size);

  pthread_barrier_t start, done;
  pthread_barrier_init(&start, NULL, L + 1);
  pthread_barrier_init(&done, NULL, L + 1);
  struct lane ln[MAXL];
  pthread_t th[MAXL];
  size_t off = 0;
  for (int i = 0; i < L; i++) {
    size_t ch = size / L + (i < (int)(size % L) ? 1 : 0);
    ln[i] = (struct lane){.id = i, .fd = fd[i], .cpu = (cpu0 >= 0 ? cpu0 + i : -1),
                          .sbuf = sbuf + off, .rbuf = rbuf + off, .chunk = ch,
                          .total = warm + iters, .start = &start, .done = &done};
    off += ch;
    pthread_create(&th[i], NULL, lane_main, &ln[i]);
  }

  double *t = malloc(sizeof(double) * iters);
  for (int it = -warm; it < iters; it++) {
    double t0 = now_us();
    pthread_barrier_wait(&start);
    pthread_barrier_wait(&done);
    if (it >= 0) t[it] = now_us() - t0;
  }
  for (int i = 0; i < L; i++) pthread_join(th[i], NULL);

  qsort(t, iters, sizeof(double), cmpd);
  double p50 = t[iters / 2], p90 = t[(int)(iters * 0.90)], p99 = t[(int)(iters * 0.99)];
  printf("%s mt lanes=%d size=%zu B iters=%d  p50 %.1f us  p90 %.1f us  p99 %.1f us  "
         "eff %.2f GB/s (one-way bytes / p50)\n",
         role, L, size, iters, p50, p90, p99, size / (p50 * 1e-6) / 1e9);
  for (int i = 0; i < L; i++) close(fd[i]);
  return 0;
}
