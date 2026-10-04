// lane_xchg.c -- measure the floor of a 2-rank "exchange-and-add" all-reduce
// over N kernel-TCP lanes, with no NCCL, no GPU, no XDP.
//
// Each iteration, BOTH sides send their whole buffer (split evenly across the
// lanes) and receive the peer's whole buffer, then optionally add it in. That
// is one network trip per all-reduce, versus the two dependent trips a 2-rank
// ring (reduce-scatter + all-gather) needs. Iteration time ~= one-way latency +
// (bytes / lanes) serialisation + host overhead.
//
// Compare its p50 at 70 KB against nccl_ar_sweep.py's 194 us to see how much of
// NCCL's per-op cost is the network vs NCCL's proxy/protocol machinery.
//
// Build:  gcc -O2 -o lane_xchg lane_xchg.c
// Run (start the server side first):
//   thorc2: ./lane_xchg -r server -l 10.0.0.2,10.0.1.2,10.0.2.2,10.0.3.2 -s 71680 -n 5000
//   thorc1: ./lane_xchg -r client -l 10.0.0.1,10.0.1.1,10.0.2.1,10.0.3.1
//                                 -p 10.0.0.2,10.0.1.2,10.0.2.2,10.0.3.2 -s 71680 -n 5000
//           (one command line; split here for width)
// Use 1, 2 or 4 addresses in -l/-p to compare lane counts. Add -a to include
// the float add (CPU cost of reducing the received buffer).
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
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

int main(int argc, char **argv) {
  char *role = NULL, *lstr = NULL, *pstr = NULL;
  size_t size = 71680;
  int iters = 5000, warm = 200, port = 47000, doadd = 0, c;
  while ((c = getopt(argc, argv, "r:l:p:s:n:w:P:a")) != -1) {
    switch (c) {
      case 'r': role = optarg; break;
      case 'l': lstr = strdup(optarg); break;
      case 'p': pstr = strdup(optarg); break;
      case 's': size = strtoull(optarg, 0, 10); break;
      case 'n': iters = atoi(optarg); break;
      case 'w': warm = atoi(optarg); break;
      case 'P': port = atoi(optarg); break;
      case 'a': doadd = 1; break;
    }
  }
  if (!role || !lstr || (strcmp(role, "client") == 0 && !pstr)) {
    fprintf(stderr, "usage: %s -r server|client -l local_ips [-p peer_ips] [-s bytes] [-n iters] [-a]\n", argv[0]);
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
      if (bind(fd[i], (void *)&la, sizeof la)) die("bind local");   // pin egress lane
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

  size_t chunk[MAXL], off[MAXL];
  for (int i = 0, o = 0; i < L; i++) {
    chunk[i] = size / L + (i < (int)(size % L) ? 1 : 0);
    off[i] = o; o += chunk[i];
  }
  float *sbuf = aligned_alloc(64, (size + 63) & ~63UL), *rbuf = aligned_alloc(64, (size + 63) & ~63UL);
  memset(sbuf, 0, size); memset(rbuf, 0, size);
  double *t = malloc(sizeof(double) * iters);

  for (int it = -warm; it < iters; it++) {
    size_t sent[MAXL] = {0}, got[MAXL] = {0};
    int pending = 2 * L;
    double t0 = now_us();
    while (pending) {
      for (int i = 0; i < L; i++) {
        if (sent[i] < chunk[i]) {
          ssize_t k = send(fd[i], (char *)sbuf + off[i] + sent[i], chunk[i] - sent[i], MSG_DONTWAIT | MSG_NOSIGNAL);
          if (k > 0 && (sent[i] += k) == chunk[i]) pending--;
          else if (k < 0 && errno != EAGAIN) die("send");
        }
        if (got[i] < chunk[i]) {
          ssize_t k = recv(fd[i], (char *)rbuf + off[i] + got[i], chunk[i] - got[i], MSG_DONTWAIT);
          if (k > 0 && (got[i] += k) == chunk[i]) pending--;
          else if (k == 0) { fprintf(stderr, "peer closed\n"); return 1; }
          else if (k < 0 && errno != EAGAIN) die("recv");
        }
      }
    }
    if (doadd)
      for (size_t j = 0; j < size / sizeof(float); j++) sbuf[j] += rbuf[j];
    if (it >= 0) t[it] = now_us() - t0;
  }

  qsort(t, iters, sizeof(double), cmpd);
  double p50 = t[iters / 2], p90 = t[(int)(iters * 0.90)], p99 = t[(int)(iters * 0.99)];
  printf("%s lanes=%d size=%zu B iters=%d add=%d  p50 %.1f us  p90 %.1f us  p99 %.1f us  "
         "eff %.2f GB/s (one-way bytes / p50)\n",
         role, L, size, iters, doadd, p50, p90, p99, size / (p50 * 1e-6) / 1e9);
  for (int i = 0; i < L; i++) close(fd[i]);
  return 0;
}
