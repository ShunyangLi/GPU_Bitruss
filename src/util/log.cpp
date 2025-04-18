/**
 * @file log.cpp
 *
 * @brief logger file
 * 
 * @author rxi
 * Source: https://github.com/rxi/log.c
 * @date on 2023/10/30.
 */

#include "log.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>

using namespace std;
using namespace std::chrono;

std::mutex global_log_mutex;
time_point<high_resolution_clock> clk_beg = high_resolution_clock::now();
static struct {
    void *udata;
    log_LockFn lock;
    FILE *fp;
    int level;
    int quiet;
} L;


static const char *level_names[] = {
        "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "SUCC"};
#ifndef LOG_USE_COLOR
#define LOG_USE_COLOR
#endif
#ifdef LOG_USE_COLOR
static const char *level_colors[] = {
        "\x1b[94m", "\x1b[36m", "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[35m", "\x1b[92m"};
#endif


static void lock() {
    if (L.lock) {
        L.lock(L.udata, reinterpret_cast<void *>(1));
    }
}


static void unlock() {
    if (L.lock) {
        L.lock(L.udata, nullptr);
    }
}


void log_set_udata(void *udata) {
    L.udata = udata;
}


void log_set_lock(log_LockFn fn) {
    L.lock = fn;
}


void log_set_fp(FILE *fp) {
    L.fp = fp;
}


void log_set_level(int level) {
    L.level = level;
}


void log_set_quiet(int enable) {
    L.quiet = enable ? 1 : 0;
}


void log_log(int level, const char *file, int line, const char *fmt, ...) {
    if (level < L.level) {
        return;
    }

    using namespace std::chrono;
    time_point<high_resolution_clock> clock_now = high_resolution_clock::now();
    auto elapsed_time = duration_cast<nanoseconds>(clock_now - clk_beg).count();
    {
        unique_lock<mutex> lock_global(global_log_mutex);
        /* Acquire lock */
        lock();

        /* Get current time */
        time_t t = time(nullptr);
        struct tm *lt = localtime(&t);

        /* Log to stderr */
        if (!L.quiet) {
            va_list args;
            char buf[64];
            buf[strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", lt)] = '\0';
#ifdef LOG_USE_COLOR
            fprintf(
                    stdout, "%s %s%-4s \x1b[0m\x1b[90m%s:%d:\x1b[0m ",
                    buf, level_colors[level], level_names[level], file, line);
#else
            fprintf(stdout, "%s %-5s %s:%d: ", buf, level_names[level], file, line);
#endif
            va_start(args, fmt);
            vfprintf(stdout, fmt, args);
            va_end(args);
            fprintf(stdout, "\n");
        }

        /* Log to file */
        if (L.fp) {
            va_list args;
            char buf[32];
            buf[strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", lt)] = '\0';
            fprintf(L.fp, "%s %-5s (ts: %.6lf s,  et: %.6lf s) %s:%d: ", buf, level_names[level],
                    duration_cast<nanoseconds>(clock_now.time_since_epoch()).count() / pow(10, 9),
                    elapsed_time / pow(10, 9), file, line);
            va_start(args, fmt);
            vfprintf(L.fp, fmt, args);
            va_end(args);
            fprintf(L.fp, "\n");
        }

        /* Release lock */
        unlock();
    }
}