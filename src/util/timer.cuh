#pragma once
#ifndef BITRUSS_TIMER_CUH
#define BITRUSS_TIMER_CUH

#include <chrono>
#include <iostream>

#include "cuda_runtime.h"

// cpu timer class
class Timer {
public:
    Timer() : beg_(clock_::now()) {}

    auto reset() -> void { beg_ = clock_::now(); }

    [[nodiscard]] auto elapsed() const -> double {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

    auto elapsed_and_reset() -> double {
        double elapsed = std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
        beg_ = clock_::now();
        return elapsed;
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

// cuda timer class
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);

        cudaEventRecord(_start);
    }

    auto reset() -> void {
        cudaEventRecord(_start);
    }

    auto start() -> void {
        cudaEventRecord(_start);
    }

    [[nodiscard]] auto elapsed() const -> float {
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, _start, _stop);

        return elapsedTime / 1000;
    }

    [[nodiscard]] auto elapsed_and_reset() -> float {
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, _start, _stop);
        cudaEventRecord(_start);

        return elapsedTime / 1000;
    }

    ~CudaTimer() {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

private:
    cudaEvent_t _start{}, _stop{};
};


#endif//BITRUSS_TIMER_CUH