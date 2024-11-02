/**
 * @file timer.cuh
 * @brief Timer classes for measuring execution time on CPU and CUDA.
 *
 * This header file contains two classes: `Timer` and `CudaTimer`.
 * The `Timer` class utilizes C++ standard library features to provide
 * high-resolution timing capabilities for CPU processes. The `CudaTimer`
 * class leverages CUDA events to measure execution time for CUDA
 * kernels efficiently.
 *
 * Usage:
 * - Create an instance of `Timer` for CPU timing or `CudaTimer` for
 *   CUDA kernel timing.
 * - Use the `elapsed` or `elapsed_and_reset` methods to retrieve the
 *   elapsed time as needed.
 * - Make sure to manage CUDA resources properly; the `CudaTimer`
 *   class will automatically clean up upon destruction.
 *
 * @note This file should be included in source files that require
 *       timing functionalities for performance measurement.
 */

#pragma once
#ifndef BITRUSS_TIMER_CUH
#define BITRUSS_TIMER_CUH

#include <chrono>
#include <iostream>

#include "cuda_runtime.h"


/**
 * @class Timer
 * @brief A high-resolution timer class for CPU timing measurements.
 *
 * @details This class provides functionalities to measure elapsed time using the
 * high-resolution clock provided by the C++ standard library. It can reset
 * the timer and retrieve the elapsed time since its last reset.
 *
 * @note This class is useful for profiling CPU algorithms and measuring
 * performance in time-critical applications.
 */
class Timer {
public:
    /**
     * @brief Constructs a Timer object and initializes the start time.
     */
    Timer() : beg_(clock_::now()) {}

    /**
     * @brief Resets the timer to the current time.
     *
     * This method updates the start time to the current time, effectively
     * resetting the timer.
     */
    auto reset() -> void { beg_ = clock_::now(); }

    /**
     * @brief Retrieves the elapsed time since the last reset.
     *
     * @return The elapsed time in seconds.
     */
    [[nodiscard]] auto elapsed() const -> double {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

    /**
     * @brief Retrieves the elapsed time since the last reset and resets the timer.
     *
     * @return The elapsed time in seconds before resetting the timer.
     */
    auto elapsed_and_reset() -> double {
        double elapsed = std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
        beg_ = clock_::now();
        return elapsed;
    }

private:
    typedef std::chrono::high_resolution_clock clock_; ///< Type alias for high-resolution clock.
    typedef std::chrono::duration<double, std::ratio<1>> second_; ///< Type alias for seconds duration.
    std::chrono::time_point<clock_> beg_; ///< The starting time point.
};


/**
 * @class CudaTimer
 * @brief A timer class for CUDA kernel execution timing.
 *
 * @details This class provides functionalities to measure elapsed time for CUDA
 * kernel execution using CUDA events. It can reset the timer and retrieve
 * the elapsed time since its last reset.
 *
 * @note This class is essential for profiling GPU algorithms and measuring
 * performance in CUDA applications.
 */
class CudaTimer {
public:
    /**
     * @brief Constructs a CudaTimer object and initializes CUDA events.
     *
     * This constructor creates two CUDA events for measuring time and records
     * the start time.
     */
    CudaTimer() {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);

        cudaEventRecord(_start);
    }

    /**
     * @brief Resets the timer to the current time.
     *
     * This method records the current time as the start time, effectively
     * resetting the timer.
     */
    auto reset() -> void {
        cudaEventRecord(_start);
    }

    /**
     * @brief Starts the timer by recording the current time.
     *
     * This method records the current time as the start time for timing
     * the CUDA operations.
     */
    auto start() -> void {
        cudaEventRecord(_start);
    }

    /**
     * @brief Retrieves the elapsed time since the last reset.
     *
     * @return The elapsed time in seconds since the last reset.
     */
    [[nodiscard]] auto elapsed() const -> float {
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, _start, _stop);

        return elapsedTime / 1000; // Convert milliseconds to seconds
    }

    /**
     * @brief Retrieves the elapsed time since the last reset and resets the timer.
     *
     * @return The elapsed time in seconds before resetting the timer.
     */
    [[nodiscard]] auto elapsed_and_reset() -> float {
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, _start, _stop);
        cudaEventRecord(_start);

        return elapsedTime / 1000; // Convert milliseconds to seconds
    }

    /**
     * @brief Destroys the CUDA events.
     *
     * This destructor releases the resources allocated for the CUDA events
     * when the CudaTimer object is destroyed.
     */
    ~CudaTimer() {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

private:
    cudaEvent_t _start{}, _stop{}; ///< CUDA event handles for timing.
};


#endif//BITRUSS_TIMER_CUH