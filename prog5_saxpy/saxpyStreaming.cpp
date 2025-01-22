#include <smmintrin.h> // For _mm_stream_load_si128
#include <emmintrin.h> // For _mm_mul_ps
#include <immintrin.h>
#include <assert.h>
#include <stdint.h>

#define USE_AVX

extern void saxpySerial(int N,
                        float scale,
                        float X[],
                        float Y[],
                        float result[]);

void saxpyStreaming(int N,
                    float scale,
                    float X[],
                    float Y[],
                    float result[])
{
    // Replace this code with ones that make use of the streaming instructions
#ifdef USE_AVX
    size_t num_iterations = N / 8; // 8 integers fit into a 256-bit register

    // __m128 scale_vec = _mm_set1_ps(scale);
    __m256 scale_vec = _mm256_set1_ps(scale);

    for (size_t i = 0; i < num_iterations; ++i)
    {
        __m256 data_x = _mm256_loadu_ps(&X[8 * i]); // __m128 data_x = _mm_load_ss(&X[4 * i]);
        __m256 data_y = _mm256_loadu_ps(&Y[8 * i]); // __m128 data_y = _mm_load_ss(&Y[4 * i]);

        // __m128 data_z = _mm_add_ps(_mm_mul_ps(scale_vec, data_x), data_y);
        __m256 data_z = _mm256_add_ps(_mm256_mul_ps(scale_vec, data_x), data_y);

        // _mm_stream_ps(&result[4 * i], data_z);
        _mm256_stream_ps(&result[8 * i], data_z); // 32-byte aligned required
    }
#else
    size_t num_iterations = N / 4;

    __m128 scale_vec = _mm_set1_ps(scale);

    for (size_t i = 0; i < num_iterations; i++)
    {
        __m128 data_x = _mm_load_ss(&X[4 * i]);
        __m128 data_y = _mm_load_ss(&Y[4 * i]);

        __m128 data_z = _mm_add_ps(_mm_mul_ps(scale_vec, data_x), data_y);

        _mm_stream_ps(&result[4 * i], data_z);
    }
#endif
}
