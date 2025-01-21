#include <smmintrin.h> // For _mm_stream_load_si128
#include <emmintrin.h> // For _mm_mul_ps
#include <assert.h>
#include <stdint.h>

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

    size_t num_iterations = N / 4; // 4 integers fit into a 128-bit register

    __m128 scale_vec = _mm_set1_ps(scale);

    for (size_t i = 0; i < num_iterations; ++i)
    {
        __m128 data_x = _mm_load_ss(&X[4 * i]);
        __m128 data_y = _mm_load_ss(&Y[4 * i]);

        __m128 data_z = _mm_add_ps(_mm_mul_ps(scale_vec, data_x), data_y);

        _mm_stream_ps(&result[4 * i], data_z);
    }
}
