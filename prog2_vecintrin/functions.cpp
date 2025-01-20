#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;

void absSerial(float *values, float *output, int N)
{
	for (int i = 0; i < N; i++)
	{
		float x = values[i];
		if (x < 0)
		{
			output[i] = -x;
		}
		else
		{
			output[i] = x;
		}
	}
}

// implementation of absolute value using 15418 instrinsics
void absVector(float *values, float *output, int N)
{
	__cmu418_vec_float x;
	__cmu418_vec_float result;
	__cmu418_vec_float zero = _cmu418_vset_float(0.f);
	__cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

	//  Note: Take a careful look at this loop indexing.  This example
	//  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
	//  Why is that the case?
	for (int i = 0; i < N; i += VECTOR_WIDTH)
	{
		// All ones
		maskAll = _cmu418_init_ones();

		// All zeros
		maskIsNegative = _cmu418_init_ones(0);

		// Load vector of values from contiguous memory addresses
		_cmu418_vload_float(x, values + i, maskAll); // x = values[i];

		// Set mask according to predicate
		_cmu418_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

		// Execute instruction using mask ("if" clause)
		_cmu418_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

		// Inverse maskIsNegative to generate "else" mask
		maskIsNotNegative = _cmu418_mask_not(maskIsNegative); // } else {

		// Execute instruction ("else" clause)
		_cmu418_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

		// Write results back to memory
		_cmu418_vstore_float(output + i, result, maskAll);
	}
}

// Accepts an array of values and an array of exponents
// For each element, compute values[i]^exponents[i] and clamp value to
// 4.18.  Store result in outputs.
// Uses iterative squaring, so that total iterations is proportional
// to the log_2 of the exponent
void clampedExpSerial(float *values, int *exponents, float *output, int N)
{
	for (int i = 0; i < N; i++)
	{
		float x = values[i];
		float result = 1.f;
		int y = exponents[i];
		float xpower = x;
		while (y > 0)
		{
			if (y & 0x1)
			{
				result *= xpower;
			}
			xpower = xpower * xpower;
			y >>= 1;
		}
		if (result > 4.18f)
		{
			result = 4.18f;
		}
		output[i] = result;
	}
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
	// Implement your vectorized version of clampedExpSerial here
	//  ...
	__cmu418_vec_float x;
	__cmu418_vec_int y, one = _cmu418_vset_int(1), zero = _cmu418_vset_int(0);
	__cmu418_vec_float result;
	__cmu418_vec_float upper_limit = _cmu418_vset_float(4.18f);
	__cmu418_mask mask = _cmu418_init_ones(), run, mask_clamp;

	auto compute_exp_vec = [&](__cmu418_mask &mask, int start)
	{
		// reset
		result = _cmu418_vset_float(1.f);

		// Load x and y
		_cmu418_vload_float(x, values + start, mask);
		_cmu418_vload_int(y, exponents + start, mask);

		// printf("x0, y0: %f, %d\n", x.value[0], y.value[0]);

		__cmu418_vec_float xpower;
		_cmu418_vmove_float(xpower, x, mask);

		_cmu418_vgt_int(run, y, zero, mask);

		while (_cmu418_cntbits(run))
		{
			// y & 0x1
			__cmu418_vec_int run_mul;
			_cmu418_vbitand_int(run_mul, y, one, mask);

			__cmu418_mask y_mask;
			_cmu418_veq_int(y_mask, run_mul, one, mask);

			// result *= xpower;
			_cmu418_vmult_float(result, result, xpower, y_mask);

			// xpower = xpower * xpower;
			_cmu418_vmult_float(xpower, xpower, xpower, mask);

			// y >>= 1
			_cmu418_vshiftright_int(y, y, one, mask);

			// update run
			_cmu418_vgt_int(run, y, zero, mask);
		}

		_cmu418_vgt_float(mask_clamp, result, upper_limit, mask);
		_cmu418_vmove_float(result, upper_limit, mask_clamp);

		// output[i] = result;
		_cmu418_vstore_float(output + start, result, mask);
	};

	if (N % VECTOR_WIDTH != 0)
	{
		for (int i = 0; i < N - VECTOR_WIDTH; i += VECTOR_WIDTH)
		{
			compute_exp_vec(mask, i);
		}

		for (int i = N - VECTOR_WIDTH; i < N; i += VECTOR_WIDTH)
		{
			mask = _cmu418_init_ones(N % VECTOR_WIDTH);
			mask = _cmu418_mask_not(mask); // important!!
			// printf("mask cnt 1: %d\n", _cmu418_cntbits(mask));
			compute_exp_vec(mask, i);
		}
	}
	else
	{
		for (int i = 0; i < N; i += VECTOR_WIDTH)
		{
			compute_exp_vec(mask, i);
		}
	}
}

float arraySumSerial(float *values, int N)
{
	float sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum += values[i];
	}

	return sum;
}

// Assume N % VECTOR_WIDTH == 0
// Assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
	// Implement your vectorized version here
	//  ...
	return 0.f;
}
