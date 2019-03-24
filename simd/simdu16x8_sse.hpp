#pragma once

#include "simd_base.hpp"

template <>
class simd<uint16_t, 8> : public simd_common<simd<uint16_t, 8>> {
public:
    using type = uint16_t;
    static constexpr size_t value_count = 8;

    simd() noexcept
        : _d(_mm_setzero_si128())
    {
    }
    explicit simd(uint16_t value) noexcept
        : _d(_mm_set1_epi16(value)) //TODO: avoid conversion to short
    {
    }
    simd(uint16_t s0, uint16_t s1, uint16_t s2, uint16_t s3,
        uint16_t s4, uint16_t s5, uint16_t s6, uint16_t s7) noexcept
        : _d(_mm_setr_epi16(s0, s1, s2, s3, s4, s5, s6, s7)) //TODO: avoid conversion to short
    {
    }
    explicit simd(__m128i value) noexcept
        : _d(value)
    {
    }
    explicit simd(const uint16_t* input) noexcept
        : _d(is_aligned(input, 16) ? _mm_load_si128(reinterpret_cast<const __m128i*>(input))
                                   : _mm_loadu_si128(reinterpret_cast<const __m128i*>(input)))
    {
    }

    explicit simd(const uint16_t* inputLo, const uint16_t* inputHi) noexcept
        : simd(inputLo[0], inputLo[1], inputLo[2], inputLo[3],
              inputHi[0], inputHi[1], inputHi[2], inputHi[3])
    {
    }

    void store(uint16_t* output) const noexcept
    {
        if (is_aligned(output, 16)) {
            this->store_aligned(output);
        } else {
            alignas(16) uint16_t tmp[8];
            this->store_aligned(tmp);
            memcpy(output, tmp, 8 * sizeof(tmp[0]));
        }
    }

    void store_aligned(uint16_t* output) const noexcept
    {
        _mm_store_si128(reinterpret_cast<__m128i*>(output), this->_d);
    }

private:
    __m128i _d;
};

using simdu16x8 = simd<uint16_t, 8>;