#include "catch.hpp"

#include "simd/simdu16x8.hpp"
#include <iostream>

TEST_CASE("simd u16 x 8")
{
    SECTION("load / store")
    {
        std::array<uint16_t, 8> data = { 0, 1, 2, 3, 4, 5, 6, 7 };
        alignas(16) std::array<uint16_t, 8> result = {};
        simdu16x8 s(data.data());
        s.store(result.data());
        REQUIRE(data == result);
        REQUIRE(simdu16x8(5).to_array() == std::array<uint16_t, 8>{ 5, 5, 5, 5, 5, 5, 5, 5 });
        REQUIRE(simdu16x8(5, 4, 3, 7, 1, 0, 0, 2).to_array() == std::array<uint16_t, 8>{ 5, 4, 3, 7, 1, 0, 0, 2 });
    }
}
