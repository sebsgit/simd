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
        REQUIRE(simdu16x8(data.data(), data.data() + 4).to_array() == data);
    }
    SECTION("arthmetic")
    {
        simdu16x8 result = simdu16x8(5) + simdu16x8(1, 2, 3, 4, 5, 6, 7, 8);
        REQUIRE(result.to_array() == std::array<uint16_t, 8>{ 6, 7, 8, 9, 10, 11, 12, 13 });

        result = simdu16x8(4) - simdu16x8(0, 1, 2, 3, 4, 5, 6, 7);
        REQUIRE(result.to_array() == std::array<uint16_t, 8>{ 4, 3, 2, 1, 0, 0, 0, 0 });

        result = simdu16x8(4) * simdu16x8(0, 1, 2, 3, 4, 5, 6, 7);
        REQUIRE(result.to_array() == std::array<uint16_t, 8>{ 0, 4, 8, 12, 16, 20, 24, 28 });

        REQUIRE(result.sum() == 112);
        REQUIRE(simdu16x8().sum() == 0);
        REQUIRE(simdu16x8(1).sum() == 8);
    }
}
