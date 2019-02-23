#include "catch.hpp"

#include "simd/simd_base.hpp"
#include "simd/simdf4.hpp"
#include <iostream>

TEST_CASE("simd float x4")
{
    SECTION("basics")
    {
        static_assert(simdf4::bit_count() == 128, "");
        simdf4 a(5);
        simdf4 b{ 4, 5, 6, 7 };
        auto c = a + b;

        auto res = c.to_array();
        REQUIRE(res[0] == Approx(9));
        REQUIRE(res[1] == Approx(10));
        REQUIRE(res[2] == Approx(11));
        REQUIRE(res[3] == Approx(12));

        c /= simdf4(2.0f);
        c = c * simdf4(4.0f);
        c = c - a;
        c.store(res.data());
        REQUIRE(res[0] == Approx(13));
        REQUIRE(res[1] == Approx(15));
        REQUIRE(res[2] == Approx(17));
        REQUIRE(res[3] == Approx(19));
        c = simdf4{ -1, 2, 0, 4 };
        b = simdf4{ -2, 0, 1, 3 };
        REQUIRE(c.min(b).to_array() == std::array<float, 4>{ -2, 0, 0, 3 });
        REQUIRE(c.max(b).to_array() == std::array<float, 4>{ -1, 2, 1, 4 });

        res = simdf4{ 2, 3, 4, 9 }.sqrt().to_array();
        REQUIRE(res[0] == Approx(sqrt(2)));
        REQUIRE(res[1] == Approx(sqrt(3)));
        REQUIRE(res[2] == Approx(sqrt(4)));
        REQUIRE(res[3] == Approx(sqrt(9)));
    }
    SECTION("compare")
    {
        simdf4 a{ -2, 0, 1, 4 };
        simdf4 b{ -1, 9, 1, 78 };

        auto res = a.compare(b, simd_base::compare_flags::equal).to_array();
        REQUIRE(res[0] == 0);
        REQUIRE(res[1] == 0);
        REQUIRE(res[2] != 0);
        REQUIRE(res[3] == 0);

        res = a.compare(b, simd_base::compare_flags::lower).to_array();
        REQUIRE(res[0] != 0);
        REQUIRE(res[1] != 0);
        REQUIRE(res[2] == 0);
        REQUIRE(res[3] != 0);

        res = a.compare(b, simd_base::compare_flags::greater).to_array();
        REQUIRE(res[0] == 0);
        REQUIRE(res[1] == 0);
        REQUIRE(res[2] == 0);
        REQUIRE(res[3] == 0);

        res = a.compare(b, simd_base::compare_flags::not_equal).to_array();
        REQUIRE(res[0] != 0);
        REQUIRE(res[1] != 0);
        REQUIRE(res[2] == 0);
        REQUIRE(res[3] != 0);

        res = a.compare(b, simd_base::compare_flags::lower_equal).to_array();
        REQUIRE(res[0] != 0);
        REQUIRE(res[1] != 0);
        REQUIRE(res[2] != 0);
        REQUIRE(res[3] != 0);

        res = a.compare(b, simd_base::compare_flags::greater_equal).to_array();
        REQUIRE(res[0] == 0);
        REQUIRE(res[1] == 0);
        REQUIRE(res[2] != 0);
        REQUIRE(res[3] == 0);
    }
    SECTION("horizontal add")
    {
        simdf4 a{ 1, 2, 3, 4 };
        simdf4 b{ 5, 6, 7, 8 };
        auto res = simdf4::horizontal_add(a, b);
        REQUIRE(res.to_array() == std::array<float, 4>{ 3, 7, 11, 15 });
    }
    SECTION("unpack")
    {
        simdf4 a{ 1, 2, 3, 4 };
        simdf4 b{ 5, 6, 7, 8 };
        auto res = simdf4::unpack_low(a, b);
        REQUIRE(res.to_array() == std::array<float, 4>{ 1, 5, 2, 6 });
        res = simdf4::unpack_high(a, b);
        REQUIRE(res.to_array() == std::array<float, 4>{ 3, 7, 4, 8 });
    }
    SECTION("shuffle")
    {
        simdf4 a{ 1, 2, 3, 4 };
        simdf4 b{ 5, 6, 7, 8 };
        auto res = simdf4::shuffle<0, 1, 2, 3>(a, b);
        REQUIRE(res.to_array() == std::array<float, 4>{ 1, 2, 7, 8 });

        res = simdf4::shuffle<2, 1, 0, 3>(a, b);
        REQUIRE(res.to_array() == std::array<float, 4>{ 3, 2, 5, 8 });
    }
}
