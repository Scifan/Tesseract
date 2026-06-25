#include <catch2/catch_test_macros.hpp>

#include "tesseract/core/Shape.hpp"
#include "tesseract/utils/Logging.hpp"

using tesseract::Shape;

TEST_CASE("Shape default and basic accessors", "[shape]") {
  Shape s;
  REQUIRE(s.rank() == 0);
  REQUIRE(s.empty());
  REQUIRE(s.numel() == 1);  // 0-dim scalar convention

  Shape r{2, 3, 4};
  REQUIRE(r.rank() == 3);
  REQUIRE_FALSE(r.empty());
  REQUIRE(r[0] == 2);
  REQUIRE(r[1] == 3);
  REQUIRE(r[2] == 4);
  REQUIRE(r.numel() == 24);
  REQUIRE(r.front() == 2);
  REQUIRE(r.back() == 4);
}

TEST_CASE("Shape equality and to_string", "[shape]") {
  Shape a{2, 3};
  Shape b{2, 3};
  Shape c{3, 2};
  REQUIRE(a == b);
  REQUIRE(a != c);

  REQUIRE(Shape{}.to_string() == "[]");
  REQUIRE(Shape{5}.to_string() == "[5]");
  REQUIRE(Shape{2, 3, 4}.to_string() == "[2, 3, 4]");
}

TEST_CASE("Shape contiguous_strides", "[shape]") {
  REQUIRE(Shape{}.contiguous_strides() == Shape{});
  REQUIRE(Shape{5}.contiguous_strides() == Shape{1});

  // For [2, 3, 4] the strides should be [12, 4, 1]
  Shape s{2, 3, 4};
  REQUIRE(s.contiguous_strides() == Shape{12, 4, 1});

  Shape m{4, 1, 7};
  REQUIRE(m.contiguous_strides() == Shape{7, 7, 1});
}

TEST_CASE("Shape push_back and resize", "[shape]") {
  Shape s;
  s.push_back(3);
  s.push_back(5);
  REQUIRE(s == Shape{3, 5});

  s.resize(4);
  REQUIRE(s.rank() == 4);
  REQUIRE(s[0] == 3);
  REQUIRE(s[1] == 5);

  s.resize(1);
  REQUIRE(s.rank() == 1);
  REQUIRE(s.front() == 3);

  s.pop_back();
  REQUIRE(s.empty());
}

TEST_CASE("Shape rejects over-rank inputs", "[shape]") {
  REQUIRE_THROWS_AS((Shape{1, 2, 3, 4, 5, 6, 7, 8, 9}), tesseract::Error);
}
