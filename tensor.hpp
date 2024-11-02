#pragma once

#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <vector>

namespace evol {

///
///
///
/// DEVICE
///
///
///
namespace device {
#define DEF_DEVICE(name, value)                                                \
  template <int64_t N = 0> struct name {                                       \
    static constexpr torch::DeviceType DEVICE = value;                         \
    static constexpr int64_t INDEX = N;                                        \
  };

DEF_DEVICE(CPU, torch::kCPU);
DEF_DEVICE(CUDA, torch::kCUDA);
DEF_DEVICE(HIP, torch::kHIP);
DEF_DEVICE(FPGA, torch::kFPGA);
DEF_DEVICE(MAIA, torch::kMAIA);
DEF_DEVICE(XLA, torch::kXLA);
DEF_DEVICE(MPS, torch::kMPS);
DEF_DEVICE(Meta, torch::kMeta);
DEF_DEVICE(Vulkan, torch::kVulkan);
DEF_DEVICE(Metal, torch::kMetal);
DEF_DEVICE(XPU, torch::kXPU);
DEF_DEVICE(HPU, torch::kHPU);
DEF_DEVICE(VE, torch::kVE);
DEF_DEVICE(Lazy, torch::kLazy);
DEF_DEVICE(IPU, torch::kIPU);
DEF_DEVICE(MTIA, torch::kMTIA);
DEF_DEVICE(PrivateUse1, torch::kPrivateUse1);
DEF_DEVICE(OPENGL, torch::DeviceType::OPENGL);
DEF_DEVICE(OPENCL, torch::DeviceType::OPENCL);
DEF_DEVICE(IDEEP, torch::DeviceType::IDEEP);
DEF_DEVICE(MKLDNN, torch::DeviceType::MKLDNN);

} // namespace device
template <int64_t N = 0> using CPU = device::CPU<N>;
template <int64_t N = 0> using CUDA = device::CUDA<N>;
template <int64_t N = 0> using HIP = device::HIP<N>;
template <int64_t N = 0> using FPGA = device::FPGA<N>;
template <int64_t N = 0> using MAIA = device::MAIA<N>;
template <int64_t N = 0> using XLA = device::XLA<N>;
template <int64_t N = 0> using MPS = device::MPS<N>;
template <int64_t N = 0> using Meta = device::Meta<N>;
template <int64_t N = 0> using Vulkan = device::Vulkan<N>;
template <int64_t N = 0> using Metal = device::Metal<N>;
template <int64_t N = 0> using XPU = device::XPU<N>;
template <int64_t N = 0> using HPU = device::HPU<N>;
template <int64_t N = 0> using VE = device::VE<N>;
template <int64_t N = 0> using Lazy = device::Lazy<N>;
template <int64_t N = 0> using IPU = device::IPU<N>;
template <int64_t N = 0> using MTIA = device::MTIA<N>;
template <int64_t N = 0> using PrivateUse1 = device::PrivateUse1<N>;
template <int64_t N = 0> using OPENGL = device::OPENGL<N>;
template <int64_t N = 0> using OPENCL = device::OPENCL<N>;
template <int64_t N = 0> using IDEEP = device::IDEEP<N>;
template <int64_t N = 0> using MKLDNN = device::MKLDNN<N>;

///
///
///
/// DTYPE
///
///
///

namespace dtype {

#define DEF_DTYPE(name, value)                                                 \
  struct name {                                                                \
    static constexpr torch::Dtype DTYPE = torch::value;                        \
  };

DEF_DTYPE(i8, kInt8);
DEF_DTYPE(i16, kInt16);
DEF_DTYPE(i32, kInt32);
DEF_DTYPE(i64, kInt64);

DEF_DTYPE(u8, kUInt8);
DEF_DTYPE(u16, kUInt16);
DEF_DTYPE(u32, kUInt32);
DEF_DTYPE(u64, kUInt64);

DEF_DTYPE(f16, kFloat16);
DEF_DTYPE(f32, kFloat32);
DEF_DTYPE(f64, kFloat64);

DEF_DTYPE(byte, kByte);

DEF_DTYPE(c16, kComplexHalf);
DEF_DTYPE(c32, kComplexFloat);
DEF_DTYPE(c64, kComplexDouble);

DEF_DTYPE(q8, kQInt8);
DEF_DTYPE(q32, kQInt32);
DEF_DTYPE(q2x4, kQUInt2x4);
DEF_DTYPE(q4x2, kQUInt4x2);

DEF_DTYPE(bits8, kBits8);
DEF_DTYPE(bits16, kBits16);
DEF_DTYPE(bits2x4, kBits2x4);
DEF_DTYPE(bits4x2, kBits4x2);
DEF_DTYPE(bits1x8, kBits1x8);

template <typename T> struct DType;

#define DEF_DTYPE_IMPL(name)                                                   \
  template <> struct DType<name> {                                             \
    static constexpr torch::Dtype DTYPE = name::DTYPE;                         \
  };

#define DEF_DTYPE_IMPL_BASE(name, val)                                         \
  template <> struct DType<name> {                                             \
    static constexpr torch::Dtype DTYPE = torch::val;                          \
  };

DEF_DTYPE_IMPL(i8);
DEF_DTYPE_IMPL(i16);
DEF_DTYPE_IMPL(i32);
DEF_DTYPE_IMPL(i64);

DEF_DTYPE_IMPL(u8);
DEF_DTYPE_IMPL(u16);
DEF_DTYPE_IMPL(u32);
DEF_DTYPE_IMPL(u64);

DEF_DTYPE_IMPL(f16);
DEF_DTYPE_IMPL(f32);
DEF_DTYPE_IMPL(f64);

DEF_DTYPE_IMPL(byte);

DEF_DTYPE_IMPL(c16);
DEF_DTYPE_IMPL(c32);
DEF_DTYPE_IMPL(c64);

DEF_DTYPE_IMPL(q8);
DEF_DTYPE_IMPL(q32);
DEF_DTYPE_IMPL(q2x4);
DEF_DTYPE_IMPL(q4x2);

DEF_DTYPE_IMPL(bits8);
DEF_DTYPE_IMPL(bits16);
DEF_DTYPE_IMPL(bits2x4);
DEF_DTYPE_IMPL(bits4x2);
DEF_DTYPE_IMPL(bits1x8);

DEF_DTYPE_IMPL_BASE(bool, kBool);
DEF_DTYPE_IMPL_BASE(char, kChar);

DEF_DTYPE_IMPL_BASE(int, kInt);
DEF_DTYPE_IMPL_BASE(int8_t, kInt8);
DEF_DTYPE_IMPL_BASE(int16_t, kInt16);
DEF_DTYPE_IMPL_BASE(int64_t, kInt64);

DEF_DTYPE_IMPL_BASE(unsigned int, kUInt32);
DEF_DTYPE_IMPL_BASE(uint8_t, kUInt8);
DEF_DTYPE_IMPL_BASE(uint16_t, kUInt16);
DEF_DTYPE_IMPL_BASE(uint64_t, kUInt64);

DEF_DTYPE_IMPL_BASE(float, kFloat);
DEF_DTYPE_IMPL_BASE(double, kDouble);
DEF_DTYPE_IMPL_BASE(long double, kLong);

}; // namespace dtype

using i8 = dtype::i8;
using i16 = dtype::i16;
using i32 = dtype::i32;
using i64 = dtype::i64;

using u8 = dtype::u8;
using u16 = dtype::u16;
using u32 = dtype::u32;
using u64 = dtype::u64;

using f16 = dtype::f16;
using f32 = dtype::f32;
using f64 = dtype::f64;

using byte = dtype::byte;

using c16 = dtype::c16;
using c32 = dtype::c32;
using c64 = dtype::c64;

using q8 = dtype::q8;
using q32 = dtype::q32;
using q2x4 = dtype::q2x4;
using q4x2 = dtype::q4x2;

using bits8 = dtype::bits8;
using bits16 = dtype::bits16;
using bits2x4 = dtype::bits2x4;
using bits4x2 = dtype::bits4x2;
using bits1x8 = dtype::bits1x8;

///
/// TensorPair
///
template <int64_t N, int64_t D> struct TensorPair {
  static constexpr int64_t K = N;
  static constexpr int64_t DIM = D;
};
template <int64_t K, int64_t Dim> using Tp = TensorPair<K, Dim>;

///
/// TensorRange
///
template <int64_t S, int64_t E, int64_t D> struct TensorRange {
  static constexpr int64_t START = S;
  static constexpr int64_t END = E;
  static constexpr int64_t DIM = D;
};
template <int64_t Start, int64_t End, int64_t Dim>
using Tr = TensorRange<Start, End, Dim>;

///
///
///
/// SHAPE
///
///
///
template <int64_t... Dims> struct Shape;

template <> struct Shape<> {
  static constexpr int64_t DIMS = 0;
  static constexpr int64_t NELEMS = 0;
  static constexpr std::array<int64_t, DIMS> SHAPE_DIMS = {};
  template <int64_t... NewDims> using prepend = Shape<NewDims...>;
  template <int64_t... NewDims> using append = Shape<NewDims...>;
};

template <int64_t First> struct Shape<First> {
  static constexpr int64_t DIMS = 1;
  static constexpr int64_t NELEMS = First;
  static constexpr int64_t LAST = First;
  static constexpr std::array<int64_t, DIMS> SHAPE_DIMS = {First};
  template <int64_t... NewDims> using prepend = Shape<NewDims..., First>;
  template <int64_t... NewDims> using append = Shape<First, NewDims...>;
};

template <int64_t First, int64_t Second> struct Shape<First, Second> {
  static constexpr int64_t DIMS = 2;
  static constexpr int64_t NELEMS = First * Second;
  static constexpr int64_t LAST = Second;
  static constexpr int64_t PENULTIMATE = First;
  static constexpr std::array<int64_t, DIMS> SHAPE_DIMS = {First, Second};
  using AllButLastTwo = Shape<>;
  template <int64_t... NewDims>
  using prepend = Shape<NewDims..., First, Second>;
  template <int64_t... NewDims> using append = Shape<First, Second, NewDims...>;
};

template <int64_t First, int64_t Second, int64_t... Rest>
struct Shape<First, Second, Rest...> {
  static constexpr int64_t DIMS = 2 + sizeof...(Rest);
  static constexpr int64_t NELEMS = First * Second * Shape<Rest...>::NELEMS;
  static constexpr int64_t LAST = Shape<Rest...>::LAST;
  static constexpr int64_t PENULTIMATE = Shape<Second, Rest...>::PENULTIMATE;
  static constexpr std::array<int64_t, DIMS> SHAPE_DIMS = {First, Second,
                                                           Rest...};
  using AllButLastTwo =
      typename Shape<Second, Rest...>::AllButLastTwo::template prepend<First>;
  template <int64_t... NewDims>
  using prepend = Shape<NewDims..., First, Second, Rest...>;
  template <int64_t... NewDims>
  using append = Shape<First, Second, Rest..., NewDims...>;
};

///
/// IndicesPair
///
template <int64_t D, int64_t... I> struct IndicesPair {
  static constexpr int64_t DIM = D;
  using Indices = Shape<I...>;
};
template <int64_t Dim, int64_t... Indices>
using Ip = IndicesPair<Dim, Indices...>;

///
///
///
/// UTILS
///
///
///

///
/// IsSamePack
///
template <typename P1, typename P2> struct IsSameAsPack;

template <int64_t First1, int64_t First2>
struct IsSameAsPack<Shape<First1>, Shape<First2>>
    : std::__bool_constant<First1 == First2> {};

template <int64_t First1, int64_t... Rest1, int64_t First2, int64_t... Rest2>
struct IsSameAsPack<Shape<First1, Rest1...>, Shape<First2, Rest2...>>
    : std::__bool_constant<
          (First1 == First2) &&
          IsSameAsPack<Shape<Rest1...>, Shape<Rest2...>>::value> {};

///
/// IsCatCompatible
///
template <int Idx, typename P1, typename P2> struct IsCatCompatible;

template <int64_t First1, int64_t First2>
struct IsCatCompatible<0, Shape<First1>, Shape<First2>> : std::true_type {};

template <int64_t First1, int64_t... Rest1, int64_t First2, int64_t... Rest2>
struct IsCatCompatible<0, Shape<First1, Rest1...>, Shape<First2, Rest2...>>
    : std::__bool_constant<
          IsSameAsPack<Shape<Rest1...>, Shape<Rest2...>>::value> {};

template <int Idx, int64_t First1, int64_t... Rest1, int64_t First2,
          int64_t... Rest2>
struct IsCatCompatible<Idx, Shape<First1, Rest1...>, Shape<First2, Rest2...>>
    : std::__bool_constant<
          First1 == First2 &&
          IsCatCompatible<Idx - 1, Shape<Rest1...>, Shape<Rest2...>>::value> {};

///
/// SetDimAtIdx
///
template <int64_t Idx, int64_t Dim, typename T> struct SetDimAtIdx;
template <int64_t Dim, int64_t First, int64_t... Dims>
struct SetDimAtIdx<0, Dim, Shape<First, Dims...>> {
  using type = Shape<Dim, Dims...>;
};
template <int64_t Idx, int64_t Dim, int64_t First, int64_t... Rest>
struct SetDimAtIdx<Idx, Dim, Shape<First, Rest...>> {
  static_assert(Idx <= sizeof...(Rest), "\nError: Index out of bounds.\n");
  using type =
      typename SetDimAtIdx<Idx - 1, Dim,
                           Shape<Rest...>>::type::template prepend<First>;
};

///
/// IsNotInPack
///
template <int64_t N, typename T> struct IsNotInPack;

template <int64_t N, int64_t First>
struct IsNotInPack<N, Shape<First>> : std::__bool_constant<N != First> {};

template <int64_t N, int64_t First, int64_t... Rest>
struct IsNotInPack<N, Shape<First, Rest...>>
    : std::__bool_constant<N != First &&
                           IsNotInPack<N, Shape<Rest...>>::value> {};

///
/// AreAllUnique
///
template <typename T> struct AreAllUnique;

template <> struct AreAllUnique<Shape<>> : std::true_type {};

template <int64_t First> struct AreAllUnique<Shape<First>> : std::true_type {};

template <int64_t First, int64_t... Rest>
struct AreAllUnique<Shape<First, Rest...>>
    : std::__bool_constant<IsNotInPack<First, Shape<Rest...>>::value &&
                           AreAllUnique<Shape<Rest...>>::value> {};

///
/// AreAllLessThanN
///
template <int64_t N, typename T> struct AreAllLessThanN;

template <int64_t N> struct AreAllLessThanN<N, Shape<>> : std::true_type {};

template <int64_t N, int64_t First>
    struct AreAllLessThanN<N, Shape<First>> : std::__bool_constant < First<N> {
};

template <int64_t N, int64_t First, int64_t... Rest>
    struct AreAllLessThanN<N, Shape<First, Rest...>>
    : std::__bool_constant <
      First<N && AreAllLessThanN<N, Shape<Rest...>>::value> {};

///
/// AreAllSameAs
///
template <typename As, typename... T> struct AreAllSameAs;

template <typename As, typename First>
struct AreAllSameAs<As, First>
    : std::__bool_constant<std::is_same<As, First>::value> {};

template <typename As, typename First, typename... Rest>
struct AreAllSameAs<As, First, Rest...>
    : std::__bool_constant<std::is_same<As, First>::value &&
                           AreAllSameAs<As, Rest...>::value> {};

///
/// AreAllInIncreasingOrder
///
template <int64_t... Dims> struct AreAllInIncreasingOrder;

template <int64_t First>
struct AreAllInIncreasingOrder<First> : std::true_type {};

template <int64_t First, int64_t Second>
    struct AreAllInIncreasingOrder<First, Second> : std::__bool_constant <
                                                    First<Second> {};

template <int64_t First, int64_t Second, int64_t... Rest>
    struct AreAllInIncreasingOrder<First, Second, Rest...>
    : std::__bool_constant <
      First<Second && AreAllInIncreasingOrder<Second, Rest...>::value> {};

///
/// AreTensorRangesAllInDecreasingOrder
///
template <typename... TensorRanges> struct AreTensorRangesAllInDecreasingOrder;

template <typename First>
struct AreTensorRangesAllInDecreasingOrder<First> : std::true_type {};

template <typename First, typename Second>
    struct AreTensorRangesAllInDecreasingOrder<First, Second>
    : std::__bool_constant < Second::DIM<First::DIM> {};

template <typename First, typename Second, typename... Rest>
    struct AreTensorRangesAllInDecreasingOrder<First, Second, Rest...>
    : std::__bool_constant <
      Second::DIM<First::DIM &&
                  AreTensorRangesAllInDecreasingOrder<Second, Rest...>::value> {
};

///
/// AreTensorRangesAllInIncreasingOrder
///
template <typename... TensorRanges> struct AreTensorRangesAllInIncreasingOrder;

template <typename First>
struct AreTensorRangesAllInIncreasingOrder<First> : std::true_type {};

template <typename First, typename Second>
    struct AreTensorRangesAllInIncreasingOrder<First, Second>
    : std::__bool_constant < First::DIM<Second::DIM> {};

template <typename First, typename Second, typename... Rest>
    struct AreTensorRangesAllInIncreasingOrder<First, Second, Rest...>
    : std::__bool_constant <
      First::DIM<Second::DIM &&
                 AreTensorRangesAllInDecreasingOrder<Second, Rest...>::value> {
};

///
/// AreTensorPairDimsUnique
///
template <typename... RepeatPairs> struct AreTensorPairDimsUnique;

template <typename First>
struct AreTensorPairDimsUnique<First> : std::true_type {};

template <typename First, typename Second>
struct AreTensorPairDimsUnique<First, Second>
    : std::__bool_constant<First::DIM != Second::DIM> {};

template <typename First, typename Second, typename... Rest>
struct AreTensorPairDimsUnique<First, Second, Rest...>
    : std::__bool_constant<First::DIM != Second::DIM &&
                           AreTensorPairDimsUnique<Second, Rest...>::value> {};

///
///
///
/// SHAPE GENERATORS
///
///
///

///
/// CatArrayShape
///
template <typename S, int64_t Idx, int64_t Mul> struct CatArrayShape;
template <int64_t Mul> struct CatArrayShape<Shape<>, 0, Mul> {
  using type = Shape<>;
};
template <int64_t First, int64_t Mul>
struct CatArrayShape<Shape<First>, 0, Mul> {
  using type = Shape<First * Mul>;
};
template <int64_t First, int64_t... Rest, int64_t Mul>
struct CatArrayShape<Shape<First, Rest...>, 0, Mul> {
  using type = Shape<First * Mul, Rest...>;
};
template <int64_t First, int64_t... Rest, int64_t Idx, int64_t Mul>
struct CatArrayShape<Shape<First, Rest...>, Idx, Mul> {
  static_assert(Idx <= sizeof...(Rest), "Error: Index out of bounds.");
  using type = typename CatArrayShape<Shape<Rest...>, Idx - 1,
                                      Mul>::type::template prepend<First>;
};

//
// CatShape
//

template <int64_t Idx, typename ShapeType, typename... Shapes> struct CatShape;

template <int64_t Idx, typename ShapeType> struct CatShape<Idx, ShapeType> {
  static_assert(Idx < ShapeType::DIMS, "Index out of bounds.");
};

template <int64_t Idx, typename ShapeType, typename FirstShape>
struct CatShape<Idx, ShapeType, FirstShape> {
  static_assert(Idx < ShapeType::DIMS, "Index out of bounds.");
  static_assert(FirstShape::DIMS == ShapeType::DIMS,
                "Shapes must have the same number of dimensions.");
  static_assert(
      IsCatCompatible<Idx, ShapeType, FirstShape>::value,
      "Shapes must have the same dimensions except for the one at Idx");
  static constexpr int64_t new_dim =
      ShapeType::SHAPE_DIMS[Idx] + FirstShape::SHAPE_DIMS[Idx];
  using type = typename SetDimAtIdx<Idx, new_dim, ShapeType>::type;
};

template <int64_t Idx, typename ShapeType, typename FirstShape,
          typename... RestShapes>
struct CatShape<Idx, ShapeType, FirstShape, RestShapes...> {
  static_assert(Idx < ShapeType::DIMS, "Index out of bounds.");
  static_assert(FirstShape::DIMS == ShapeType::DIMS,
                "Shapes must have the same number of dimensions.");
  static_assert(
      IsCatCompatible<Idx, ShapeType, FirstShape>::value,
      "Shapes must have the same dimensions except for the one at Idx");
  static constexpr int64_t new_dim =
      FirstShape::SHAPE_DIMS[Idx] +
      CatShape<Idx, ShapeType, RestShapes...>::new_dim;
  using type = typename SetDimAtIdx<Idx, new_dim, ShapeType>::type;
};

///
/// Matmul
///

template <typename S1, typename S2> struct MatmulShape {
  static_assert(S1::DIMS == S2::DIMS,
                "\nShape mismatch: To perform matrix multiplication both "
                "shapes must have the same number of elements.\n");
  static_assert(S1::DIMS > 1,
                "\nShape mismatch: To perform matrix multiplication S1 "
                "must have at least 2 dimensions\n");
  static_assert(S2::DIMS > 1,
                "\nShape mismatch: To perform matrix multiplication S2 "
                "must have at least 2 dimensions\n");
  static_assert(
      std::is_same<typename S1::AllButLastTwo, typename S2::AllButLastTwo>(),
      "\nShape mismatch: To perform matrix multiplication"
      " all dimensions except the last two must be the same.\n");
  static_assert(
      S1::LAST == S2::PENULTIMATE,
      "\nShape mismatch: To perform matrix multiplication"
      "the last dimension of S1 must match the penultimate dimension of S2");
  using type = typename S1::AllButLastTwo::template append<
      S1::PENULTIMATE>::template append<S2::LAST>;
};

///
/// TransposeShape
///

template <typename S1, typename S2> struct TransposeShape;
template <int64_t First, typename S2> struct TransposeShape<Shape<First>, S2> {
  static_assert(
      First < S2::DIMS,
      "\nError: The order contains too many items, must be == Shape::DIMS.\n");
  using type = Shape<S2::SHAPE_DIMS[First]>;
};
template <int64_t First, int64_t... Rest, typename S2>
struct TransposeShape<Shape<First, Rest...>, S2> {
  static_assert(
      First < S2::DIMS,
      "\nError: The order contains too many items, must be == Shape::DIMS.\n");
  using type = typename TransposeShape<
      Shape<Rest...>, S2>::type::template prepend<S2::SHAPE_DIMS[First]>;
};

///
/// Unsqueeze
///
template <int64_t Dim, typename S> struct UnsqueezeShape;

template <> struct UnsqueezeShape<0, Shape<>> {
  using type = Shape<1>;
};

template <int64_t First, int64_t... Dims>
struct UnsqueezeShape<0, Shape<First, Dims...>> {
  using type = Shape<1, First, Dims...>;
};

template <int64_t Dim, int64_t First, int64_t... Rest>
struct UnsqueezeShape<Dim, Shape<First, Rest...>> {
  using type =
      typename UnsqueezeShape<Dim - 1,
                              Shape<Rest...>>::type::template prepend<First>;
};

///
/// VariadicUnsqueeze
///
template <int64_t Curr, typename D, typename S> struct VariadicUnsqueezeShape;

template <int64_t Curr, int64_t First, typename S>
struct VariadicUnsqueezeShape<Curr, Shape<First>, S> {
  using type = typename UnsqueezeShape<First + Curr, S>::type;
};

template <int64_t Curr, int64_t First, int64_t... Rest, typename S>
struct VariadicUnsqueezeShape<Curr, Shape<First, Rest...>, S> {
  static_assert(AreAllInIncreasingOrder<First, Rest...>::value,
                "\nThe provided dimensions must be in increasing order.\n");
  using type = typename VariadicUnsqueezeShape<
      Curr + 1, Shape<Rest...>,
      typename UnsqueezeShape<First + Curr, S>::type>::type;
};

///
/// UnsqueezeDimShape
///
template <int64_t Dim, typename S> struct UnsqueezeDimShape;

template <typename S> struct UnsqueezeDimShape<0, S> {
  using type = typename S::template prepend<1>;
};

template <int64_t Dim, typename S> struct UnsqueezeDimShape {
  using type =
      typename UnsqueezeDimShape<Dim - 1,
                                 typename S::template prepend<1>>::type;
};

///
/// Squeeze
///
template <typename T> struct SqueezeShape;

template <> struct SqueezeShape<Shape<>> {
  using type = Shape<>;
};

template <int64_t First> struct SqueezeShape<Shape<First>> {
  using type =
      typename std::conditional<First == 1, Shape<>, Shape<First>>::type;
};

template <int64_t First, int64_t... Rest>
struct SqueezeShape<Shape<First, Rest...>> {
  using type = typename std::conditional<
      First == 1, typename SqueezeShape<Shape<Rest...>>::type,
      typename SqueezeShape<Shape<Rest...>>::type::template prepend<First>>::
      type;
};

///
/// SqueezeShapeAtIdx
///
template <int64_t Idx, typename S> struct SqueezeShapeAtIdx;

template <int64_t First, int64_t... Rest>
struct SqueezeShapeAtIdx<0, Shape<First, Rest...>> {
  using type = typename std::conditional<First == 1, Shape<Rest...>,
                                         Shape<First, Rest...>>::type;
};

template <int64_t Idx, int64_t First, int64_t... Rest>
struct SqueezeShapeAtIdx<Idx, Shape<First, Rest...>> {
  static_assert(Idx < Shape<First, Rest...>::DIMS,
                "\nIdx must be < Shape::DIMS.\n");
  using type =
      typename SqueezeShapeAtIdx<Idx - 1,
                                 Shape<Rest...>>::type::template prepend<First>;
};

///
/// VariadicSqueeze
///
template <int64_t Curr, typename T, typename S> struct VariadicSqueezeShape;

template <int64_t Curr, int64_t First, typename S>
struct VariadicSqueezeShape<Curr, Shape<First>, S> {
  using type = typename SqueezeShapeAtIdx<First - Curr, S>::type;
};

template <int64_t Curr, int64_t First, int64_t... Rest, typename S>
struct VariadicSqueezeShape<Curr, Shape<First, Rest...>, S> {
  static_assert(AreAllInIncreasingOrder<First, Rest...>::value,
                "\nThe provided dimensions must be in increasing order.\n");
  using type = typename std::conditional<
      S::SHAPE_DIMS[First - Curr] == 1,
      typename VariadicSqueezeShape<
          Curr + 1, Shape<Rest...>,
          typename SqueezeShapeAtIdx<First - Curr, S>::type>::type,
      typename VariadicSqueezeShape<Curr, Shape<Rest...>,
                                    S>::type::template prepend<First>>::type;
};

///
/// SwapDimsShape
///
template <int64_t I, int64_t J, typename S> struct SwapDimsShape {
  static_assert(I < S::DIMS, "\nI must be < Shape::DIMS.\n");
  static_assert(J < S::DIMS, "\nJ must be < Shape::DIMS.\n");
  using type = typename SetDimAtIdx<
      J, S::SHAPE_DIMS[I],
      typename SetDimAtIdx<I, S::SHAPE_DIMS[J], S>::type>::type;
};

///
/// RepeatShape
///
template <typename S, typename... RepeatPairs> struct RepeatShape;

template <typename S, typename First> struct RepeatShape<S, First> {
  static_assert(First::Dim < S::DIMS, "\nError: Index out of bounds.\n");
  using type =
      typename SetDimAtIdx<First::Dim, S::SHAPE_DIMS[First::Dim] * First::K,
                           S>::type;
};

template <typename S, typename First, typename... Rest>
struct RepeatShape<S, First, Rest...> {
  static_assert(
      AreTensorPairDimsUnique<First, Rest...>::value &&
          sizeof...(Rest) <= S::DIMS,
      "\n.You don't need to repeat the same dimension more than once\n");
  static_assert(First::Dim < S::DIMS, "\nError: Index out of bounds.\n");
  using type = typename RepeatShape<
      typename SetDimAtIdx<First::Dim, S::SHAPE_DIMS[First::Dim] * First::K,
                           S>::type,
      Rest...>::type;
};

///
/// TopShape
///
template <typename S, typename... TopPairs> struct TopShape;

template <typename S, typename First> struct TopShape<S, First> {
  static_assert(First::Dim < S::DIMS, "\nError: Index out of bounds.\n");
  static_assert(
      First::K <= S::SHAPE_DIMS[First::Dim],
      "\nError: K is > than the number of elements in the given dimension.\n");
  using type = typename SetDimAtIdx<First::Dim, First::K, S>::type;
};

template <typename S, typename First, typename... Rest>
struct TopShape<S, First, Rest...> {
  static_assert(AreTensorPairDimsUnique<First, Rest...>::value &&
                    sizeof...(Rest) <= S::DIMS,
                "\n.You don't need to get the top elements of the same "
                "dimension more than once\n");
  static_assert(First::Dim < S::DIMS, "\nError: Index out of bounds.\n");
  static_assert(
      First::K <= S::SHAPE_DIMS[First::Dim],
      "\nError: K is > than the number of elements in the given dimension.\n");
  using type =
      typename TopShape<typename SetDimAtIdx<First::Dim, First::K, S>::type,
                        Rest...>::type;
};

///
/// NarrowShape
///
template <typename S, typename... TensorRanges> struct NarrowShape;

template <typename S, typename First> struct NarrowShape<S, First> {
  static_assert(First::Dim < S::DIMS, "\nError: Index out of bounds.\n");
  static_assert(First::Start <= First::End,
                "\nSlice::Start must be <= Slice::End.\n");
  static_assert(First::End <= S::SHAPE_DIMS[First::Dim],
                "\nSlice index out of bounds, End > Dim.\n");
  static_assert(First::End - First::Start <= S::SHAPE_DIMS[First::Dim],
                "\nError: The range size is > than the number of elements in "
                "the given dimension.\n");
  using type =
      typename SetDimAtIdx<First::Dim, First::End - First::Start, S>::type;
};

template <typename S, typename First, typename... Rest>
struct NarrowShape<S, First, Rest...> {
  static_assert(
      AreTensorPairDimsUnique<First, Rest...>::value &&
          sizeof...(Rest) <= S::DIMS,
      "\n.You don't need to slice the same dimension more than once\n");
  static_assert(First::Dim < S::DIMS, "\nError: Index out of bounds.\n");
  static_assert(First::Start <= First::End,
                "\nSlice::Start must be <= Slice::End.\n");
  static_assert(First::End <= S::SHAPE_DIMS[First::Dim],
                "\nSlice index out of bounds, End > Dim.\n");
  static_assert(First::End - First::Start <= S::SHAPE_DIMS[First::Dim],
                "\nError: The range size is > than the number of elements in "
                "the given dimension.\n");
  using type = typename NarrowShape<
      typename SetDimAtIdx<First::Dim, First::End - First::Start, S>::type,
      Rest...>::type;
};

///
/// SliceShape
///
template <int64_t Curr, typename S, typename... TensorRanges> struct SliceShape;

template <typename S, typename First, typename... Rest>
struct SliceShape<0, S, First, Rest...> {
  static_assert(First::Dim < S::DIMS, "\nError: Index out of bounds.\n");
  static_assert(First::Start <= First::End,
                "\nSlice::Start must be <= Slice::End.\n");
  static_assert(First::End <= S::SHAPE_DIMS[First::Dim],
                "\nSlice index out of bounds, End > Dim.\n");
  static_assert(First::End - First::Start <= S::SHAPE_DIMS[First::Dim],
                "\nError: The range size is > than the number of elements in "
                "the given dimension.\n");
  using type = typename std::conditional<
      First::Dim == 0, Shape<First::End - First::Start>, Shape<>>::type;
};

template <int64_t Curr, typename S, typename First>
struct SliceShape<Curr, S, First> {
  static_assert(First::Dim < S::DIMS, "\nError: Index out of bounds.\n");
  static_assert(First::Start <= First::End,
                "\nSlice::Start must be <= Slice::End.\n");
  static_assert(First::End <= S::SHAPE_DIMS[First::Dim],
                "\nSlice index out of bounds, End > Dim.\n");
  static_assert(First::End - First::Start <= S::SHAPE_DIMS[First::Dim],
                "\nError: The range size is > than the number of elements in "
                "the given dimension.\n");
  using type =
      typename std::conditional <
      S::DIMS<Curr, Shape<>,
              typename std::conditional<
                  Curr == First::Dim, Shape<First::End - First::Start>,
                  typename SliceShape<Curr - 1, S, First>::type>::type>::type;
};

template <int64_t Curr, typename S, typename First, typename... Rest>
struct SliceShape<Curr, S, First, Rest...> {
  static_assert(AreTensorRangesAllInDecreasingOrder<First, Rest...>::value,
                "\nThe provided dimensions must be in decreasing order.\n");
  static_assert(
      AreTensorPairDimsUnique<First, Rest...>::value &&
          sizeof...(Rest) <= S::DIMS,
      "\n.You don't need to slice the same dimension more than once\n");
  static_assert(First::Dim < S::DIMS, "\nError: Index out of bounds.\n");
  static_assert(First::Start <= First::End,
                "\nSlice::Start must be <= Slice::End.\n");
  static_assert(First::End <= S::SHAPE_DIMS[First::Dim],
                "\nSlice index out of bounds, End > Dim.\n");
  static_assert(First::End - First::Start <= S::SHAPE_DIMS[First::Dim],
                "\nError: The range size is > than the number of elements in "
                "the given dimension.\n");
  using type = typename std::conditional<
      Curr == First::Dim,
      typename SliceShape<Curr - 1, S, Rest...>::type::template append<
          First::End - First::Start>,
      typename SliceShape<Curr - 1, S, First, Rest...>::type>::type;
};

///
/// SelectOnlyShape
///
template <typename S, typename... SelectPairs> struct SelectOnlyShape;

template <typename S, typename First> struct SelectOnlyShape<S, First> {
  static_assert(S::DIMS > First::DIM, "\n.Error: Index out of bounds\n");
  static_assert(
      AreAllLessThanN<S::SHAPE_DIMS[First::DIM], typename First::Indices>(),
      "\nAll indices must be less than the relative dim.\n");
  static_assert(AreAllUnique<typename First::Indices>(),
                "\n.You can't have duplicate indices.\n");
  using type = Shape<First::Indices::DIMS>;
};

template <typename S, typename First, typename... Rest>
struct SelectOnlyShape<S, First, Rest...> {
  static_assert(S::DIMS > First::DIM, "\n.Error: Index out of bounds\n");
  static_assert(
      AreAllLessThanN<S::SHAPE_DIMS[First::DIM], typename First::Indices>(),
      "\nAll indices must be less than the relative dim.\n");
  static_assert(AreAllUnique<typename First::Indices>::value,
                "\n.You can't have duplicate indices.\n");
  static_assert(AreTensorPairDimsUnique<First, Rest...>::value,
                "\nYou can't have duplicate indices for the same dimension.\n");
  static_assert(AreTensorRangesAllInIncreasingOrder<First, Rest...>::value,
                "\nThe SelectPairs Dims must be in increasing order.\n");
  using type = typename SelectOnlyShape<S, Rest...>::type::template prepend<
      First::Indices::DIMS>;
};

///
/// SelectShape
///
template <typename S, typename... SelectPairs> struct SelectShape;

template <typename S, typename First> struct SelectShape<S, First> {
  static_assert(S::DIMS > First::DIM, "\n.Error: Index out of bounds\n");
  static_assert(
      AreAllLessThanN<S::SHAPE_DIMS[First::DIM], typename First::Indices>(),
      "\nAll indices must be less than the relative dim.\n");
  static_assert(AreAllUnique<typename First::Indices>(),
                "\n.You can't have duplicate indices.\n");
  using type = typename SetDimAtIdx<First::DIM, First::Indices::DIMS, S>::type;
};

template <typename S, typename First, typename... Rest>
struct SelectShape<S, First, Rest...> {
  static_assert(S::DIMS > First::DIM, "\n.Error: Index out of bounds\n");
  static_assert(
      AreAllLessThanN<S::SHAPE_DIMS[First::DIM], typename First::Indices>(),
      "\nAll indices must be less than the relative dim.\n");
  static_assert(AreAllUnique<typename First::Indices>::value,
                "\n.You can't have duplicate indices.\n");
  static_assert(AreTensorPairDimsUnique<First, Rest...>::value,
                "\nYou can't have duplicate indices for the same dimension.\n");
  static_assert(AreTensorRangesAllInIncreasingOrder<First, Rest...>::value,
                "\nThe SelectPairs Dims must be in increasing order.\n");
  using type = typename SelectShape<
      typename SetDimAtIdx<First::DIM, First::Indices::DIMS, S>::type,
      Rest...>::type;
};

///
/// PadShape
///
template <typename S, int64_t Height, int64_t Width> struct PadShape {
  static_assert(S::DIMS > 1,
                "\nPad only works on tensors with at least 2 dimensions.\n");
  using type = typename SetDimAtIdx<
      S::DIMS - 1, S::SHAPE_DIMS[S::DIMS - 1] + Width,
      typename SetDimAtIdx<S::DIMS - 2, S::SHAPE_DIMS[S::DIMS - 2] + Height,
                           S>::type>::type;
};

///
///
///
/// Tensor
///
///
///

template <typename TShape, typename TType = f32, typename TDevice = CPU<0>>
class Tensor {
private:
  std::unique_ptr<TType[]> data;
  const std::vector<int64_t> SHAPE_VEC =
      std::vector(SHAPE_DIMS.begin(), SHAPE_DIMS.end());

public:
  using TensorShape = TShape;
  using TensorType = TType;
  using TensorDevice = TDevice;

  static constexpr int64_t DIMS = TShape::DIMS;
  static constexpr int64_t NELEMS = TShape::NELEMS;
  static constexpr std::array<int64_t, DIMS> SHAPE_DIMS = TShape::SHAPE_DIMS;
  static constexpr torch::ScalarType DTYPE = TType::DTYPE;
  static constexpr torch::DeviceType DEVICE = TDevice::DEVICE;

  torch::Tensor base;

  // Constructors

  Tensor()
      : base(torch::zeros(SHAPE_VEC, torch::TensorOptions().dtype(DTYPE).device(
                                         DEVICE, TDevice::INDEX))) {}
  explicit Tensor(const Tensor &other) : base(other->base) {}
  explicit Tensor(Tensor &&other) : base(other->base) {}

  // Operators

  Tensor &operator=(const Tensor &other) {
    this->base = other.base;
    return this;
  }

  [[nodiscard]] Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      data = std::move(other.data);
    }
    return *this;
  }

  [[nodiscard]] Tensor operator+(const Tensor &other) const noexcept {
    Tensor result;
    for (int64_t i = 0; i < TShape::NELEMS; i++) {
      result.data[i] = other.data[i] + this->data[i];
    }
    return result;
  }

  void operator+=(const Tensor &other) noexcept {
    for (int64_t i = 0; i < TShape::NELEMS; i++) {
      this->data[i] += other.data[i];
    }
  }

  [[nodiscard]] Tensor operator-(const Tensor &other) const noexcept {
    Tensor result;
    for (int64_t i = 0; i < TShape::NELEMS; i++) {
      result.data[i] = other.data[i] + this->data[i];
    }
    return result;
  }

  void operator-=(const Tensor &other) noexcept {
    for (int64_t i = 0; i < TShape::NELEMS; i++) {
      this->data[i] -= other.data[i];
    }
  }

  [[nodiscard]] Tensor operator*(const Tensor &other) const noexcept {
    Tensor result;
    for (int64_t i = 0; i < TShape::NELEMS; i++) {
      result.data[i] = other.data[i] * this->data[i];
    }
    return result;
  }

  void operator*=(const Tensor &other) noexcept {
    for (int64_t i = 0; i < TShape::NELEMS; i++) {
      this->data[i] *= other.data[i];
    }
  }

  bool operator==(const Tensor &other) const noexcept {
    for (int64_t i = 0; i < TShape::NELEMS; i++) {
      if (this->data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Tensor &other) const noexcept {
    return !(*this == other);
  }

  void print() const {
    for (int64_t i = 0; i < TShape::NELEMS; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << std::endl;
  }

  // Reshape
  template <int64_t... NewDims>
  [[nodiscard]] Tensor<Shape<NewDims...>, TType, TDevice> reshape() const {
    static_assert(Shape<NewDims...>::NELEMS == TShape::NELEMS,
                  "\nShape mismatch: To reshape the shapes must have the same"
                  "number of elements.\n");
    Tensor<Shape<NewDims...>, TType, TDevice> result;
    std::copy(this->data.get(), this->data.get() + TShape::NELEMS,
              result.data.get());
    return result;
  }

  template <typename NewShape>
  [[nodiscard]] Tensor<NewShape, TType, TDevice> reshape() const {
    static_assert(NewShape::NELEMS == TShape::NELEMS,
                  "\nShape mismatch: To reshape the shapes must have the same"
                  "number of elements.\n");
    Tensor<NewShape, TType, TDevice> result;
    std::copy(this->data.get(), this->data.get() + TShape::NELEMS,
              result.data.get());
    return result;
  }

  // Matmul
  template <typename NewShape>
  [[nodiscard]] auto
  matmul(Tensor<NewShape, TType, TDevice> &other) const noexcept
      -> Tensor<typename MatmulShape<TShape, NewShape>::type, TType, TDevice> {
    Tensor<typename MatmulShape<TShape, NewShape>::type, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  };

  // Stack
  template <int64_t Dim = 0, int64_t NTensors = 2>
  [[nodiscard]] static auto stack(const std::array<Tensor, NTensors> &others)
      -> Tensor<typename TShape::template prepend<NTensors>, TType,
                TDevice> const {
    static_assert(
        Dim < TShape::DIMS,
        "\nDimension mismatch: To stack the tensors the given dimension"
        " must be within the range 0..Shape::DIMS.\n");
    Tensor<typename TShape::template prepend<NTensors>, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  template <int64_t Dim = 0, int64_t NTensors = 2>
  [[nodiscard]] static auto stack(const Tensor other[NTensors])
      -> Tensor<typename TShape::template prepend<NTensors>, TType,
                TDevice> const {
    static_assert(
        Dim < TShape::DIMS,
        "\nDimension mismatch: To stack the tensors the given dimension"
        " must be within the range 0..Shape::DIMS.\n");
    Tensor<typename TShape::template prepend<NTensors>, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  template <typename... Tensors>
  [[nodiscard]] auto stack(const Tensors &...others)
      -> const Tensor<typename TShape::template prepend<sizeof...(Tensors) + 1>,
                      TType, TDevice> {
    static_assert(AreAllSameAs<Tensor, Tensors...>(),
                  "\nAll tensors provided must have the same dimensions.\n");
    Tensor<typename TShape::template prepend<sizeof...(Tensors) + 1>, TType,
           TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Cat
  template <int64_t Dim = 0, int64_t NTensors = 2>
  [[nodiscard]] static auto cat(const std::array<Tensor, NTensors> &others)
      -> Tensor<typename CatArrayShape<TShape, Dim, NTensors>::type, TType,
                TDevice> const {
    static_assert(
        Dim < TShape::DIMS,
        "\nDimension mismatch: To concatenate the tensors the given dimension"
        " must be within the range 0..Shape::DIMS.\n");
    Tensor<typename CatArrayShape<TShape, Dim, NTensors>::type, TType, TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  template <int64_t Dim = 0, int64_t NTensors = 2>
  [[nodiscard]] static auto cat(const Tensor others[NTensors])
      -> Tensor<typename CatArrayShape<TShape, Dim, NTensors>::type, TType,
                TDevice> const {
    static_assert(
        Dim < TShape::DIMS,
        "\nDimension mismatch: To concatenate the tensors the given dimension"
        " must be within the range 0..Shape::DIMS.\n");
    Tensor<typename CatArrayShape<TShape, Dim, NTensors>::type, TType, TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  template <int64_t Dim = 0, typename... Shape2>
  [[nodiscard]] auto cat(const Tensor<Shape2, TType, TDevice> &...others) const
      -> Tensor<typename CatShape<Dim, TShape, Shape2...>::type, TType,
                TDevice> {
    static_assert(
        Dim < TShape::DIMS,
        "\nDimension mismatch: To concatenate the tensors the given dimension"
        " must be within the range 0..Shape::DIMS.\n");
    Tensor<typename CatShape<Dim, TShape, Shape2...>::type, TType, TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Transpose
  template <int64_t... Dims>
  [[nodiscard]] auto transpose() const
      -> Tensor<typename TransposeShape<Shape<Dims...>, TShape>::type, TType,
                TDevice> {
    static_assert(sizeof...(Dims) == TShape::DIMS,
                  "\nThe length new order must be equal to the number of "
                  "dimensions.\n");
    static_assert(AreAllUnique<Shape<Dims...>>::value,
                  "\nThe provided order doesn't cover all numbers from 0 to "
                  "DIMS all exactly once.\n");
    static_assert(AreAllLessThanN<sizeof...(Dims), Shape<Dims...>>::value,
                  "\nThe provided order doesn't cover all numbers from 0 to "
                  "DIMS all exactly once.\n");
    Tensor<typename TransposeShape<Shape<Dims...>, TShape>::type, TType,
           TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Flatten
  [[nodiscard]] auto flatten() const
      -> Tensor<Shape<TShape::NELEMS>, TType, TDevice> {
    Tensor<Shape<TShape::NELEMS>, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Unsqueeze <Dims>
  template <int64_t... Dims>
  [[nodiscard]] auto unsqueeze() const -> Tensor<
      typename VariadicUnsqueezeShape<0, Shape<Dims...>, TShape>::type, TType,
      TDevice> {
    Tensor<typename VariadicUnsqueezeShape<0, Shape<Dims...>, TShape>::type,
           TType, TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Unsqueeze
  [[nodiscard]] auto unsqueeze() const
      -> Tensor<typename VariadicUnsqueezeShape<0, Shape<0>, TShape>::type,
                TType, TDevice> {
    return this->unsqueeze<0>();
  }

  // Unsqueeze dim
  template <int64_t Dim = 1>
  [[nodiscard]] auto unsqueeze_dim() const
      -> Tensor<typename UnsqueezeDimShape<Dim - TShape::DIMS, TShape>::type,
                TType, TDevice> {
    Tensor<typename UnsqueezeDimShape<Dim - TShape::DIMS, TShape>::type, TType,
           TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Squeeze
  [[nodiscard]] auto squeeze() const
      -> Tensor<typename SqueezeShape<TShape>::type, TType, TDevice> {
    Tensor<typename SqueezeShape<TShape>::type, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Squeeze<Dims>
  template <int64_t... Dims>
  [[nodiscard]] auto squeeze() const
      -> Tensor<typename VariadicSqueezeShape<0, Shape<Dims...>, TShape>::type,
                TType, TDevice> {
    Tensor<typename VariadicSqueezeShape<0, Shape<Dims...>, TShape>::type,
           TType, TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Arange
  template <int64_t Start = 0, int64_t End, int64_t Step = 1>
  [[nodiscard]] static auto arange() noexcept
      -> Tensor<Shape<(End - Start) / Step>, TType, TDevice> const {
    static_assert(Start <= End, "\nStart must be <= End.\n");
    static_assert(TShape::DIMS == 1, "\nArange only creates 1-dimensional "
                                     "tensors.\n");
    static_assert(TShape::NELEMS == (End - Start) / Step,
                  "\nShape mismatch.\n");
    Tensor<Shape<(End - Start) / Step>, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Swap dims
  template <int64_t I = 0, int64_t J = 1>
  [[nodiscard]] auto swap_dims() const
      -> Tensor<typename SwapDimsShape<I, J, TShape>::type, TType, TDevice> {
    Tensor<typename SwapDimsShape<I, J, TShape>::type, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Permute
  template <int64_t... Dims>
  [[nodiscard]] auto permute() const -> Tensor<Shape<Dims...>, TType, TDevice> {
    static_assert(
        Shape<Dims...>::DIMS == TShape::DIMS,
        "\nThe resulting shape must have the same number of dimensions "
        "as the original one.\n");
    static_assert(Shape<Dims...>::NELEMS == TShape::NELEMS,
                  "\nThe resulting shape must have the same number of elements "
                  "as the original one.\n");
    static_assert(AreAllUnique<TShape>::value,
                  "\nAll dimensions must be unique.\n");
    Tensor<Shape<Dims...>, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Repeat<Dims>
  template <typename... RepeatPairs>
  [[nodiscard]] auto repeat()
      -> const Tensor<typename RepeatShape<TShape, RepeatPairs...>::type, TType,
                      TDevice> {
    Tensor<typename RepeatShape<TShape, RepeatPairs...>::type, TType, TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  // top<K, Dims>
  template <typename... TopPairs>
  [[nodiscard]] auto top() const
      -> Tensor<typename TopShape<TShape, TopPairs...>::type, TType, TDevice> {
    Tensor<typename TopShape<TShape, TopPairs...>::type, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // apply
  void apply(std::function<void(Tensor *)> fn) { fn(this); }

  // narrow<Start, End, Dims>
  template <typename... TensorRanges>
  [[nodiscard]] auto narrow() const
      -> Tensor<typename NarrowShape<TShape, TensorRanges...>::type, TType,
                TDevice> {
    Tensor<typename NarrowShape<TShape, TensorRanges...>::type, TType, TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  // slice<Start, End, Dims>
  template <typename... TensorRanges>
  [[nodiscard]] auto slice() const -> Tensor<
      typename SliceShape<TShape::DIMS, TShape, TensorRanges...>::type, TType,
      TDevice> {
    Tensor<typename SliceShape<TShape::DIMS, TShape, TensorRanges...>::type,
           TType, TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  template <int64_t Start = 0, int64_t End, int64_t Dim = 0>
  [[nodiscard]] auto slice() const -> Tensor<
      typename SliceShape<TShape::DIMS, TShape, Tr<Start, End, Dim>>::type,
      TType, TDevice> {
    return this->slice<Tr<Start, End, Dim>>();
  }

  // to_dtype
  template <typename DType = float>
  [[nodiscard]] auto to_dtype() const -> Tensor<TShape, DType, TDevice> {
    Tensor<TShape, DType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // to_device
  template <typename Device = CPU<0>>
  [[nodiscard]] auto to_device() const -> Tensor<TShape, TType, Device> {
    Tensor<TShape, TType, Device> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Chunk
  template <int64_t Chunks = 1, int64_t Dim = 0>
  [[nodiscard]] auto chunk() const -> std::array<
      Tensor<typename SetDimAtIdx<Dim, TShape::SHAPE_DIMS[Dim] / Chunks,
                                  TShape>::type,
             TType, TDevice>,
      Chunks> {
    static_assert(TShape::SHAPE_DIMS[Dim] % Chunks == 0,
                  "\nThe dimension must be evenly divisible by the chunks.\n");
    std::array<Tensor<typename SetDimAtIdx<
                          Dim, TShape::SHAPE_DIMS[Dim] / Chunks, TShape>::type,
                      TType, TDevice>,
               Chunks>
        result = {};
    // TODO: Implementation with libtorch
    return result;
  }

  // select_only<Dim, Indices...>
  template <typename... SelectPairs>
  [[nodiscard]] auto select_only() const
      -> Tensor<typename SelectOnlyShape<TShape, SelectPairs...>::type, TType,
                TDevice> {
    Tensor<typename SelectOnlyShape<TShape, SelectPairs...>::type, TType,
           TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  template <int64_t Dim = 0, int64_t... Indices>
  [[nodiscard]] auto select_only() const
      -> Tensor<typename SelectOnlyShape<TShape, Ip<Dim, Indices...>>::type,
                TType, TDevice> {
    return this->select_only<Ip<Dim, Indices...>>();
  }

  // select<Dim, Indices...>
  template <typename... SelectPairs>
  [[nodiscard]] auto select() const
      -> Tensor<typename SelectShape<TShape, SelectPairs...>::type, TType,
                TDevice> {
    Tensor<typename SelectShape<TShape, SelectPairs...>::type, TType, TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  template <int64_t Dim = 0, int64_t... Indices>
  [[nodiscard]] auto select() const
      -> Tensor<typename SelectShape<TShape, Ip<Dim, Indices...>>::type, TType,
                TDevice> {
    return this->select<Ip<Dim, Indices...>>();
  }

  // pad
  template <int64_t Top, int64_t Bottom, int64_t Left, int64_t Right>
  [[nodiscard]] auto pad(TType value)
      -> Tensor<typename PadShape<TShape, Top + Bottom, Left + Right>::type,
                TType, TDevice> {
    Tensor<typename PadShape<TShape, Top + Bottom, Left + Right>::type, TType,
           TDevice>
        result;
    // TODO: Implementation with libtorch
    return result;
  }

  // bool_not
  template <typename T = TType>
  [[nodiscard]]
  typename std::enable_if<std::is_same<T, bool>::value, Tensor>::type
  bool_not() const {
    static_assert(std::is_same<T, TType>::value,
                  "\nbool_not is only available for boolean tensors.\n");
    Tensor result;
    // TODO: Implementation with libtorch
    return result;
  }

  // aggregate operations over dims

  // generating like some other tensor:
  // zeros_like
  // ones_like
  // ...

  // libtorch ops like argmax, argmin, clamp...
  // variance ?
  // covariance ?

  // Broadcast ?

  // Doesn't modify shape
  // Flip<Dims>
  // Any
  // Any<Dims>
  // All
  // All<Dims>
  // sort<Dims>
  // fill
  // sampling distributions
  // math functions
};

} // namespace evol

#endif
