#pragma once

#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <torch/torch.h>

namespace tensor {

///
/// Devices
///
struct Cpu {};
template <size_t N> struct Cuda {};
template <size_t N> struct Metal {};

///
/// TensorPair
///
template <size_t N, size_t D> struct TensorPair {
  static constexpr size_t K = N;
  static constexpr size_t DIM = D;
};
template <size_t K, size_t Dim> using Tp = TensorPair<K, Dim>;

///
/// TensorRange
///
template <size_t S, size_t E, size_t D> struct TensorRange {
  static constexpr size_t START = S;
  static constexpr size_t END = E;
  static constexpr size_t DIM = D;
};
template <size_t Start, size_t End, size_t Dim>
using Tr = TensorRange<Start, End, Dim>;

///
///
///
/// SHAPE
///
///
///
template <size_t... Dims> struct Shape;

template <> struct Shape<> {
  static constexpr size_t DIMS = 0;
  static constexpr size_t NELEMS = 0;
  static constexpr std::array<size_t, DIMS> SHAPE_DIMS = {};
  template <size_t... NewDims> using prepend = Shape<NewDims...>;
  template <size_t... NewDims> using append = Shape<NewDims...>;
};

template <size_t First> struct Shape<First> {
  static constexpr size_t DIMS = 1;
  static constexpr size_t NELEMS = First;
  static constexpr size_t LAST = First;
  static constexpr std::array<size_t, DIMS> SHAPE_DIMS = {First};
  template <size_t... NewDims> using prepend = Shape<NewDims..., First>;
  template <size_t... NewDims> using append = Shape<First, NewDims...>;
};

template <size_t First, size_t Second> struct Shape<First, Second> {
  static constexpr size_t DIMS = 2;
  static constexpr size_t NELEMS = First * Second;
  static constexpr size_t LAST = Second;
  static constexpr size_t PENULTIMATE = First;
  static constexpr std::array<size_t, DIMS> SHAPE_DIMS = {First, Second};
  using AllButLastTwo = Shape<>;
  template <size_t... NewDims> using prepend = Shape<NewDims..., First, Second>;
  template <size_t... NewDims> using append = Shape<First, Second, NewDims...>;
};

template <size_t First, size_t Second, size_t... Rest>
struct Shape<First, Second, Rest...> {
  static constexpr size_t DIMS = 2 + sizeof...(Rest);
  static constexpr size_t NELEMS = First * Second * Shape<Rest...>::NELEMS;
  static constexpr size_t LAST = Shape<Rest...>::LAST;
  static constexpr size_t PENULTIMATE = Shape<Second, Rest...>::PENULTIMATE;
  static constexpr std::array<size_t, DIMS> SHAPE_DIMS = {First, Second,
                                                          Rest...};
  using AllButLastTwo =
      typename Shape<Second, Rest...>::AllButLastTwo::template prepend<First>;
  template <size_t... NewDims>
  using prepend = Shape<NewDims..., First, Second, Rest...>;
  template <size_t... NewDims>
  using append = Shape<First, Second, Rest..., NewDims...>;
};

///
/// IndicesPair
///
template <size_t D, size_t... I> struct IndicesPair {
  static constexpr size_t DIM = D;
  using Indices = Shape<I...>;
};
template <size_t Dim, size_t... Indices>
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

template <size_t First1, size_t First2>
struct IsSameAsPack<Shape<First1>, Shape<First2>>
    : std::__bool_constant<First1 == First2> {};

template <size_t First1, size_t... Rest1, size_t First2, size_t... Rest2>
struct IsSameAsPack<Shape<First1, Rest1...>, Shape<First2, Rest2...>>
    : std::__bool_constant<
          (First1 == First2) &&
          IsSameAsPack<Shape<Rest1...>, Shape<Rest2...>>::value> {};

///
/// IsCatCompatible
///
template <int Idx, typename P1, typename P2> struct IsCatCompatible;

template <size_t First1, size_t First2>
struct IsCatCompatible<0, Shape<First1>, Shape<First2>> : std::true_type {};

template <size_t First1, size_t... Rest1, size_t First2, size_t... Rest2>
struct IsCatCompatible<0, Shape<First1, Rest1...>, Shape<First2, Rest2...>>
    : std::__bool_constant<
          IsSameAsPack<Shape<Rest1...>, Shape<Rest2...>>::value> {};

template <int Idx, size_t First1, size_t... Rest1, size_t First2,
          size_t... Rest2>
struct IsCatCompatible<Idx, Shape<First1, Rest1...>, Shape<First2, Rest2...>>
    : std::__bool_constant<
          First1 == First2 &&
          IsCatCompatible<Idx - 1, Shape<Rest1...>, Shape<Rest2...>>::value> {};

///
/// SetDimAtIdx
///
template <size_t Idx, size_t Dim, typename T> struct SetDimAtIdx;
template <size_t Dim, size_t First, size_t... Dims>
struct SetDimAtIdx<0, Dim, Shape<First, Dims...>> {
  using type = Shape<Dim, Dims...>;
};
template <size_t Idx, size_t Dim, size_t First, size_t... Rest>
struct SetDimAtIdx<Idx, Dim, Shape<First, Rest...>> {
  static_assert(Idx <= sizeof...(Rest), "\nError: Index out of bounds.\n");
  using type =
      typename SetDimAtIdx<Idx - 1, Dim,
                           Shape<Rest...>>::type::template prepend<First>;
};

///
/// IsNotInPack
///
template <size_t N, typename T> struct IsNotInPack;

template <size_t N, size_t First>
struct IsNotInPack<N, Shape<First>> : std::__bool_constant<N != First> {};

template <size_t N, size_t First, size_t... Rest>
struct IsNotInPack<N, Shape<First, Rest...>>
    : std::__bool_constant<N != First &&
                           IsNotInPack<N, Shape<Rest...>>::value> {};

///
/// AreAllUnique
///
template <typename T> struct AreAllUnique;

template <> struct AreAllUnique<Shape<>> : std::true_type {};

template <size_t First> struct AreAllUnique<Shape<First>> : std::true_type {};

template <size_t First, size_t... Rest>
struct AreAllUnique<Shape<First, Rest...>>
    : std::__bool_constant<IsNotInPack<First, Shape<Rest...>>::value &&
                           AreAllUnique<Shape<Rest...>>::value> {};

///
/// AreAllLessThanN
///
template <size_t N, typename T> struct AreAllLessThanN;

template <size_t N> struct AreAllLessThanN<N, Shape<>> : std::true_type {};

template <size_t N, size_t First>
    struct AreAllLessThanN<N, Shape<First>> : std::__bool_constant < First<N> {
};

template <size_t N, size_t First, size_t... Rest>
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
template <size_t... Dims> struct AreAllInIncreasingOrder;

template <size_t First>
struct AreAllInIncreasingOrder<First> : std::true_type {};

template <size_t First, size_t Second>
    struct AreAllInIncreasingOrder<First, Second> : std::__bool_constant <
                                                    First<Second> {};

template <size_t First, size_t Second, size_t... Rest>
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
template <typename S, size_t Idx, size_t Mul> struct CatArrayShape;
template <size_t Mul> struct CatArrayShape<Shape<>, 0, Mul> {
  using type = Shape<>;
};
template <size_t First, size_t Mul> struct CatArrayShape<Shape<First>, 0, Mul> {
  using type = Shape<First * Mul>;
};
template <size_t First, size_t... Rest, size_t Mul>
struct CatArrayShape<Shape<First, Rest...>, 0, Mul> {
  using type = Shape<First * Mul, Rest...>;
};
template <size_t First, size_t... Rest, size_t Idx, size_t Mul>
struct CatArrayShape<Shape<First, Rest...>, Idx, Mul> {
  static_assert(Idx <= sizeof...(Rest), "Error: Index out of bounds.");
  using type = typename CatArrayShape<Shape<Rest...>, Idx - 1,
                                      Mul>::type::template prepend<First>;
};

//
// CatShape
//

template <size_t Idx, typename ShapeType, typename... Shapes> struct CatShape;

template <size_t Idx, typename ShapeType> struct CatShape<Idx, ShapeType> {
  static_assert(Idx < ShapeType::DIMS, "Index out of bounds.");
};

template <size_t Idx, typename ShapeType, typename FirstShape>
struct CatShape<Idx, ShapeType, FirstShape> {
  static_assert(Idx < ShapeType::DIMS, "Index out of bounds.");
  static_assert(FirstShape::DIMS == ShapeType::DIMS,
                "Shapes must have the same number of dimensions.");
  static_assert(
      IsCatCompatible<Idx, ShapeType, FirstShape>::value,
      "Shapes must have the same dimensions except for the one at Idx");
  static constexpr size_t new_dim =
      ShapeType::SHAPE_DIMS[Idx] + FirstShape::SHAPE_DIMS[Idx];
  using type = typename SetDimAtIdx<Idx, new_dim, ShapeType>::type;
};

template <size_t Idx, typename ShapeType, typename FirstShape,
          typename... RestShapes>
struct CatShape<Idx, ShapeType, FirstShape, RestShapes...> {
  static_assert(Idx < ShapeType::DIMS, "Index out of bounds.");
  static_assert(FirstShape::DIMS == ShapeType::DIMS,
                "Shapes must have the same number of dimensions.");
  static_assert(
      IsCatCompatible<Idx, ShapeType, FirstShape>::value,
      "Shapes must have the same dimensions except for the one at Idx");
  static constexpr size_t new_dim =
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
template <size_t First, typename S2> struct TransposeShape<Shape<First>, S2> {
  static_assert(
      First < S2::DIMS,
      "\nError: The order contains too many items, must be == Shape::DIMS.\n");
  using type = Shape<S2::SHAPE_DIMS[First]>;
};
template <size_t First, size_t... Rest, typename S2>
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
template <size_t Dim, typename S> struct UnsqueezeShape;

template <> struct UnsqueezeShape<0, Shape<>> {
  using type = Shape<1>;
};

template <size_t First, size_t... Dims>
struct UnsqueezeShape<0, Shape<First, Dims...>> {
  using type = Shape<1, First, Dims...>;
};

template <size_t Dim, size_t First, size_t... Rest>
struct UnsqueezeShape<Dim, Shape<First, Rest...>> {
  using type =
      typename UnsqueezeShape<Dim - 1,
                              Shape<Rest...>>::type::template prepend<First>;
};

///
/// VariadicUnsqueeze
///
template <size_t Curr, typename D, typename S> struct VariadicUnsqueezeShape;

template <size_t Curr, size_t First, typename S>
struct VariadicUnsqueezeShape<Curr, Shape<First>, S> {
  using type = typename UnsqueezeShape<First + Curr, S>::type;
};

template <size_t Curr, size_t First, size_t... Rest, typename S>
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
template <size_t Dim, typename S> struct UnsqueezeDimShape;

template <typename S> struct UnsqueezeDimShape<0, S> {
  using type = typename S::template prepend<1>;
};

template <size_t Dim, typename S> struct UnsqueezeDimShape {
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

template <size_t First> struct SqueezeShape<Shape<First>> {
  using type =
      typename std::conditional<First == 1, Shape<>, Shape<First>>::type;
};

template <size_t First, size_t... Rest>
struct SqueezeShape<Shape<First, Rest...>> {
  using type = typename std::conditional<
      First == 1, typename SqueezeShape<Shape<Rest...>>::type,
      typename SqueezeShape<Shape<Rest...>>::type::template prepend<First>>::
      type;
};

///
/// SqueezeShapeAtIdx
///
template <size_t Idx, typename S> struct SqueezeShapeAtIdx;

template <size_t First, size_t... Rest>
struct SqueezeShapeAtIdx<0, Shape<First, Rest...>> {
  using type = typename std::conditional<First == 1, Shape<Rest...>,
                                         Shape<First, Rest...>>::type;
};

template <size_t Idx, size_t First, size_t... Rest>
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
template <size_t Curr, typename T, typename S> struct VariadicSqueezeShape;

template <size_t Curr, size_t First, typename S>
struct VariadicSqueezeShape<Curr, Shape<First>, S> {
  using type = typename SqueezeShapeAtIdx<First - Curr, S>::type;
};

template <size_t Curr, size_t First, size_t... Rest, typename S>
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
template <size_t I, size_t J, typename S> struct SwapDimsShape {
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
template <size_t Curr, typename S, typename... TensorRanges> struct SliceShape;

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

template <size_t Curr, typename S, typename First>
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

template <size_t Curr, typename S, typename First, typename... Rest>
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
template <typename S, size_t Height, size_t Width> struct PadShape {
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

template <typename TShape, typename TType = float, typename TDevice = Cpu>
class Tensor {
private:
  std::unique_ptr<TType[]> data;

public:
  using TensorShape = TShape;
  using TensorType = TType;
  using TensorDevice = TDevice;

  static constexpr size_t DIMS = TShape::DIMS;
  static constexpr size_t NELEMS = TShape::NELEMS;
  static constexpr std::array<size_t, DIMS> SHAPE_DIMS = TShape::SHAPE_DIMS;

  // Constructors

  [[nodiscard]] Tensor() : data(new TType[TShape::NELEMS]) {
    std::fill(data.get(), data.get() + TShape::NELEMS, TType{});
  }
  [[nodiscard]] Tensor(const TType *values) : data(new TType[TShape::NELEMS]) {
    std::copy(values, values + TShape::NELEMS, data.get());
  }
  [[nodiscard]] Tensor(const Tensor &other) : data(new TType[TShape::NELEMS]) {
    std::copy(other.data.get(), other.data.get() + TShape::NELEMS, data.get());
  }
  [[nodiscard]] Tensor(Tensor &&other) noexcept : data(std::move(other.data)) {}

  // Operators

  [[nodiscard]] Tensor &operator=(const Tensor &other) {
    if (this != &other) {
      data.reset(new TType[TShape::NELEMS]); // Allocate new memory
      std::copy(other.data.get(), other.data.get() + TShape::NELEMS,
                data.get());
    }
    return *this;
  }

  [[nodiscard]] Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      data = std::move(other.data);
    }
    return *this;
  }

  [[nodiscard]] Tensor operator+(const Tensor &other) const noexcept {
    Tensor result;
    for (size_t i = 0; i < TShape::NELEMS; i++) {
      result.data[i] = other.data[i] + this->data[i];
    }
    return result;
  }

  void operator+=(const Tensor &other) noexcept {
    for (size_t i = 0; i < TShape::NELEMS; i++) {
      this->data[i] += other.data[i];
    }
  }

  [[nodiscard]] Tensor operator-(const Tensor &other) const noexcept {
    Tensor result;
    for (size_t i = 0; i < TShape::NELEMS; i++) {
      result.data[i] = other.data[i] + this->data[i];
    }
    return result;
  }

  void operator-=(const Tensor &other) noexcept {
    for (size_t i = 0; i < TShape::NELEMS; i++) {
      this->data[i] -= other.data[i];
    }
  }

  [[nodiscard]] Tensor operator*(const Tensor &other) const noexcept {
    Tensor result;
    for (size_t i = 0; i < TShape::NELEMS; i++) {
      result.data[i] = other.data[i] * this->data[i];
    }
    return result;
  }

  void operator*=(const Tensor &other) noexcept {
    for (size_t i = 0; i < TShape::NELEMS; i++) {
      this->data[i] *= other.data[i];
    }
  }

  bool operator==(const Tensor &other) const noexcept {
    for (size_t i = 0; i < TShape::NELEMS; i++) {
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
    for (size_t i = 0; i < TShape::NELEMS; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << std::endl;
  }

  // Reshape
  template <size_t... NewDims>
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
  template <size_t Dim = 0, size_t NTensors = 2>
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

  template <size_t Dim = 0, size_t NTensors = 2>
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
  template <size_t Dim = 0, size_t NTensors = 2>
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

  template <size_t Dim = 0, size_t NTensors = 2>
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

  template <size_t Dim = 0, typename... Shape2>
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
  template <size_t... Dims>
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
  template <size_t... Dims>
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
  template <size_t Dim = 1>
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
  template <size_t... Dims>
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
  template <size_t Start = 0, size_t End, size_t Step = 1>
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
  template <size_t I = 0, size_t J = 1>
  [[nodiscard]] auto swap_dims() const
      -> Tensor<typename SwapDimsShape<I, J, TShape>::type, TType, TDevice> {
    Tensor<typename SwapDimsShape<I, J, TShape>::type, TType, TDevice> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Permute
  template <size_t... Dims>
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

  template <size_t Start = 0, size_t End, size_t Dim = 0>
  [[nodiscard]] auto slice() const -> Tensor<
      typename SliceShape<TShape::DIMS, TShape, Tr<Start, End, Dim>>::type,
      TType, TDevice> {
    return this->slice<Tr<Start, End, Dim>>();
  }

  // to_dtype
  template <typename DType = float>
  [[nodiscard]] auto to_dtype() const -> Tensor<TShape, DType> {
    Tensor<TShape, DType> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // to_device
  template <typename Device = Cpu>
  [[nodiscard]] auto to_device() const -> Tensor<TShape, TType, Device> {
    Tensor<TShape, TType, Device> result;
    // TODO: Implementation with libtorch
    return result;
  }

  // Chunk
  template <size_t Chunks = 1, size_t Dim = 0>
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

  template <size_t Dim = 0, size_t... Indices>
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

  template <size_t Dim = 0, size_t... Indices>
  [[nodiscard]] auto select() const
      -> Tensor<typename SelectShape<TShape, Ip<Dim, Indices...>>::type, TType,
                TDevice> {
    return this->select<Ip<Dim, Indices...>>();
  }

  // pad
  template <size_t Top, size_t Bottom, size_t Left, size_t Right>
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

} // namespace tensor

#endif
