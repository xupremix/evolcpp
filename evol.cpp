#include "tensor.hpp"

using namespace evol;

int main() {

  auto t1 = Tensor<Shape<1, 2, 3>, f32, CUDA<>>::ones();
  auto t2 = Tensor<Shape<1, 3, 4>, f32, CUDA<>>::ones();
  auto t3 = t1.matmul(t2);

  t3.print();

  return 0;
}
