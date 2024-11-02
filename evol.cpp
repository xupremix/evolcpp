#include "tensor.hpp"

using namespace evol;

int main() {
  auto t1 = Tensor<Shape<2, 3>>::zeros();
  auto t2 = Tensor<Shape<3, 2>>::ones();
  t1.print();
  t2.print();

  auto t1_2 = t1.to_dtype<i32>();
  auto t2_2 = t2.to_device<CPU<>>();
  t1_2.print();
  t2_2.print();

  return 0;
}
