#include "tensor.hpp"

using namespace evol;

int main() {
  Tensor<Shape<2, 3, 4>> t;

  std::cout << "T.dim: " << t.base.dim() << std::endl;
  for (size_t i = 0; i < t.DIMS; i++) {
    std::cout << "T.size[" << i << "] = " << t.base.size(i) << std::endl;
  }

  std::cout << t.base << std::endl;
  return 0;
}
