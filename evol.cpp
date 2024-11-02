#include "tensor.hpp"
#include <ATen/ops/zeros.h>

using namespace evol;

int main() {

  torch::Tensor t = torch::ones({2, 3});
  at::Tensor a = at::ones({2, 3});
  auto ris = a + t;
  std::cout << ris << std::endl;

  return 0;
}
