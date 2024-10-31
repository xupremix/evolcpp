#include "tensor.hpp"

using namespace tensor;

int main() {
  torch::Tensor t = torch::eye(3);
  std::cout << t << std::endl;
  return 0;
}
