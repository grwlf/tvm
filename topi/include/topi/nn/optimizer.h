#include "tvm/tvm.h"
#include "topi/tags.h"

namespace topi {
namespace nn {
using namespace tvm;

inline Array<Tensor> SGDOptimizer(Array<Tensor> weights,
                                  Array<Tensor> gradients,
                                  Expr learning_rate,
                                  std::string name = "tensor",
                                  std::string tag = kOptimizer) {
  Array<Tensor> result;
  for (int i = 0; i < weights.size(); i++) {
    Array<Expr> output_shape(weights[i]->shape.begin(), weights[i]->shape.end());
    auto func = 
      [weights, gradients, learning_rate, i]
      (const Array<Var>& input_indices) {
        return weights[i](input_indices) - learning_rate * gradients[i](input_indices);
      };
    result.push_back(compute(output_shape, func, name, tag));
  }
  return result;
}

inline Array<Array<Tensor>> AdamOptimizer(Array<Tensor> weights,
                                          Array<Tensor> gradients,
                                          Expr learning_rate,
                                          Expr t,
                                          Expr beta1,
                                          Expr beta2,
                                          Expr epsilon,
                                          Array<Tensor> ms,
                                          Array<Tensor> vs,
                                          std::string name = "tensor",
                                          std::string tag = kOptimizer) {
  Array<Array<Tensor>> result;
  Array<Tensor> new_weights;
  Array<Tensor> new_ms;
  Array<Tensor> new_vs;
  auto lr = learning_rate * sqrt(Expr(1.0f) - power(beta2, t)) / (Expr(1.0f) - power(beta1, t));
  for (int i = 0; i < weights.size(); i++) {
    Array<Expr> output_shape(weights[i]->shape.begin(), weights[i]->shape.end());
    auto func1 = 
      [gradients, ms, beta1, i]
      (const Array<Var>& input_indices) {
        return beta1 * ms[i](input_indices) + (Expr(1.0f) - beta1) * gradients[i](input_indices);
      };
    new_ms.push_back(compute(output_shape, func1, name, tag));
    auto func2 = 
      [gradients, vs, beta2, i]
      (const Array<Var>& input_indices) {
        return beta2 * vs[i](input_indices) + (Expr(1.0f) - beta2) * gradients[i](input_indices) * gradients[i](input_indices);
      };
    new_vs.push_back(compute(output_shape, func2, name, tag));
    auto func3 = 
      [weights, new_ms, new_vs, lr, epsilon, i]
      (const Array<Var>& input_indices) {
        return weights[i](input_indices) - lr * new_ms[i](input_indices) / (sqrt(new_vs[i](input_indices)) + epsilon);
      };
    new_weights.push_back(compute(output_shape, func3, name, tag));
  }
  result.push_back(new_weights);
  result.push_back(new_ms);
  result.push_back(new_vs);
  return result;
}

}
}
