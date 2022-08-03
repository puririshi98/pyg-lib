#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/script.h>

#include "pyg_lib/csrc/utils/convert.h"

namespace pyg {
namespace ops {

namespace {

std::vector<at::Tensor> _grouped_matmul(const std::vector<at::Tensor>& input,
                                        const std::vector<at::Tensor>& other) {
  // TODO (matthias) Add TensorArg definitions.
  // TODO (matthias) Add dispatcher support.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::cuda_grouped_matmul", "")
                       .typed<decltype(_grouped_matmul)>();
  return op.call(input, other);
}

at::Tensor _segment_matmul(const at::Tensor& input,
                           const at::Tensor& ptr,
                           const at::Tensor& other) {
  // TODO (matthias) Add TensorArg definitions.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul", "")
                       .typed<decltype(_segment_matmul)>();
  return op.call(input, ptr, other);
}

std::vector<at::Tensor> concat(std::vector<at::Tensor> t1,
                               std::vector<at::Tensor> t2) {
  for (size_t i = 0; i < t2.size(); ++i) {
    t1.push_back(t2[i]);
  }
  return t1;
}

at::Tensor _segment_matmul_back(const at::Tensor& input,
                                const at::Tensor& ptr,
                                const at::Tensor& other) {
  // TODO (matthias) Add TensorArg definitions.
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::segment_matmul_back_kern", "")
                       .typed<decltype(_segment_matmul_back)>();
  return op.call(input, ptr, other);
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

std::tuple<at::Tensor, at::Tensor> segment_matmul_backwards(
    const at::Tensor& input,
    const at::Tensor& ptr,
    const at::Tensor& other,
    const at::Tensor& grad_out,
    bool input_req_grad,
    bool other_req_grad) {
  auto input_grad = Variable();
  if (input_req_grad) {
    auto other_t = other.transpose(-2, -1);
    input_grad = _segment_matmul(grad_out, ptr, other_t);
  }

  auto other_grad = Variable();
  if (other_req_grad) {
    auto input_t = input.transpose(-2, -1);
    other_grad = _segment_matmul_back(input_t, ptr, grad_out);
  }
  return std::make_tuple(input_grad, other_grad);
};


class GroupedMatmul : public torch::autograd::Function<GroupedMatmul> {
 public:
  static variable_list forward(AutogradContext* ctx,
                               variable_list input,
                               variable_list other) {
    auto out = _grouped_matmul(input, other);
    variable_list input_and_other = concat(input, other);
    ctx->save_for_backward(input_and_other);
    ctx->saved_data["input_len"] = (int)input.size();
    return out;
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
    auto input_and_other = ctx->get_saved_variables();
    int input_len = ctx->saved_data["input_len"].toInt();
    variable_list input(input_and_other.begin(),
                        input_and_other.begin() + input_len);
    variable_list other(input_and_other.begin() + input_len,
                        input_and_other.end());
    variable_list other_grad;
    // For Simplicity:
    // We assume entire input variable list either requires grad or does not
    if (torch::autograd::any_variable_requires_grad(other)) {
      for (size_t i = 0; i < input.size(); ++i)
        other[i] = other[i].transpose(-2, -1);
      other_grad = _grouped_matmul(grad_outs, other);
    } else {
      for (size_t i = 0; i < other.size(); ++i)
        other_grad.push_back(Variable());
    }

    variable_list input_grad;
    if (torch::autograd::any_variable_requires_grad(input)) {
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = input[i].transpose(-2, -1);
      input_grad = _grouped_matmul(input, grad_outs);
    } else {
      for (size_t i = 0; i < input.size(); ++i)
        input_grad.push_back(Variable());
    }
    return concat(input_grad, other_grad);
  }
};

class SegmentMatmul : public torch::autograd::Function<SegmentMatmul> {
 public:
  static variable_list forward(AutogradContext* ctx,
                               Variable input,
                               Variable ptr,
                               Variable other) {
    Variable out = _segment_matmul(input, ptr, other);
    ctx->save_for_backward({input, ptr, other});
    return {out};
  }

  static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto input = saved[0], ptr = saved[1], other = saved[2];
    auto input_grad = Variable();
    auto other_grad = Variable();
    bool input_req_grad = torch::autograd::any_variable_requires_grad({input});
    bool other_req_grad = torch::autograd::any_variable_requires_grad({other});
    auto both_grads = segment_matmul_backwards(input, ptr, other, grad_out,
                                               input_req_grad, other_req_grad);
    input_grad = std::get<0>(both_grads);
    other_grad = std::get<1>(both_grads);
    return {input_grad, Variable(), other_grad};
  }
};

}  // namespace

// Performs matrix multiplication across list of elements.
std::vector<at::Tensor> grouped_matmul(const std::vector<at::Tensor>& input,
                                       const std::vector<at::Tensor>& other) {
  // TODO (matthias) Add autograd support.
  /* return GroupedMatmul::apply(input, other)[0]; */
  return _grouped_matmul(input, other);
}

// Performs matrix multiplication according to segments.
at::Tensor segment_matmul(const at::Tensor& input,
                          const at::Tensor& ptr,
                          const at::Tensor& other) {
  // TODO (matthias) Add autograd support.
  /* return SegmentMatmul::apply(input, ptr, other)[0]; */
  return _segment_matmul(input, ptr, other);
}

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def((
      "pyg::grouped_matmul(Tensor[] input, Tensor[] other) -> Tensor[]"));
  m.def(
      ("pyg::segment_matmul(Tensor input, Tensor ptr, "
                             "Tensor other) -> Tensor"));
  m.def(
      ("pyg::segment_matmul_backwards(Tensor input, Tensor ptr, Tensor other, "
      "Tensor grad_out, bool input_req_grad, bool other_req_grad) -> (Tensor, "
      "Tensor)"));
}

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl("pyg::segment_matmul_backwards", segment_matmul_backwards);
}

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl("pyg::grouped_matmul", grouped_matmul);
  m.impl("pyg::segment_matmul", segment_matmul);
}

}  // namespace ops
}  // namespace pyg
