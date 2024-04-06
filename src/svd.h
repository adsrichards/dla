#pragma once
#include <torch/torch.h>

class SafeSVD : public torch::autograd::Function<SafeSVD>
{
public:
    SafeSVD();
    ~SafeSVD();

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const torch::autograd::Variable &A);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs);

private:
    double epsilon = 1E-12; // broadening factor in safe inverse
};