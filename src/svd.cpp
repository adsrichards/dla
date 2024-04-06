#include <torch/torch.h>
#include "svd.h"

SafeSVD::SafeSVD() {}
SafeSVD::~SafeSVD() {}

inline static torch::Tensor safe_inverse(const torch::Tensor &x, double epsilon)
{
    return x / (x.pow(2) + epsilon);
}

torch::autograd::variable_list SafeSVD::forward(
    torch::autograd::AutogradContext *ctx,
    const torch::autograd::Variable &A)
{
    auto result = torch::svd(A);
    auto U = std::get<0>(result);
    auto S = std::get<1>(result);
    auto V = std::get<2>(result);
    ctx->save_for_backward({U, S, V});

    return {U, S, V};
}

torch::autograd::variable_list SafeSVD::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs)
{
    auto saved_variables = ctx->get_saved_variables();
    auto U = saved_variables[0];
    auto S = saved_variables[1];
    auto V = saved_variables[2];

    auto dU = grad_outputs[0];
    auto dS = grad_outputs[1];
    auto dV = grad_outputs[2];

    auto Vt = V.t();
    auto Ut = U.t();
    auto M = U.size(0);
    auto N = V.size(0);
    auto NS = S.size(0);

    auto F = (S - S.view({NS, 1}));
    F = safe_inverse(F, 1E-12);
    F.diagonal().fill_(0);

    auto G = (S + S.view({NS, 1}));
    G.diagonal().fill_(INFINITY);
    G = 1 / G;

    auto UdU = Ut.mm(dU);
    auto VdV = Vt.mm(dV);

    auto Su = (F + G) * (UdU - UdU.t()) / 2;
    auto Sv = (F - G) * (VdV - VdV.t()) / 2;

    auto dA = U.mm(Su + Sv + torch::diag(dS)).mm(Vt);

    if (M > NS)
    {
        dA += (torch::eye(M, torch::kFloat32) - U.mm(Ut)).mm(dU / S).mm(Vt);
    }

    if (N > NS)
    {
        dA += (U / S).mm(dV.t()).mm(torch::eye(N, torch::kFloat32) - V.mm(Vt));
    }

    return {dA};
}