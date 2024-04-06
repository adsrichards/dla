#pragma once
#include <torch/torch.h>

namespace dla
{
    class SafeSVD : public torch::autograd::Function<SafeSVD>
    {
    public:
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext* ctx,
            const torch::autograd::Variable &A)
        {
            auto result = torch::svd(A);
            auto U = std::get<0>(result);
            auto S = std::get<1>(result);
            auto V = std::get<2>(result);
            ctx->save_for_backward({U, S, V});

            return {U, S, V};
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad_outputs)
        {
            at::TensorOptions options;
            options = options.dtype(at::kDouble);

            const double broadening = 1.E-12;

            const auto saved_variables = ctx->get_saved_variables();
            const auto U = saved_variables[0];
            const auto S = saved_variables[1];
            const auto V = saved_variables[2];

            const auto dU = grad_outputs[0];
            const auto dS = grad_outputs[1];
            const auto dV = grad_outputs[2];

            const auto Vt = V.t();
            const auto Ut = U.t();
            const auto M = U.size(0);
            const auto N = V.size(0);
            const auto NS = S.size(0);

            auto F = (S - S.view({NS, 1}));
            F /= (F * F + broadening);
            F.diagonal().fill_(0);

            auto G = (S + S.view({NS, 1}));
            G.diagonal().fill_(INFINITY);
            G = 1 / G;

            const auto UdU = Ut.mm(dU);
            const auto VdV = Vt.mm(dV);

            const auto Su = (F + G) * (UdU - UdU.t()) / 2;
            const auto Sv = (F - G) * (VdV - VdV.t()) / 2;

            auto dA = U.mm(Su + Sv + torch::diag(dS)).mm(Vt);

            if (M > NS)
            {
                dA += (torch::eye(M, options) - U.mm(Ut)).mm(dU / S).mm(Vt);
            }

            if (N > NS)
            {
                dA += (U / S).mm(dV.t()).mm(torch::eye(N, options) - V.mm(Vt));
            }

            return {dA};
        }
    };
}