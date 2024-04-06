#include <torch/torch.h>
#include <gtest/gtest.h>

#include "svd.h"

TEST(SVDForwardTest, trivialSVD) {
    at::TensorOptions options;
    options = options.dtype(at::kDouble);

    // 3x3 identity matrix
    const int tensorSize = 3;
    const torch::Tensor tensor = torch::eye(tensorSize, options);

    // calculate SVD
    const SafeSVD safeSVD;
    torch::autograd::AutogradContext* ctx = new torch::autograd::AutogradContext;
    const torch::autograd::variable_list usv = safeSVD.forward(ctx, tensor);

    // check norm of U,S,V
    EXPECT_DOUBLE_EQ(usv[0].norm().item<double>(), std::sqrt(tensorSize));
    EXPECT_DOUBLE_EQ(usv[1].norm().item<double>(), std::sqrt(tensorSize));
    EXPECT_DOUBLE_EQ(usv[2].norm().item<double>(), std::sqrt(tensorSize));
}

TEST(SVDForwardTest, orthogonalUV) {
    at::TensorOptions options;
    options = options.dtype(at::kDouble);

    // 3x5 ones matrix
    const int tensorSize1 = 3;
    const int tensorSize2 = 5;
    const torch::Tensor tensor = torch::ones({tensorSize1,tensorSize2}, options);

    // calculate SVD
    const SafeSVD safeSVD;
    torch::autograd::AutogradContext* ctx = new torch::autograd::AutogradContext;
    const torch::autograd::variable_list usv = safeSVD.forward(ctx, tensor);

    // check norm of U,V
    ASSERT_LE(tensorSize1, tensorSize2);
    EXPECT_DOUBLE_EQ(torch::mm(usv[0].t(),usv[0]).norm().item<double>(), std::sqrt(tensorSize1));
    EXPECT_DOUBLE_EQ(torch::mm(usv[0],usv[0].t()).norm().item<double>(), std::sqrt(tensorSize1));
    EXPECT_DOUBLE_EQ(torch::mm(usv[2].t(),usv[2]).norm().item<double>(), std::sqrt(tensorSize1));
}

TEST(SVDBackwardTest, gradEye3) {
    at::TensorOptions options;
    options = options.dtype(at::kDouble);
    options = options.requires_grad(true);

    // 3x3 identity matrix
    const int tensorSize = 3;
    const torch::Tensor tensor = torch::eye(tensorSize, options);

    // calculate SVD from forward
    SafeSVD safeSVD;
    torch::autograd::AutogradContext* ctx = new torch::autograd::AutogradContext;
    const torch::autograd::variable_list usv = safeSVD.forward(ctx, tensor);

    // define loss and calculate gradient from backward
    torch::Tensor loss = usv[1].norm() * usv[1].norm();
    loss.backward();

    // check loss and gradient
    EXPECT_DOUBLE_EQ(loss.norm().item<double>(), tensorSize);
    EXPECT_DOUBLE_EQ(tensor.grad().norm().item<double>(), 2.0 * std::sqrt(tensorSize));
}