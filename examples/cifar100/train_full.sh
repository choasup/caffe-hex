#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar100/cifar10_full_solver.prototxt --gpu=2 $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar100/cifar10_full_solver_lr1.prototxt \
    --snapshot=examples/cifar100/cifar10_full_iter_60000.solverstate.h5 --gpu=2 $@

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar100/cifar10_full_solver_lr2.prototxt \
    --snapshot=examples/cifar100/cifar10_full_iter_65000.solverstate.h5 --gpu=2 $@
