# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import torch
from typing import Any, Iterable, List, Tuple, Callable
import torch.distributed as dist

def get_gpu_states(fwd_gpu_devices) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_states

def get_gpu_device(*args):

    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))
    return fwd_gpu_devices

def set_device_states(fwd_cpu_state, devices, states) -> None:
    torch.set_rng_state(fwd_cpu_state)
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

def detach_and_grad(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)

def get_cpu_and_gpu_states(gpu_devices):
    return torch.get_rng_state(), get_gpu_states(gpu_devices)

class ReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions  = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c3) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0  = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0*alpha0
            ctx.cpu_states_1, ctx.gpu_states_1  = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1*alpha1
            ctx.cpu_states_2, ctx.gpu_states_2  = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2*alpha2
            ctx.cpu_states_3, ctx.gpu_states_3  = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3*alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
            torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            
            g3_up = g3_right
            g3_left = g3_up*alpha3 ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)                    
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1/alpha3)*(c3 - oup3) ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up*alpha2 ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)          
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left*alpha3 ##alpha3 update
            torch.autograd.backward(cout3, g3_up)
            
            with torch.no_grad():
                c2_left = (1/alpha2)*(c2 - oup2) ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right+c1.grad
            g1_left = g1_up*alpha1 ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)     
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left*alpha2 ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1/alpha1)*(c1 - oup1) ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up*alpha0 ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left ## Fusion
            
            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)     
            oup0 = l0(x, c1_left)            
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left*alpha1 ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1/alpha0)*(c0 - oup0) ## feature reverse
            gx_up = x.grad ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left*alpha0 ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
        
        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left


class ReverseFunction_first(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        # if type(c3) == int
        ctx.first_col = True

        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)


        return None, None, gx_up, None, None, None, None

class ReverseFunction_nofuse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c3) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, None) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, None) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, None) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, None)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, None)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, None)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left

class ReverseFunction_nofuse_first(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args

        ctx.first_col = True

        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, None) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, None) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, None) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, None)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, None)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, None)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        return None, None, gx_up, None, None, None, None

class ReverseFunctionSimpleDecoder4level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c3) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, g0_left, g1_left, g2_left, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left

class ReverseFunctionSimpleDecoder4level_first(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args

        ctx.first_col = True

        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, g0_left, g1_left, g2_left, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left
class ReverseFunctionSimpleMDecoder4level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 9
        [x, c0, c1, c2, c3, d0, d1, d2, d3] = args
        if type(c3) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, d0) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, d1) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, d2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, d3) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3, d0, d1, d2, d3)
        return x, c0, c1, c2, c3, d0, d1, d2, d3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, d0, d1, d2, d3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, gd0_right, gd1_right, gd2_right, gd3_right = grad_outputs
        (x, c0, c1, c2, c3, d0, d1, d2, d3) = detach_and_grad((x, c0, c1, c2, c3, d0, d1, d2, d3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, d3)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, d2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, d1)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, d0)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            gd0_left = d0.grad
            gd1_left = d1.grad
            gd2_left = d2.grad
            gd3_left = d3.grad
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, g0_left, g1_left, g2_left, None, gd0_left, gd1_left, gd2_left, gd3_left
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, gd0_left, gd1_left, gd2_left, gd3_left


class ReverseFunctionX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c3) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left

class ReverseFunction5Level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 6
        [x, c0, c1, c2, c3, c4] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4)
        return x, c0, c1, c2, c3, c4

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right = grad_outputs
        (x, c0, c1, c2, c3, c4) = detach_and_grad((x, c0, c1, c2, c3, c4))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_down = g3_right + c3.grad
            g3_left = g3_down * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_down, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left
            g2_down = g2_right + c2.grad
            g2_left = g2_down * alpha2  ## shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)

            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_down, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ## alpha3 update
            torch.autograd.backward(cout3, g3_down)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_down = g1_right + c1.grad
            g1_left = g1_down * alpha1  ## shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_down, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ## alpha2 update
            torch.autograd.backward(cout2, g2_down)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_down = g0_right + c0.grad
            g0_left = g0_down * alpha0  ## shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_down, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ## alpha1 update
            torch.autograd.backward(cout1, g1_down)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_down = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ## alpha0 update
            torch.autograd.backward(cout0, g0_down)

        if ctx.first_col:
            return None, None, gx_down, None, None, None, None, None
        else:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, g4_left
class ReverseFunction5Level_KernelEst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 7
        [inp_image, x, c0, c1, c2, c3, c4] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, inp_image) + c4 * alpha4
        ctx.save_for_backward(inp_image, x, c0, c1, c2, c3, c4)
        return x, c0, c1, c2, c3, c4

    @staticmethod
    def backward(ctx, *grad_outputs):
        inp_image, x, c0, c1, c2, c3, c4 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right = grad_outputs
        (x, c0, c1, c2, c3, c4) = detach_and_grad((x, c0, c1, c2, c3, c4))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, inp_image)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_down = g3_right + c3.grad
            g3_left = g3_down * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_down, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left
            g2_down = g2_right + c2.grad
            g2_left = g2_down * alpha2  ## shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)

            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_down, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ## alpha3 update
            torch.autograd.backward(cout3, g3_down)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_down = g1_right + c1.grad
            g1_left = g1_down * alpha1  ## shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_down, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ## alpha2 update
            torch.autograd.backward(cout2, g2_down)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_down = g0_right + c0.grad
            g0_left = g0_down * alpha0  ## shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_down, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ## alpha1 update
            torch.autograd.backward(cout1, g1_down)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_down = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ## alpha0 update
            torch.autograd.backward(cout0, g0_down)

        if ctx.first_col:
            return None, None, None, gx_down, g0_left, g1_left, g2_left, None, None
        else:
            return None, None, None, gx_down, g0_left, g1_left, g2_left, g3_left, g4_left
class ReverseFunction5Level_Adapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 6
        [x, c0, c1, c2, c3, c4] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4)
        return x, c0, c1, c2, c3, c4

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right = grad_outputs
        (x, c0, c1, c2, c3, c4) = detach_and_grad((x, c0, c1, c2, c3, c4))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None)
            # torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_down = g3_right + c3.grad if c3.grad is not None else g3_right
            g3_left = g3_down * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)

            # torch.autograd.backward(oup3, g3_down, retain_graph=True)
            c4_left.requires_grad = False
            # cout4 = c4_left * alpha4  ## alpha3 update
            # torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left
            g2_down = g2_right + c2.grad if c2.grad is not None else g2_right
            g2_left = g2_down * alpha2  ## shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)

            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_down, retain_graph=True)
            c3_left.requires_grad = False
            # cout3 = c3_left * alpha3  ## alpha3 update
            # torch.autograd.backward(cout3, g3_down)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_down = g1_right + c1.grad if c1.grad is not None else g1_right
            g1_left = g1_down * alpha1  ## shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            # torch.autograd.backward(oup1, g1_down, retain_graph=True)
            c2_left.requires_grad = False
            # cout2 = c2_left * alpha2  ## alpha2 update
            # torch.autograd.backward(cout2, g2_down)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_down = g0_right + c0.grad if c0.grad is not None else g0_right
            g0_left = g0_down * alpha0  ## shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_down, retain_graph=True)
            c1_left.requires_grad = False
            # cout1 = c1_left * alpha1  ## alpha1 update
            # torch.autograd.backward(cout1, g1_down)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_down = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            # cout0 = c0_left * alpha0  ## alpha0 update
            # torch.autograd.backward(cout0, g0_down)

        if ctx.first_col:
            return None, None, gx_down, None, None, None, None, None
        else:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, g4_left
class ReverseFunctionDecoder5Level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 6
        [x, c0, c1, c2, c3, c4] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4)
        return x, c0, c1, c2, c3, c4

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right = grad_outputs
        (x, c0, c1, c2, c3, c4) = detach_and_grad((x, c0, c1, c2, c3, c4))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_down = g3_right + c3.grad
            g3_left = g3_down * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_down, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left
            g2_down = g2_right + c2.grad
            g2_left = g2_down * alpha2  ## shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)

            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_down, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ## alpha3 update
            torch.autograd.backward(cout3, g3_down)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_down = g1_right + c1.grad
            g1_left = g1_down * alpha1  ## shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_down, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ## alpha2 update
            torch.autograd.backward(cout2, g2_down)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_down = g0_right + c0.grad
            g0_left = g0_down * alpha0  ## shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_down, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ## alpha1 update
            torch.autograd.backward(cout1, g1_down)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_down = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ## alpha0 update
            torch.autograd.backward(cout0, g0_down)

        if ctx.first_col:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, None
        else:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, g4_left

class ReverseFunctionDecoder5Level_Adapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 6
        [x, c0, c1, c2, c3, c4] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4)
        return x, c0, c1, c2, c3, c4

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right = grad_outputs
        (x, c0, c1, c2, c3, c4) = detach_and_grad((x, c0, c1, c2, c3, c4))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            g3_down = g3_right + c3.grad if c3.grad is not None else g3_right
            g3_left = g3_down * alpha3  ##shortcut
            g2_down = g2_right + c2.grad if c2.grad is not None else g2_right
            g2_left = g2_down * alpha2  ## shortcut
            g1_down = g1_right + c1.grad if c1.grad is not None else g1_right
            g1_left = g1_down * alpha1  ## shortcut
            g0_down = g0_right + c0.grad if c0.grad is not None else g0_right
            g0_left = g0_down * alpha0  ## shortcut

            gx_down = x.grad  ## Fusion


        if ctx.first_col:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, None
        else:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, g4_left
class ReverseFunctionDecoder5Level_Adapterv1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 6
        [x, c0, c1, c2, c3, c4] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4)
        return x, c0, c1, c2, c3, c4

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right = grad_outputs
        (x, c0, c1, c2, c3, c4) = detach_and_grad((x, c0, c1, c2, c3, c4))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None)
            # torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_down = g3_right + c3.grad if c3.grad is not None else g3_right
            g3_left = g3_down * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_down, retain_graph=True)
            c4_left.requires_grad = False
            # cout4 = c4_left * alpha4  ## alpha3 update
            # torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left
            g2_down = g2_right + c2.grad
            g2_left = g2_down * alpha2  ## shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)

            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_down, retain_graph=True)
            c3_left.requires_grad = False
            # cout3 = c3_left * alpha3  ## alpha3 update
            # torch.autograd.backward(cout3, g3_down)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_down = g1_right + c1.grad
            g1_left = g1_down * alpha1  ## shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_down, retain_graph=True)
            c2_left.requires_grad = False
            # cout2 = c2_left * alpha2  ## alpha2 update
            # torch.autograd.backward(cout2, g2_down)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_down = g0_right + c0.grad
            g0_left = g0_down * alpha0  ## shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            # torch.autograd.backward(oup0, g0_down, retain_graph=True)
            c1_left.requires_grad = False
            # cout1 = c1_left * alpha1  ## alpha1 update
            # torch.autograd.backward(cout1, g1_down)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_down = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            # cout0 = c0_left * alpha0  ## alpha0 update
            # torch.autograd.backward(cout0, g0_down)

        if ctx.first_col:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, None
        else:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, g4_left
class ReverseFunctionDecoderF5Level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 6
        [x, c0, c1, c2, c3, c4] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4)
        return x, c0, c1, c2, c3, c4

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right = grad_outputs
        (x, c0, c1, c2, c3, c4) = detach_and_grad((x, c0, c1, c2, c3, c4))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_down = g3_right + c3.grad
            g3_left = g3_down * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_down, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left
            g2_down = g2_right + c2.grad
            g2_left = g2_down * alpha2  ## shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)

            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_down, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ## alpha3 update
            torch.autograd.backward(cout3, g3_down)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_down = g1_right + c1.grad
            g1_left = g1_down * alpha1  ## shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_down, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ## alpha2 update
            torch.autograd.backward(cout2, g2_down)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_down = g0_right + c0.grad
            g0_left = g0_down * alpha0  ## shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_down, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ## alpha1 update
            torch.autograd.backward(cout1, g1_down)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_down = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ## alpha0 update
            torch.autograd.backward(cout0, g0_down)

        if ctx.first_col:
            return None, None, gx_down, None, None, None, None, None
        else:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, g4_left
class ReverseFunctionDegradationDecoder5Level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 6
        [x, c0, c1, c2, c3, c4] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4)
        return x, c0, c1, c2, c3, c4

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right = grad_outputs
        (x, c0, c1, c2, c3, c4) = detach_and_grad((x, c0, c1, c2, c3, c4))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_down = g3_right + c3.grad
            g3_left = g3_down * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_down, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left
            g2_down = g2_right + c2.grad
            g2_left = g2_down * alpha2  ## shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)

            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_down, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ## alpha3 update
            torch.autograd.backward(cout3, g3_down)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_down = g1_right + c1.grad
            g1_left = g1_down * alpha1  ## shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_down, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ## alpha2 update
            torch.autograd.backward(cout2, g2_down)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_down = g0_right + c0.grad
            g0_left = g0_down * alpha0  ## shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_down, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ## alpha1 update
            torch.autograd.backward(cout1, g1_down)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_down = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ## alpha0 update
            torch.autograd.backward(cout0, g0_down)

        if ctx.first_col:
            return None, None, gx_down, None, None, None, None, None
        else:
            return None, None, gx_down, g0_left, g1_left, g2_left, g3_left, g4_left
class SimpleReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left
class SimpleReverseFunctionKernelPrior(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 8
        [x, c0, c1, c2, c3, k0, k1, k2] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, k0) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, k1) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, k2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3, k0, k1, k2)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, k0, k1, k2 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3, k0, k1, k2) = detach_and_grad((x, c0, c1, c2, c3, k0, k1, k2))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, k2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, k1)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, k0)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None, k0.grad, k1.grad, k2.grad
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, k0.grad, k1.grad, k2.grad
class ReverseFunctionKernelPrior(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 8
        [x, c0, c1, c2, c3, k0, k1, k2] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1, k0) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2, k1) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3, k2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3, k0, k1, k2)
        return x, c0, c1, c2, c3, k0, k1, k2

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, k0, k1, k2 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, k0_right, k1_right, k2_right = grad_outputs
        # (x, c0, c1, c2, c3, k0, k1, k2) = detach_and_grad((x, c0, c1, c2, c3, k0, k1, k2))
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left, k2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left, k1)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left, k0)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, None, None, None
class ReverseFunctionKernelPriorV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, lk0, lk1, lk2 = run_functions
        alpha0, alpha1, alpha2, alpha3, kalpha0, kalpha1, kalpha2 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 9
        [x, c0, c1, c2, c3, k, k0, k1, k2] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
            k0 = lk0(k, k1) + k0 * kalpha0
            ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
            k1 = lk1(k0, k2) + k1 * kalpha1
            ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
            k2 = lk2(k1, None) + k2 * kalpha2

            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1, k0) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2, k1) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3, k2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3, k, k0, k1, k2)
        return x, c0, c1, c2, c3, k, k0, k1, k2

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, k, k0, k1, k2 = ctx.saved_tensors
        l0, l1, l2, l3, lk0, lk1, lk2 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, kalpha0, kalpha1, kalpha2 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, gk_right, gk0_right, gk1_right, gk2_right = grad_outputs
        (x, c0, c1, c2, c3, k, k0, k1, k2) = detach_and_grad((x, c0, c1, c2, c3, k, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left, k2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)
            
            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left, k1)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left, k0)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

            # kernel
            gk2_up = gk2_right
            gk2_left = gk2_up * kalpha2  ##shortcut
            set_device_states(ctx.cpu_states_20, ctx.gpu_devices, ctx.gpu_states_20)
            koup2 = lk2(k1, None)
            torch.autograd.backward(koup2, gk2_up, retain_graph=True)

            with torch.no_grad():
                k2_left = (1 / kalpha2) * (k2 - koup2)  ## feature reverse

            gk1_up = gk1_right + k1.grad
            gk1_left = gk1_up * kalpha1  ##shortcut

            (k2_left,) = detach_and_grad((k2_left,))
            set_device_states(ctx.cpu_states_10, ctx.gpu_devices, ctx.gpu_states_10)
            koup1 = lk1(k0, k2_left)
            torch.autograd.backward(koup1, gk1_up, retain_graph=True)
            k2_left.requires_grad = False
            kout2 = k2_left * kalpha2  ##alpha2 update
            torch.autograd.backward(kout2, gk2_up)

            with torch.no_grad():
                k1_left = (1 / kalpha1) * (k1 - koup1)  ## feature reverse
            gk0_up = gk0_right + k0.grad
            gk0_left = gk0_up * kalpha0  ##shortcut
            gk2_left = gk2_left + k2_left.grad if k2_left.grad is not None else gk2_left  ## Fusion

            (k1_left,) = detach_and_grad((k1_left,))
            set_device_states(ctx.cpu_states_00, ctx.gpu_devices, ctx.gpu_states_00)
            oupk0 = lk0(k, k1_left)
            torch.autograd.backward(oupk0, gk0_up, retain_graph=True)
            k1_left.requires_grad = False
            kout1 = k1_left * kalpha1  ##alpha1 update
            torch.autograd.backward(kout1, gk1_up)

            with torch.no_grad():
                k0_left = (1 / kalpha0) * (k0 - oupk0)  ## feature reverse
            gk_up = k.grad  ## Fusion
            gk1_left = gk1_left + k1_left.grad if k1_left.grad is not None else gk1_left  ## Fusion
            k0_left.requires_grad = False
            kout0 = k0_left * kalpha0  ##alpha0 update
            torch.autograd.backward(kout0, gk0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None, gk_up, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, gk_up, gk0_left, gk1_left, gk2_left


class ReverseFunctionKernelPrior5levelV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4, lk0, lk1, lk2 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4, kalpha0, kalpha1, kalpha2 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 10
        [x, c0, c1, c2, c3, c4, k, k0, k1, k2] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
            k0 = lk0(k, k1) + k0 * kalpha0
            ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
            k1 = lk1(k0, k2) + k1 * kalpha1
            ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
            k2 = lk2(k1, None) + k2 * kalpha2

            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1, k0) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2, k1) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3, k2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4, k, k0, k1, k2)
        return x, c0, c1, c2, c3, c4, k, k0, k1, k2

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4, k, k0, k1, k2 = ctx.saved_tensors
        l0, l1, l2, l3, l4, lk0, lk1, lk2 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4, kalpha0, kalpha1, kalpha2 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right, gk_right, gk0_right, gk1_right, gk2_right = grad_outputs
        (x, c0, c1, c2, c3, c4, k, k0, k1, k2) = detach_and_grad((x, c0, c1, c2, c3, c4, k, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_up = g3_right + c3.grad
            g3_left = g3_up * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left

            # g3_up = g3_right
            # g3_left = g3_up * alpha3  ##shortcut
            # set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            # oup3 = l3(c2, None)
            # torch.autograd.backward(oup3, g3_up, retain_graph=True)
            # with torch.no_grad():
            #     c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse

            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left, k2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left, k1)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left, k0)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

            # kernel
            gk2_up = gk2_right
            gk2_left = gk2_up * kalpha2  ##shortcut
            set_device_states(ctx.cpu_states_20, ctx.gpu_devices, ctx.gpu_states_20)
            koup2 = lk2(k1, None)
            torch.autograd.backward(koup2, gk2_up, retain_graph=True)

            with torch.no_grad():
                k2_left = (1 / kalpha2) * (k2 - koup2)  ## feature reverse

            gk1_up = gk1_right + k1.grad
            gk1_left = gk1_up * kalpha1  ##shortcut

            (k2_left,) = detach_and_grad((k2_left,))
            set_device_states(ctx.cpu_states_10, ctx.gpu_devices, ctx.gpu_states_10)
            koup1 = lk1(k0, k2_left)
            torch.autograd.backward(koup1, gk1_up, retain_graph=True)
            k2_left.requires_grad = False
            kout2 = k2_left * kalpha2  ##alpha2 update
            torch.autograd.backward(kout2, gk2_up)

            with torch.no_grad():
                k1_left = (1 / kalpha1) * (k1 - koup1)  ## feature reverse
            gk0_up = gk0_right + k0.grad
            gk0_left = gk0_up * kalpha0  ##shortcut
            gk2_left = gk2_left + k2_left.grad if k2_left.grad is not None else gk2_left  ## Fusion

            (k1_left,) = detach_and_grad((k1_left,))
            set_device_states(ctx.cpu_states_00, ctx.gpu_devices, ctx.gpu_states_00)
            oupk0 = lk0(k, k1_left)
            torch.autograd.backward(oupk0, gk0_up, retain_graph=True)
            k1_left.requires_grad = False
            kout1 = k1_left * kalpha1  ##alpha1 update
            torch.autograd.backward(kout1, gk1_up)

            with torch.no_grad():
                k0_left = (1 / kalpha0) * (k0 - oupk0)  ## feature reverse
            gk_up = k.grad  ## Fusion
            gk1_left = gk1_left + k1_left.grad if k1_left.grad is not None else gk1_left  ## Fusion
            k0_left.requires_grad = False
            kout0 = k0_left * kalpha0  ##alpha0 update
            torch.autograd.backward(kout0, gk0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None, None, gk_up, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, g4_left, gk_up, gk0_left, gk1_left, gk2_left

class ReverseFunctionKernelFeature3level_Adapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        # lk0, lk1, lk2 = run_functions
        # kalpha0, kalpha1, kalpha2 = alpha
        # ctx.run_functions = run_functions
        # ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 4
        [k, k0, k1, k2] = args
        if type(k0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        # with torch.no_grad():
        #     gpu_devices = get_gpu_device(*args)
        #     ctx.gpu_devices = gpu_devices
        #     ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
        #     k0 = lk0(k, k1) + k0 * kalpha0
        #     ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
        #     k1 = lk1(k0, k2) + k1 * kalpha1
        #     ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
        #     k2 = lk2(k1, None) + k2 * kalpha2

        ctx.save_for_backward(k, k0, k1, k2)
        return k, k0, k1, k2

    @staticmethod
    def backward(ctx, *grad_outputs):
        k, k0, k1, k2 = ctx.saved_tensors
        # lk0, lk1, lk2 = ctx.run_functions
        # kalpha0, kalpha1, kalpha2 = ctx.alpha
        gk_right, gk0_right, gk1_right, gk2_right = grad_outputs
        (k, k0, k1, k2) = detach_and_grad((k, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        # with torch.enable_grad(), \
        #         torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
        #         torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
        #         torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            # kernel
        gk2_up = gk2_right
        gk2_left = gk2_up
        gk1_up = gk1_right
        gk1_left = gk1_up
        gk0_up = gk0_right
        gk0_left = gk0_up
        gk_up = k.grad  ## Fusion

        if ctx.first_col:
            return None, None, gk_up, None, None, None
        else:
            return None, None, gk_up, gk0_left, gk1_left, gk2_left
class ReverseFunctionKernelFeature3level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        lk0, lk1, lk2 = run_functions
        kalpha0, kalpha1, kalpha2 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 4
        [k, k0, k1, k2] = args
        if type(k0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
            k0 = lk0(k, k1) + k0 * kalpha0
            ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
            k1 = lk1(k0, k2) + k1 * kalpha1
            ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
            k2 = lk2(k1, None) + k2 * kalpha2

        ctx.save_for_backward(k, k0, k1, k2)
        return k, k0, k1, k2

    @staticmethod
    def backward(ctx, *grad_outputs):
        k, k0, k1, k2 = ctx.saved_tensors
        lk0, lk1, lk2 = ctx.run_functions
        kalpha0, kalpha1, kalpha2 = ctx.alpha
        gk_right, gk0_right, gk1_right, gk2_right = grad_outputs
        (k, k0, k1, k2) = detach_and_grad((k, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            # kernel
            gk2_up = gk2_right
            gk2_left = gk2_up * kalpha2  ##shortcut
            set_device_states(ctx.cpu_states_20, ctx.gpu_devices, ctx.gpu_states_20)
            koup2 = lk2(k1, None)
            torch.autograd.backward(koup2, gk2_up, retain_graph=True)

            with torch.no_grad():
                k2_left = (1 / kalpha2) * (k2 - koup2)  ## feature reverse

            gk1_up = gk1_right + k1.grad
            gk1_left = gk1_up * kalpha1  ##shortcut

            (k2_left,) = detach_and_grad((k2_left,))
            set_device_states(ctx.cpu_states_10, ctx.gpu_devices, ctx.gpu_states_10)
            koup1 = lk1(k0, k2_left)
            torch.autograd.backward(koup1, gk1_up, retain_graph=True)
            k2_left.requires_grad = False
            kout2 = k2_left * kalpha2  ##alpha2 update
            torch.autograd.backward(kout2, gk2_up)

            with torch.no_grad():
                k1_left = (1 / kalpha1) * (k1 - koup1)  ## feature reverse
            gk0_up = gk0_right + k0.grad
            gk0_left = gk0_up * kalpha0  ##shortcut
            gk2_left = gk2_left + k2_left.grad if k2_left.grad is not None else gk2_left  ## Fusion

            (k1_left,) = detach_and_grad((k1_left,))
            set_device_states(ctx.cpu_states_00, ctx.gpu_devices, ctx.gpu_states_00)
            oupk0 = lk0(k, k1_left)
            torch.autograd.backward(oupk0, gk0_up, retain_graph=True)
            k1_left.requires_grad = False
            kout1 = k1_left * kalpha1  ##alpha1 update
            torch.autograd.backward(kout1, gk1_up)

            with torch.no_grad():
                k0_left = (1 / kalpha0) * (k0 - oupk0)  ## feature reverse
            gk_up = k.grad  ## Fusion
            gk1_left = gk1_left + k1_left.grad if k1_left.grad is not None else gk1_left  ## Fusion
            k0_left.requires_grad = False
            kout0 = k0_left * kalpha0  ##alpha0 update
            torch.autograd.backward(kout0, gk0_up)

        if ctx.first_col:
            return None, None, gk_up, None, None, None
        else:
            return None, None, gk_up, gk0_left, gk1_left, gk2_left
class ReverseFunctionMultisacleSimpleEncoder3level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        lk0, lk1, lk2 = run_functions
        kalpha0, kalpha1, kalpha2 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 7
        [k, k0, k1, k2, u0, u1, u2] = args

        if type(k0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
            k0 = lk0(k, u0) + k0 * kalpha0
            ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
            k1 = lk1(k0, u1) + k1 * kalpha1
            ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
            k2 = lk2(k1, u2) + k2 * kalpha2
        if type(u0) == int:
            ctx.fuse = False
            ctx.save_for_backward(k, k0, k1, k2)
        else:
            ctx.fuse = True
            # u0 = torch.zeros_like(k0)
            # u1 = torch.zeros_like(k1)
            # u2 = torch.zeros_like(k2)
            ctx.save_for_backward(k, k0, k1, k2, u0, u1, u2)
        return k, k0, k1, k2, u0, u1, u2

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.fuse:
            k, k0, k1, k2, u0, u1, u2 = ctx.saved_tensors
        else:
            k, k0, k1, k2 = ctx.saved_tensors
            u0, u1, u2 = 0, 0, 0
        lk0, lk1, lk2 = ctx.run_functions
        kalpha0, kalpha1, kalpha2 = ctx.alpha
        gk_right, gk0_right, gk1_right, gk2_right, gu0_right, gu1_right, gu2_right = grad_outputs
        if ctx.fuse:
            (k, k0, k1, k2, u0, u1, u2) = detach_and_grad((k, k0, k1, k2, u0, u1, u2))
        else:
            (k, k0, k1, k2) = detach_and_grad((k, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            # kernel
            gk2_up = gk2_right
            gk2_left = gk2_up * kalpha2  ##shortcut
            set_device_states(ctx.cpu_states_20, ctx.gpu_devices, ctx.gpu_states_20)
            koup2 = lk2(k1, u2)
            torch.autograd.backward(koup2, gk2_up, retain_graph=True)

            with torch.no_grad():
                k2_left = (1 / kalpha2) * (k2 - koup2)  ## feature reverse

            gk1_up = gk1_right + k1.grad
            gk1_left = gk1_up * kalpha1  ##shortcut

            (k2_left,) = detach_and_grad((k2_left,))
            set_device_states(ctx.cpu_states_10, ctx.gpu_devices, ctx.gpu_states_10)
            koup1 = lk1(k0, u1)
            torch.autograd.backward(koup1, gk1_up, retain_graph=True)
            k2_left.requires_grad = False
            kout2 = k2_left * kalpha2  ##alpha2 update
            torch.autograd.backward(kout2, gk2_up)

            with torch.no_grad():
                k1_left = (1 / kalpha1) * (k1 - koup1)  ## feature reverse
            gk0_up = gk0_right + k0.grad
            gk0_left = gk0_up * kalpha0  ##shortcut
            gk2_left = gk2_left + k2_left.grad if k2_left.grad is not None else gk2_left  ## Fusion

            (k1_left,) = detach_and_grad((k1_left,))
            set_device_states(ctx.cpu_states_00, ctx.gpu_devices, ctx.gpu_states_00)
            oupk0 = lk0(k, u0)
            torch.autograd.backward(oupk0, gk0_up, retain_graph=True)
            k1_left.requires_grad = False
            kout1 = k1_left * kalpha1  ##alpha1 update
            torch.autograd.backward(kout1, gk1_up)

            with torch.no_grad():
                k0_left = (1 / kalpha0) * (k0 - oupk0)  ## feature reverse
            gk_up = k.grad  ## Fusion
            if ctx.fuse:
                gu0_left = u0.grad # + gu0_right
                gu1_left = u1.grad # + gu1_right
                gu2_left = u2.grad # + gu2_right
            else:
                gu0_left = None
                gu1_left = None
                gu2_left = None
            gk1_left = gk1_left + k1_left.grad if k1_left.grad is not None else gk1_left  ## Fusion
            k0_left.requires_grad = False
            kout0 = k0_left * kalpha0  ##alpha0 update
            torch.autograd.backward(kout0, gk0_up)
        # print(gk_up, gk0_left, gk1_left, gk2_left, gu0_left, gu1_left, gu2_left)
        if ctx.first_col:
            return None, None, gk_up, None, None, None, gu0_left, gu1_left, gu2_left
        else:
            return None, None, gk_up, gk0_left, gk1_left, gk2_left, gu0_left, gu1_left, gu2_left
class ReverseFunctionSimpleMEncoder4level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 9
        [x, c0, c1, c2, c3, d0, d1, d2, d3] = args
        if type(c3) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, d0) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, d1) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, d2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, d3) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3, d0, d1, d2, d3)
        return x, c0, c1, c2, c3, d0, d1, d2, d3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, d0, d1, d2, d3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, gd0_right, gd1_right, gd2_right, gd3_right = grad_outputs
        (x, c0, c1, c2, c3, d0, d1, d2, d3) = detach_and_grad((x, c0, c1, c2, c3, d0, d1, d2, d3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, d3)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, d2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, d1)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, d0)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            gd0_left = d0.grad
            gd1_left = d1.grad
            gd2_left = d2.grad
            gd3_left = d3.grad
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None, gd0_left, gd1_left, gd2_left, gd3_left
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, gd0_left, gd1_left, gd2_left, gd3_left

class ReverseFunctionSimpleMEncoder4level_first(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 9
        [x, c0, c1, c2, c3, d0, d1, d2, d3] = args

        ctx.first_col = True

        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, d0) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, d1) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, d2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, d3) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3, d0, d1, d2, d3)
        return x, c0, c1, c2, c3, d0, d1, d2, d3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, d0, d1, d2, d3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, gd0_right, gd1_right, gd2_right, gd3_right = grad_outputs
        (x, c0, c1, c2, c3, d0, d1, d2, d3) = detach_and_grad((x, c0, c1, c2, c3, d0, d1, d2, d3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, d3)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, d2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, d1)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, d0)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            gd0_left = d0.grad
            gd1_left = d1.grad
            gd2_left = d2.grad
            gd3_left = d3.grad
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None, gd0_left, gd1_left, gd2_left, gd3_left
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, gd0_left, gd1_left, gd2_left, gd3_left
class ReverseFunctionMultisacleEncoder3level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        lmax0, lmax1, lmax2, lmid0, lmid1, lmid2 = run_functions
        maxalpha0, maxalpha1, maxalpha2, midalpha0, midalpha1, midalpha2 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 8
        [x_max, x_mid, max0, max1, max2, mid0, mid1, mid2] = args

        if type(max0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices

            ctx.cpu_states_01, ctx.gpu_states_01 = get_cpu_and_gpu_states(gpu_devices)
            mid0 = lmid0(x_mid, x_max) + mid0 * midalpha0
            ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
            max0 = lmax0(x_max, x_mid) + max0 * maxalpha0

            ctx.cpu_states_11, ctx.gpu_states_11 = get_cpu_and_gpu_states(gpu_devices)
            mid1 = lmid1(mid0, max0) + mid1 * midalpha1
            ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
            max1 = lmax1(max0, mid0) + max1 * maxalpha1

            ctx.cpu_states_21, ctx.gpu_states_21 = get_cpu_and_gpu_states(gpu_devices)
            mid2 = lmid2(mid1, max1) + mid2 * midalpha2
            ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
            max2 = lmax2(max1, mid1) + max2 * maxalpha2

        ctx.save_for_backward(x_max, x_mid, max0, max1, max2, mid0, mid1, mid2)
        return x_max, x_mid, max0, max1, max2, mid0, mid1, mid2

    @staticmethod
    def backward(ctx, *grad_outputs):
        # if ctx.fuse:
        x_max, x_mid, max0, max1, max2, mid0, mid1, mid2 = ctx.saved_tensors

        lmax0, lmax1, lmax2, lmid0, lmid1, lmid2 = ctx.run_functions
        maxalpha0, maxalpha1, maxalpha2, midalpha0, midalpha1, midalpha2 = ctx.alpha
        gmax_right, gmid_right, gmax0_right, gmax1_right, gmax2_right, gmid0_right, gmid1_right, gmid2_right = grad_outputs
        # if ctx.fuse:
        (x_max, x_mid, max0, max1, max2, mid0, mid1, mid2) = detach_and_grad((x_max, x_mid, max0, max1, max2, mid0, mid1, mid2))
        # else:
        #     (k, k0, k1, k2) = detach_and_grad((k, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            gmax2_up = gmax2_right
            gmax2_left = gmax2_up * maxalpha2  ##shortcut

            gmid2_up = gmid2_right
            gmid2_left = gmid2_up * midalpha2  ##shortcut

            set_device_states(ctx.cpu_states_20, ctx.gpu_devices, ctx.gpu_states_20)
            maxoup2 = lmax2(max1, mid1)
            torch.autograd.backward(maxoup2, gmax2_up, retain_graph=True)

            with torch.no_grad():
                max2_left = (1 / maxalpha2) * (max2 - maxoup2)  ## feature reverse

            gmax1_up = gmax1_right + max1.grad
            gmax1_left = gmax1_up * maxalpha1  ##shortcut
            (max2_left,) = detach_and_grad((max2_left,))

            set_device_states(ctx.cpu_states_21, ctx.gpu_devices, ctx.gpu_states_21)
            midoup2 = lmid2(mid1, max1)
            torch.autograd.backward(midoup2, gmid2_up, retain_graph=True)

            with torch.no_grad():
                mid2_left = (1 / midalpha2) * (mid2 - midoup2)  ## feature reverse

            gmid1_up = gmid1_right + mid1.grad
            gmid1_left = gmid1_up * midalpha1  ##shortcut
            (mid2_left,) = detach_and_grad((mid2_left,))

            set_device_states(ctx.cpu_states_10, ctx.gpu_devices, ctx.gpu_states_10)
            maxoup1 = lmax1(max0, mid0)
            torch.autograd.backward(maxoup1, gmax1_up, retain_graph=True)
            max2_left.requires_grad = False
            maxout2 = max2_left * maxalpha2  ##alpha2 update
            torch.autograd.backward(maxout2, gmax2_up)

            with torch.no_grad():
                max1_left = (1 / maxalpha1) * (max1 - maxoup1)  ## feature reverse
            gmax0_up = gmax0_right + max0.grad
            gmax0_left = gmax0_up * maxalpha0  ##shortcut
            gmax2_left = gmax2_left + max2_left.grad if max2_left.grad is not None else gmax2_left  ## Fusion
            (max1_left,) = detach_and_grad((max1_left,))

            set_device_states(ctx.cpu_states_11, ctx.gpu_devices, ctx.gpu_states_11)
            midoup1 = lmid1(mid0, max0)
            torch.autograd.backward(midoup1, gmid1_up, retain_graph=True)
            mid2_left.requires_grad = False
            midout2 = mid2_left * midalpha2  ##alpha2 update
            torch.autograd.backward(midout2, gmid2_up)

            with torch.no_grad():
                mid1_left = (1 / midalpha1) * (mid1 - midoup1)  ## feature reverse
            gmid0_up = gmid0_right + mid0.grad
            gmid0_left = gmid0_up * midalpha0  ##shortcut
            gmid2_left = gmid2_left + mid2_left.grad if mid2_left.grad is not None else gmid2_left  ## Fusion
            (mid1_left,) = detach_and_grad((mid1_left,))

            set_device_states(ctx.cpu_states_00, ctx.gpu_devices, ctx.gpu_states_00)
            oupmax0 = lmax0(x_max, x_mid)
            torch.autograd.backward(oupmax0, gmax0_up, retain_graph=True)
            max1_left.requires_grad = False
            maxout1 = max1_left * maxalpha1  ##alpha1 update
            torch.autograd.backward(maxout1, gmax1_up)

            with torch.no_grad():
                max0_left = (1 / maxalpha0) * (max0 - oupmax0)  ## feature reverse

            gmax1_left = gmax1_left + max1_left.grad if max1_left.grad is not None else gmax1_left  ## Fusion
            max0_left.requires_grad = False
            maxout0 = max0_left * maxalpha0  ##alpha0 update
            torch.autograd.backward(maxout0, gmax0_up)

            set_device_states(ctx.cpu_states_01, ctx.gpu_devices, ctx.gpu_states_01)
            oupmid0 = lmid0(x_mid, x_max)
            torch.autograd.backward(oupmid0, gmid0_up, retain_graph=True)
            mid1_left.requires_grad = False
            midout1 = mid1_left * midalpha1  ##alpha1 update
            torch.autograd.backward(midout1, gmid1_up)

            with torch.no_grad():
                mid0_left = (1 / midalpha0) * (mid0 - oupmid0)  ## feature reverse

            gmid1_left = gmid1_left + mid1_left.grad if mid1_left.grad is not None else gmid1_left  ## Fusion
            mid0_left.requires_grad = False
            midout0 = mid0_left * midalpha0  ##alpha0 update
            torch.autograd.backward(midout0, gmid0_up)

            gmax_up = x_max.grad
            gmid_up = x_mid.grad
        if ctx.first_col:
            return None, None, gmax_up, gmid_up, None, None, None, None, None, None
        else:
            return None, None, gmax_up, gmid_up, gmax0_left, gmax1_left, gmax2_left, gmid0_left, gmid1_left, gmid2_left

class ReverseFunctionMultisacleSimpleEncoder3level_Adapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        # lk0, lk1, lk2 = run_functions
        # kalpha0, kalpha1, kalpha2 = alpha
        # ctx.run_functions = run_functions
        # ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 7
        [k, k0, k1, k2, u0, u1, u2] = args

        if type(k0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        # with torch.no_grad():
        #     gpu_devices = get_gpu_device(*args)
        #     ctx.gpu_devices = gpu_devices
        #     ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
        #     k0 = lk0(k, u0) + k0 * kalpha0
        #     ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
        #     k1 = lk1(k0, u1) + k1 * kalpha1
        #     ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
        #     k2 = lk2(k1, u2) + k2 * kalpha2
        if type(u0) == int:
            ctx.fuse = False
            ctx.save_for_backward(k, k0, k1, k2)
        else:
            ctx.fuse = True
            # u0 = torch.zeros_like(k0)
            # u1 = torch.zeros_like(k1)
            # u2 = torch.zeros_like(k2)
            ctx.save_for_backward(k, k0, k1, k2, u0, u1, u2)
        return k, k0, k1, k2, u0, u1, u2

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.fuse:
            k, k0, k1, k2, u0, u1, u2 = ctx.saved_tensors
        else:
            k, k0, k1, k2 = ctx.saved_tensors
            u0, u1, u2 = 0, 0, 0
        # lk0, lk1, lk2 = ctx.run_functions
        # kalpha0, kalpha1, kalpha2 = ctx.alpha
        gk_right, gk0_right, gk1_right, gk2_right, gu0_right, gu1_right, gu2_right = grad_outputs
        if ctx.fuse:
            (k, k0, k1, k2, u0, u1, u2) = detach_and_grad((k, k0, k1, k2, u0, u1, u2))
        else:
            (k, k0, k1, k2) = detach_and_grad((k, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        # with torch.enable_grad(), \
        #         torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
        #         torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
        #         torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            # kernel
        gk2_up = gk2_right
        gk2_left = gk2_up
        gk1_up = gk1_right
        gk1_left = gk1_up
        gk0_up = gk0_right
        gk0_left = gk0_up

        gk_up = k.grad  ## Fusion
        if ctx.fuse:
            gu0_left = u0.grad
            gu1_left = u1.grad
            gu2_left = u2.grad
        else:
            gu0_left = None
            gu1_left = None
            gu2_left = None

        if ctx.first_col:
            return None, None, gk_up, None, None, None, gu0_left, gu1_left, gu2_left
        else:
            return None, None, gk_up, gk0_left, gk1_left, gk2_left, gu0_left, gu1_left, gu2_left
class ReverseFunctionSimpleDecoder3level_Adapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        # lk0, lk1, lk2 = run_functions
        # kalpha0, kalpha1, kalpha2 = alpha
        # ctx.run_functions = run_functions
        # ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 4
        [k, k0, k1, k2] = args
        if type(k2) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        # with torch.no_grad():
        #     gpu_devices = get_gpu_device(*args)
        #     ctx.gpu_devices = gpu_devices
        #     ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
        #     k0 = lk0(k, k1) + k0 * kalpha0
        #     ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
        #     k1 = lk1(k0, k2) + k1 * kalpha1
        #     ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
        #     k2 = lk2(k1, None) + k2 * kalpha2

        ctx.save_for_backward(k, k0, k1, k2)
        return k, k0, k1, k2

    @staticmethod
    def backward(ctx, *grad_outputs):
        # print('backward')
        k, k0, k1, k2 = ctx.saved_tensors
        # lk0, lk1, lk2 = ctx.run_functions
        # kalpha0, kalpha1, kalpha2 = ctx.alpha
        gk_right, gk0_right, gk1_right, gk2_right = grad_outputs
        (k, k0, k1, k2) = detach_and_grad((k, k0, k1, k2))

        # with torch.enable_grad(), \
        #         torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
        #         torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
        #         torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            # kernel
        gk2_up = gk2_right
        gk2_left = gk2_up # * kalpha2  ##shortcut

        gk1_up = gk1_right # + k1.grad if k1.grad is not None else gk1_right
        gk1_left = gk1_up # * kalpha1  ##shortcut

        gk0_up = gk0_right # + k0.grad if k0.grad is not None else gk1_right
        gk0_left = gk0_up # * kalpha0  ##shortcut
        gk_up = k.grad  ## Fusion

        if ctx.first_col:
            return None, None, gk_up, gk0_left, gk1_left, None
        else:
            return None, None, gk_up, gk0_left, gk1_left, gk2_left
class ReverseFunctionSimpleDecoder3level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        lk0, lk1, lk2 = run_functions
        kalpha0, kalpha1, kalpha2 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 4
        [k, k0, k1, k2] = args
        if type(k2) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
            k0 = lk0(k, k1) + k0 * kalpha0
            ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
            k1 = lk1(k0, k2) + k1 * kalpha1
            ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
            k2 = lk2(k1, None) + k2 * kalpha2

        ctx.save_for_backward(k, k0, k1, k2)
        return k, k0, k1, k2

    @staticmethod
    def backward(ctx, *grad_outputs):
        k, k0, k1, k2 = ctx.saved_tensors
        lk0, lk1, lk2 = ctx.run_functions
        kalpha0, kalpha1, kalpha2 = ctx.alpha
        gk_right, gk0_right, gk1_right, gk2_right = grad_outputs
        (k, k0, k1, k2) = detach_and_grad((k, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            # kernel
            gk2_up = gk2_right
            gk2_left = gk2_up * kalpha2  ##shortcut
            set_device_states(ctx.cpu_states_20, ctx.gpu_devices, ctx.gpu_states_20)

            koup2 = lk2(k1, None)
            # print('recall')
            torch.autograd.backward(koup2, gk2_up, retain_graph=True)

            with torch.no_grad():
                k2_left = (1 / kalpha2) * (k2 - koup2)  ## feature reverse

            gk1_up = gk1_right + k1.grad
            gk1_left = gk1_up * kalpha1  ##shortcut

            (k2_left,) = detach_and_grad((k2_left,))
            set_device_states(ctx.cpu_states_10, ctx.gpu_devices, ctx.gpu_states_10)
            koup1 = lk1(k0, k2_left)
            torch.autograd.backward(koup1, gk1_up, retain_graph=True)
            k2_left.requires_grad = False
            kout2 = k2_left * kalpha2  ##alpha2 update
            torch.autograd.backward(kout2, gk2_up)

            with torch.no_grad():
                k1_left = (1 / kalpha1) * (k1 - koup1)  ## feature reverse
            gk0_up = gk0_right + k0.grad
            gk0_left = gk0_up * kalpha0  ##shortcut
            gk2_left = gk2_left + k2_left.grad if k2_left.grad is not None else gk2_left  ## Fusion

            (k1_left,) = detach_and_grad((k1_left,))
            set_device_states(ctx.cpu_states_00, ctx.gpu_devices, ctx.gpu_states_00)
            oupk0 = lk0(k, k1_left)
            torch.autograd.backward(oupk0, gk0_up, retain_graph=True)
            k1_left.requires_grad = False
            kout1 = k1_left * kalpha1  ##alpha1 update
            torch.autograd.backward(kout1, gk1_up)

            with torch.no_grad():
                k0_left = (1 / kalpha0) * (k0 - oupk0)  ## feature reverse
            gk_up = k.grad  ## Fusion
            gk1_left = gk1_left + k1_left.grad if k1_left.grad is not None else gk1_left  ## Fusion
            k0_left.requires_grad = False
            kout0 = k0_left * kalpha0  ##alpha0 update
            torch.autograd.backward(kout0, gk0_up)

        if ctx.first_col:
            return None, None, gk_up, gk0_left, gk1_left, None
        else:
            return None, None, gk_up, gk0_left, gk1_left, gk2_left
class ReverseFunctionSimpleDecoder2level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        lk0, lk1, lk2 = run_functions
        kalpha0, kalpha1, kalpha2 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 3
        [k, k0, k1] = args
        if type(k1) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
            k0 = lk0(k, k1) + k0 * kalpha0
            ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
            k1 = lk1(k0, None) + k1 * kalpha1

        ctx.save_for_backward(k, k0, k1)
        return k, k0, k1

    @staticmethod
    def backward(ctx, *grad_outputs):
        k, k0, k1 = ctx.saved_tensors
        lk0, lk1 = ctx.run_functions
        kalpha0, kalpha1 = ctx.alpha
        gk_right, gk0_right, gk1_right = grad_outputs
        (k, k0, k1) = detach_and_grad((k, k0, k1))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            # kernel
            # gk2_up = gk2_right
            # gk2_left = gk2_up * kalpha2  ##shortcut
            # set_device_states(ctx.cpu_states_20, ctx.gpu_devices, ctx.gpu_states_20)
            #
            # koup2 = lk2(k1, None)
            # # print('recall')
            # torch.autograd.backward(koup2, gk2_up, retain_graph=True)
            #
            # with torch.no_grad():
            #     k2_left = (1 / kalpha2) * (k2 - koup2)  ## feature reverse

            gk1_up = gk1_right # + k1.grad
            gk1_left = gk1_up * kalpha1  ##shortcut

            # (k2_left,) = detach_and_grad((k2_left,))
            set_device_states(ctx.cpu_states_10, ctx.gpu_devices, ctx.gpu_states_10)
            koup1 = lk1(k0, None)
            torch.autograd.backward(koup1, gk1_up, retain_graph=True)
            # k2_left.requires_grad = False
            # # kout2 = k2_left * kalpha2  ##alpha2 update
            # torch.autograd.backward(kout2, gk2_up)

            with torch.no_grad():
                k1_left = (1 / kalpha1) * (k1 - koup1)  ## feature reverse
            gk0_up = gk0_right + k0.grad
            gk0_left = gk0_up * kalpha0  ##shortcut
            # gk2_left = gk2_left + k2_left.grad if k2_left.grad is not None else gk2_left  ## Fusion

            (k1_left,) = detach_and_grad((k1_left,))
            set_device_states(ctx.cpu_states_00, ctx.gpu_devices, ctx.gpu_states_00)
            oupk0 = lk0(k, k1_left)
            torch.autograd.backward(oupk0, gk0_up, retain_graph=True)
            k1_left.requires_grad = False
            kout1 = k1_left * kalpha1  ##alpha1 update
            torch.autograd.backward(kout1, gk1_up)

            with torch.no_grad():
                k0_left = (1 / kalpha0) * (k0 - oupk0)  ## feature reverse
            gk_up = k.grad  ## Fusion
            gk1_left = gk1_left + k1_left.grad if k1_left.grad is not None else gk1_left  ## Fusion
            k0_left.requires_grad = False
            kout0 = k0_left * kalpha0  ##alpha0 update
            torch.autograd.backward(kout0, gk0_up)

        if ctx.first_col:
            return None, None, gk_up, gk0_left, None
        else:
            return None, None, gk_up, gk0_left, gk1_left
class ReverseFunctionSimpleMDecoder3level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        lk0, lk1, lk2 = run_functions
        kalpha0, kalpha1, kalpha2 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 7
        [k, k0, k1, k2, d0, d1, d2] = args
        if type(k2) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_00, ctx.gpu_states_00 = get_cpu_and_gpu_states(gpu_devices)
            k0 = lk0(k, d0) + k0 * kalpha0
            ctx.cpu_states_10, ctx.gpu_states_10 = get_cpu_and_gpu_states(gpu_devices)
            k1 = lk1(k0, d1) + k1 * kalpha1
            ctx.cpu_states_20, ctx.gpu_states_20 = get_cpu_and_gpu_states(gpu_devices)
            k2 = lk2(k1, d2) + k2 * kalpha2
        # if type(d0) == int:
        #     d0 = torch.zeros_like(k0)
        #     d1 = torch.zeros_like(k1)
        #     d2 = torch.zeros_like(k2)
        ctx.save_for_backward(k, k0, k1, k2, d0, d1, d2)
        return k, k0, k1, k2, d0, d1, d2

    @staticmethod
    def backward(ctx, *grad_outputs):
        k, k0, k1, k2, d0, d1, d2 = ctx.saved_tensors
        lk0, lk1, lk2 = ctx.run_functions
        kalpha0, kalpha1, kalpha2 = ctx.alpha
        gk_right, gk0_right, gk1_right, gk2_right, gd0_right, gd1_right, gd2_right = grad_outputs
        (k, k0, k1, k2, d0, d1, d2) = detach_and_grad((k, k0, k1, k2, d0, d1, d2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            # kernel
            gk2_up = gk2_right
            gk2_left = gk2_up * kalpha2  ##shortcut
            set_device_states(ctx.cpu_states_20, ctx.gpu_devices, ctx.gpu_states_20)
            koup2 = lk2(k1, d2)
            torch.autograd.backward(koup2, gk2_up, retain_graph=True)

            with torch.no_grad():
                k2_left = (1 / kalpha2) * (k2 - koup2)  ## feature reverse

            gk1_up = gk1_right + k1.grad
            gk1_left = gk1_up * kalpha1  ##shortcut

            (k2_left,) = detach_and_grad((k2_left,))
            set_device_states(ctx.cpu_states_10, ctx.gpu_devices, ctx.gpu_states_10)
            koup1 = lk1(k0, d1)
            torch.autograd.backward(koup1, gk1_up, retain_graph=True)
            k2_left.requires_grad = False
            kout2 = k2_left * kalpha2  ##alpha2 update
            torch.autograd.backward(kout2, gk2_up)

            with torch.no_grad():
                k1_left = (1 / kalpha1) * (k1 - koup1)  ## feature reverse
            gk0_up = gk0_right + k0.grad
            gk0_left = gk0_up * kalpha0  ##shortcut
            gk2_left = gk2_left + k2_left.grad if k2_left.grad is not None else gk2_left  ## Fusion

            (k1_left,) = detach_and_grad((k1_left,))
            set_device_states(ctx.cpu_states_00, ctx.gpu_devices, ctx.gpu_states_00)
            oupk0 = lk0(k, d0)
            torch.autograd.backward(oupk0, gk0_up, retain_graph=True)
            k1_left.requires_grad = False
            kout1 = k1_left * kalpha1  ##alpha1 update
            torch.autograd.backward(kout1, gk1_up)

            with torch.no_grad():
                k0_left = (1 / kalpha0) * (k0 - oupk0)  ## feature reverse
            gk_up = k.grad  ## Fusion
            gd0_left = d0.grad
            gd1_left = d1.grad
            gd2_left = d2.grad
            gk1_left = gk1_left + k1_left.grad if k1_left.grad is not None else gk1_left  ## Fusion
            k0_left.requires_grad = False
            kout0 = k0_left * kalpha0  ##alpha0 update
            torch.autograd.backward(kout0, gk0_up)

        if ctx.first_col:
            return None, None, gk_up, gk0_left, gk1_left, None, gd0_left, gd1_left, gd2_left
        else:
            return None, None, gk_up, gk0_left, gk1_left, gk2_left, gd0_left, gd1_left, gd2_left
class ReverseFunctionKernelAttn5level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 9
        [x, c0, c1, c2, c3, c4, k0, k1, k2] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices

            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1, k0) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2, k1) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3, k2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4, k0, k1, k2)
        return x, c0, c1, c2, c3, c4, k0, k1, k2

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4, k0, k1, k2 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right, gk0_right, gk1_right, gk2_right = grad_outputs
        (x, c0, c1, c2, c3, c4, k0, k1, k2) = detach_and_grad((x, c0, c1, c2, c3, c4, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_up = g3_right + c3.grad
            g3_left = g3_up * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left

            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left, k2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left, k1)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left, k0)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
            gk0_left = k0.grad # + gk0_right
            gk1_left = k1.grad # + gk1_right
            gk2_left = k2.grad # + gk2_right
        if ctx.first_col:
            return None, None, gx_up, None, None, None, None, None, gk0_left, gk1_left, gk2_left
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, g4_left, gk0_left, gk1_left, gk2_left
class ReverseFunctionDecKernelAttn5level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 9
        [x, c0, c1, c2, c3, c4, k0, k1, k2] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices

            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3, k2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4, k1) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None, k0) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4, k0, k1, k2)
        return x, c0, c1, c2, c3, c4, k0, k1, k2

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4, k0, k1, k2 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right, gk0_right, gk1_right, gk2_right = grad_outputs
        (x, c0, c1, c2, c3, c4, k0, k1, k2) = detach_and_grad((x, c0, c1, c2, c3, c4, k0, k1, k2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None, k0)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_up = g3_right + c3.grad
            g3_left = g3_up * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left, k1)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left

            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left, k2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
            gk0_left = k0.grad # + gk0_right
            gk1_left = k1.grad # + gk1_right
            gk2_left = k2.grad # + gk2_right
        if ctx.first_col:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, None, gk0_left, gk1_left, gk2_left
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, g4_left, gk0_left, gk1_left, gk2_left


class ReverseFunctionDyDecKernelAttn5level(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 10
        [x, c0, c1, c2, c3, c4, k0, k1, k2, exit_index] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices

            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1, exit_index) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2, exit_index) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3, k2, exit_index) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4, k1, exit_index) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None, k0, exit_index) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4, k0, k1, k2, exit_index)
        return x, c0, c1, c2, c3, c4, k0, k1, k2, exit_index

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4, k0, k1, k2, exit_index = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right, gk0_right, gk1_right, gk2_right, gexit_index_right = grad_outputs
        (x, c0, c1, c2, c3, c4, k0, k1, k2, exit_index) = detach_and_grad((x, c0, c1, c2, c3, c4, k0, k1, k2, exit_index))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None, k0, exit_index)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_up = g3_right + c3.grad
            g3_left = g3_up * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left, k1, exit_index)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left

            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left, k2, exit_index)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left, exit_index)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left, exit_index)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
            gk0_left = k0.grad # + gk0_right
            gk1_left = k1.grad # + gk1_right
            gk2_left = k2.grad # + gk2_right
            gexit_index_left = exit_index.grad
        if ctx.first_col:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, None, gk0_left, gk1_left, gk2_left, gexit_index_left
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, g4_left, gk0_left, gk1_left, gk2_left, gexit_index_left
class ReverseFunctionDecKernelAttn5levelXF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 12
        [x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices

            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3, k2, kf2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4, k1, kf1) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None, k0, kf0) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2)
        return x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right, gk0_right, gk1_right, gk2_right, gkf0_right, gkf1_right, gkf2_right = grad_outputs
        (x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2) = detach_and_grad((x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None, k0, kf0)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_up = g3_right + c3.grad
            g3_left = g3_up * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left, k1, kf1)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left

            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left, k2, kf2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
            gk0_left = k0.grad # + gk0_right
            gk1_left = k1.grad # + gk1_right
            gk2_left = k2.grad # + gk2_right
            gkf0_left = kf0.grad  # + gk0_right
            gkf1_left = kf1.grad  # + gk1_right
            gkf2_left = kf2.grad  # + gk2_right
        if ctx.first_col:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, None, gk0_left, gk1_left, gk2_left, gkf0_left, gkf1_left, gkf2_left
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, g4_left, gk0_left, gk1_left, gk2_left, gkf0_left, gkf1_left, gkf2_left
class ReverseFunctionDecKernelAttn5level0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 7
        [x, c0, c1, c2, c3, c4, k0] = args
        if type(c4) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices

            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None, k0) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4, k0)
        return x, c0, c1, c2, c3, c4, k0

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4, k0 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right, gk0_right = grad_outputs
        (x, c0, c1, c2, c3, c4, k0) = detach_and_grad((x, c0, c1, c2, c3, c4, k0))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None, k0)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_up = g3_right + c3.grad
            g3_left = g3_up * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left

            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
            gk0_left = k0.grad # + gk0_right
        if ctx.first_col:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, None, gk0_left
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, g4_left, gk0_left
class ReverseFunctionKernelAttn5levelXF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3, l4 = run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 12
        [x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices

            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1, k0, kf0) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2, k1, kf1) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3, k2, kf2) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, c4) + c3 * alpha3
            ctx.cpu_states_4, ctx.gpu_states_4 = get_cpu_and_gpu_states(gpu_devices)
            c4 = l4(c3, None) + c4 * alpha4
        ctx.save_for_backward(x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2)
        return x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2 = ctx.saved_tensors
        l0, l1, l2, l3, l4 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3, alpha4 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right, g4_right, gk0_right, gk1_right, gk2_right, gkf0_right, gkf1_right, gkf2_right = grad_outputs
        (x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2) = detach_and_grad((x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2))
        # (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))
        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g4_down = g4_right
            g4_left = g4_down * alpha4 ## shortcut
            set_device_states(ctx.cpu_states_4, ctx.gpu_devices, ctx.gpu_states_4)
            oup4 = l4(c3, None)
            torch.autograd.backward(oup4, g4_down, retain_graph=True)
            with torch.no_grad():
                c4_left = (1 / alpha4) * (c4 - oup4)  ## feature reverse
            # (c4_left,) = detach_and_grad((c4_left,))
            # c4_left.requires_grad = False

            g3_up = g3_right + c3.grad
            g3_left = g3_up * alpha3  ##shortcut

            (c4_left,) = detach_and_grad((c4_left,))
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)

            oup3 = l3(c2, c4_left)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            c4_left.requires_grad = False
            cout4 = c4_left * alpha4  ## alpha3 update
            torch.autograd.backward(cout4, g4_down)

            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g4_left = g4_left + c4_left.grad if c4_left.grad is not None else g4_left

            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left, k2, kf2)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left, k1, kf1)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left, k0, kf0)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
            gk0_left = k0.grad # + gk0_right
            gk1_left = k1.grad # + gk1_right
            gk2_left = k2.grad # + gk2_right
            gkf0_left = kf0.grad  # + gk0_right
            gkf1_left = kf1.grad  # + gk1_right
            gkf2_left = kf2.grad  # + gk2_right
        if ctx.first_col:
            return None, None, gx_up, None, None, None, None, None, gk0_left, gk1_left, gk2_left, gkf0_left, gkf1_left, gkf2_left
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left, g4_left, gk0_left, gk1_left, gk2_left, gkf0_left, gkf1_left, gkf2_left
class DecoderReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            g3_up = g3_right
            g3_left = g3_up * alpha3  ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)  ## feature reverse
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left
class DecoderReverseFunctionOld(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, ups, alpha, *args):
        l0, l1, l2, l3 = run_functions
        up3, up2, up1 = ups
        alpha0, alpha1, alpha2, alphax = alpha
        ctx.run_functions = run_functions
        ctx.ups = ups
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 4
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(gpu_devices)
            c2 = up3(l3(c3)) + c2 * alpha2
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(gpu_devices)
            c1 = up2(l2(c2)) + c1 * alpha1
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(gpu_devices)
            c0 = up1(l1(c1))  + c0 * alpha0
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(gpu_devices)
            x = up1(l1(c0)) + x * alphax
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        up1, up2, up3 = ctx.ups
        alpha0, alpha1, alpha2, alphax = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
                torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

            c3_left = c3

            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = up1(l1(c0))
            torch.autograd.backward(oup0, gx_right, retain_graph=True)
            with torch.no_grad():
                x_left = (1 / alphax) * (x - oup0)  ## feature reverse

            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2  ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3  ##alpha3 update
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)  ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1  ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2  ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)  ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0  ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left  ## Fusion

            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1  ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)  ## feature reverse
            gx_up = x.grad  ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left  ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0  ##alpha0 update
            torch.autograd.backward(cout0, g0_up)

            g3_left = g3_right
        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left
class ReverseFunctionforHuge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions  = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0  = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1, c3) + c0*alpha0
            ctx.cpu_states_1, ctx.gpu_states_1  = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1*alpha1
            ctx.cpu_states_2, ctx.gpu_states_2  = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2*alpha2
            ctx.cpu_states_3, ctx.gpu_states_3  = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2) + c3*alpha3
            
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1 ,c2, c3
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with torch.enable_grad(), \
            torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
            torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
            torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            
            g3_up = g3_right
            g3_left = g3_up*alpha3 ##shortcut
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)                    
            oup3 = l3(c2)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1/alpha3)*(c3 - oup3) ## feature reverse
            g2_up = g2_right+ c2.grad
            g2_left = g2_up*alpha2 ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)          
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left*alpha3 ##alpha3 update
            torch.autograd.backward(cout3, g3_up)
            
            with torch.no_grad():
                c2_left = (1/alpha2)*(c2 - oup2) ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right+c1.grad
            g1_left = g1_up*alpha1 ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)     
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left*alpha2 ##alpha3 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1/alpha1)*(c1 - oup1) ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up*alpha0 ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left ## Fusion
            
            (c1_left,c3_left) = detach_and_grad((c1_left,c3_left))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)     
            oup0 = l0(x, c1_left, c3_left)            
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left*alpha1 ##alpha3 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1/alpha0)*(c0 - oup0) ## feature reverse
            gx_up = x.grad ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left ## Fusion
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left*alpha0 ##alpha3 update
            torch.autograd.backward(cout0, g0_up)
        
        if ctx.first_col:
            # print(f'c0: {c0_left.max()}, c1: {c1_left.max()}, c2: {c2_left.max()}, c3: {c3_left.max()}')
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left
