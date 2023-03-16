"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th

# import torch.distributed as dist
from QDrop import quant

from improved_diffusion import logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from QDrop.quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
    set_act_quantize_params,
)
from improved_diffusion.image_datasets import load_data
from QDrop.data.imagenet import build_imagenet_data
import torch
import torch.nn as nn


def generate_t(args, t_mode, num_samples, diffusion, device):
    if t_mode == "1":
        t = torch.tensor([1] * num_samples, device=device)  # TODO timestep gen
    elif t_mode == "-1":
        t = torch.tensor(
            [diffusion.num_timesteps - 1] * num_samples, device=device
        )  # TODO timestep gen
    elif t_mode == "mean":
        t = torch.tensor(
            [diffusion.num_timesteps // 2] * num_samples, device=device
        )  # TODO timestep gen
    elif t_mode == "manual":
        t = torch.tensor(
            [diffusion.num_timesteps * 0.1] * num_samples, device=device
        )  # TODO timestep gen
    elif t_mode == "normal":
        shape = torch.Tensor(num_samples)
        normal_val = torch.nn.init.normal_(shape, mean=args.calib_t_mode_normal_mean, std=args.calib_t_mode_normal_std)*diffusion.num_timesteps
        t = normal_val.clone().type(torch.int).to(device=device)
        # print(t.shape)
        # print(t[0:30])
    elif t_mode == "random":
        # t = torch.randint(0, diffusion.num_timesteps, [num_samples], device=device)
        t = torch.randint(0, int(diffusion.num_timesteps*0.8), [num_samples], device=device)
        print(t.shape)
        print(t)
    elif t_mode == "uniform":
        t = torch.linspace(
            0, diffusion.num_timesteps, num_samples, device=device
        ).round()
    else:
        raise NotImplementedError
    return t.clamp(0, diffusion.num_timesteps - 1)



def random_calib_data_generator(
    shape, num_samples, device, t_mode, diffusion, class_cond=True
):
    calib_data = []
    for batch in range(num_samples):
        img = torch.randn(*shape, device=device)
        calib_data.append(img)
    t = generate_t(t_mode, num_samples, diffusion, device)
    t = diffusion._scale_timesteps(t)
    if class_cond:
        cls = torch.tensor([1] * num_samples, device=device).long()  # TODO class gen
        return torch.cat(calib_data, dim=0), t, cls
    else:
        return torch.cat(calib_data, dim=0), t


def raw_calib_data_generator(
    args, num_samples, device, t_mode, diffusion, class_cond=True
):
    loader = load_data(
        data_dir=args.data_dir,
        batch_size=num_samples,
        image_size=args.image_size,
        class_cond=class_cond,
    )
    calib_data, cls = next(loader)
    calib_data = calib_data.to(device)
    t = generate_t(t_mode, num_samples, diffusion, device)
    t = diffusion._scale_timesteps(t)
    if class_cond:
        return calib_data, t, cls.to(device)
    else:
        return calib_data, t


def forward_t_calib_data_generator(
    args, num_samples, device, t_mode, diffusion, class_cond=True
):
    loader = load_data(
        data_dir=args.data_dir,
        batch_size=num_samples,
        image_size=args.image_size,
        class_cond=class_cond,
    )
    calib_data, cls = next(loader)
    calib_data = calib_data.to(device)
    t = generate_t(t_mode, num_samples, diffusion, device).long()
    x_t = diffusion.q_sample(calib_data, t)
    t = diffusion._scale_timesteps(t)
    if class_cond:
        return x_t, t, cls.to(device)
    else:
        return x_t, t


def backward_t_calib_data_generator(
    model, args, num_samples, device, t_mode, diffusion, class_cond=True
):
    model_kwargs = {}
    if class_cond:
        cls = torch.tensor([1] * num_samples, device=device).long()  # TODO class gen
        model_kwargs["y"] = cls
    loop_fn = (
        diffusion.ddim_sample_loop_progressive
        if args.use_ddim
        else diffusion.p_sample_loop_progressive
    )
    t = generate_t(args, t_mode, num_samples, diffusion, device).long()
    calib_data = None
    for now_rt, sample_t in enumerate(
        loop_fn(
            model,
            (num_samples, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
        )
    ):
        sample_t = sample_t["sample"]
        if calib_data is None:
            calib_data = torch.zeros_like(sample_t)
        mask = t == now_rt
        if mask.any():
            calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
    calib_data = calib_data.to(device)
    t = diffusion._scale_timesteps(t)
    if class_cond:
        return calib_data, t, cls.to(device)
    else:
        return calib_data, t


def quant_model(args, model, diffusion):
    # build quantization parameters
    wq_params = {
        "n_bits": args.n_bits_w,
        "channel_wise": args.channel_wise,
        "scale_method": args.init_wmode,
        "symmetric": True,
    }
    aq_params = {
        "n_bits": args.n_bits_a,
        "channel_wise": False,
        "scale_method": args.init_amode,
        "leaf_param": True,
        "prob": args.prob,
        "symmetric": True,
    }

    qnn = QuantModel(
        model=model, weight_quant_params=wq_params, act_quant_params=aq_params
    )
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print("Setting the first and the last layer to 8-bit")
        qnn.set_first_last_layer_to_8bit()
    # # if args.mixup_quant_on_cosine:
    # print("Setting the cosine embedding layer to 32-bit")
    # qnn.set_cosine_embedding_layer_to_32bit()

    qnn.disable_network_output_quantization()
    # print("check the model!")
    # print(qnn)
    print("sampling calib data")
    if args.calib_im_mode == "random":
        cali_data = random_calib_data_generator(
            [1, 3, args.image_size, args.image_size],
            args.calib_num_samples,
            "cuda",
            args.calib_t_mode,
            diffusion,
            args.class_cond,
        )
    elif args.calib_im_mode == "raw":
        cali_data = raw_calib_data_generator(
            args,
            args.calib_num_samples,
            "cuda",
            args.calib_t_mode,
            diffusion,
            args.class_cond,
        )
    elif args.calib_im_mode == "raw_forward_t":
        cali_data = forward_t_calib_data_generator(
            args,
            args.calib_num_samples,
            "cuda",
            args.calib_t_mode,
            diffusion,
            args.class_cond,
        )
    elif args.calib_im_mode == "noise_backward_t":
        cali_data = backward_t_calib_data_generator(
            model,
            args,
            args.calib_num_samples,
            "cuda",
            args.calib_t_mode,
            diffusion,
            args.class_cond,
        )
    else:
        raise NotImplementedError
    # print('the quantized model is below!')
    # Kwargs for weight rounding calibration
    assert args.wwq is True
    kwargs = dict(
        cali_data=cali_data,
        iters=args.iters_w,
        weight=args.weight,
        b_range=(args.b_start, args.b_end),
        warmup=args.warmup,
        opt_mode="mse",
        wwq=args.wwq,
        waq=args.waq,
        order=args.order,
        act_quant=args.act_quant,
        lr=args.lr,
        input_prob=args.input_prob,
        keep_gpu=not args.keep_cpu,
    )

    if args.act_quant and args.order == "before" and args.awq is False:
        """Case 2"""
        set_act_quantize_params(
            qnn, cali_data=cali_data, awq=args.awq, order=args.order
        )

    """init weight quantizer"""
    set_weight_quantize_params(qnn)
    if not args.use_adaround:
        print('setting')
        # cali_data = cali_data.detach()
        set_act_quantize_params(
            qnn, cali_data=cali_data, awq=args.awq, order=args.order
        )
        print('setting1111111')
        qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
        return qnn
    else:
        def set_weight_act_quantize_params(module):
            if isinstance(module, QuantModule):
                layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                block_reconstruction(qnn, module, **kwargs)
            else:
                raise NotImplementedError

        def recon_model(model: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, module in model.named_children():
                if isinstance(module, QuantModule):
                    print("Reconstruction for layer {}".format(name))
                    set_weight_act_quantize_params(module)
                elif isinstance(module, BaseQuantBlock):
                    print("Reconstruction for block {}".format(name))
                    set_weight_act_quantize_params(module)
                else:
                    recon_model(module)

        # Start calibration
        recon_model(qnn)

        if args.act_quant and args.order == "after" and args.waq is False:
            """Case 1"""
            set_act_quantize_params(
                qnn, cali_data=cali_data, awq=args.awq, order=args.order
            )

        qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
        return qnn


def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.cuda()
    model.eval()

    model = quant_model(args, model, diffusion)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device="cuda"
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        gathered_samples = sample.unsqueeze(0)
        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = classes.unsqueeze(0)
            # gathered_labels = [
            #     th.zeros_like(classes) for _ in range(dist.get_world_size())
            # ]
            # dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
        # if dist.get_rank() == 0:
        if 1:
            shape_str = "x".join([str(x) for x in arr.shape[2:]])
            if args.out_path == "":
                # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
                out_path = f"outputs/samples_{args.calib_im_mode}_{args.calib_t_mode}_{shape_str}_{args.calib_num_samples}.npz"
            else:
                out_path = args.out_path
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--data_dir", type=str, help="ImageNet dir")

    parser.add_argument(
        "--seed", default=3, type=int, help="random seed for results reproduction"
    )

    # quantization parameters
    parser.add_argument(
        "--n_bits_w", default=4, type=int, help="bitwidth for weight quantization"
    )
    parser.add_argument(
        "--channel_wise",
        action="store_true",
        help="apply channel_wise quantization for weights",
    )
    parser.add_argument(
        "--n_bits_a", default=4, type=int, help="bitwidth for activation quantization"
    )
    parser.add_argument(
        "--act_quant", action="store_true", help="apply activation quantization"
    )
    parser.add_argument("--disable_8bit_head_stem", action="store_true")

    # weight calibration parameters
    parser.add_argument(
        "--calib_num_samples",
        default=1024,
        type=int,
        help="size of the calibration dataset",
    )
    parser.add_argument(
        "--iters_w", default=20000, type=int, help="number of iteration for adaround"
    )
    parser.add_argument(
        "--weight",
        default=0.01,
        type=float,
        help="weight of rounding cost vs the reconstruction loss.",
    )
    parser.add_argument(
        "--keep_cpu", action="store_true", help="keep the calibration data on cpu"
    )

    parser.add_argument(
        "--wwq",
        action="store_true",
        help="weight_quant for input in weight reconstruction",
    )
    parser.add_argument(
        "--waq",
        action="store_true",
        help="act_quant for input in weight reconstruction",
    )

    parser.add_argument(
        "--b_start",
        default=20,
        type=int,
        help="temperature at the beginning of calibration",
    )
    parser.add_argument(
        "--b_end", default=2, type=int, help="temperature at the end of calibration"
    )
    parser.add_argument(
        "--warmup",
        default=0.2,
        type=float,
        help="in the warmup period no regularization is applied",
    )

    # activation calibration parameters
    parser.add_argument("--lr", default=4e-5, type=float, help="learning rate for LSQ")

    parser.add_argument(
        "--awq",
        action="store_true",
        help="weight_quant for input in activation reconstruction",
    )
    parser.add_argument(
        "--aaq",
        action="store_true",
        help="act_quant for input in activation reconstruction",
    )

    parser.add_argument(
        "--init_wmode",
        default="mse",
        type=str,
        choices=["minmax", "mse", "minmax_scale"],
        help="init opt mode for weight",
    )
    parser.add_argument(
        "--init_amode",
        default="mse",
        type=str,
        choices=["minmax", "mse", "minmax_scale"],
        help="init opt mode for activation",
    )
    # order parameters
    parser.add_argument(
        "--order",
        default="before",
        type=str,
        choices=["before", "after", "together"],
        help="order about activation compare to weight",
    )
    parser.add_argument("--prob", default=1.0, type=float)
    parser.add_argument("--input_prob", default=1.0, type=float)
    parser.add_argument("--use_adaround", action="store_true")
    parser.add_argument(
        "--calib_im_mode",
        default="random",
        type=str,
        choices=["random", "raw", "raw_forward_t", "noise_backward_t"],
    )
    parser.add_argument(
        "--calib_t_mode",
        default="random",
        type=str,
        choices=["random", "1", "-1", "mean", "uniform" , 'manual' ,'normal' ,'poisson'],
    )
    parser.add_argument(
        "--calib_t_mode_normal_mean",
        default=0.5,
        type=float,
        help='for adjusting the weights in the normal distribution'
    )
    parser.add_argument(
        "--calib_t_mode_normal_std",
        default=0.35,
        type=float,
        help='for adjusting the weights in the normal distribution'
    )
    parser.add_argument("--out_path", default="", type=str)
    return parser


if __name__ == "__main__":
    main()
