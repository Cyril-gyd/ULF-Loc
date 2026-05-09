
import os
import sys
import uuid
import yaml
import json
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import pickle
from tqdm import tqdm
from collections import defaultdict

from arguments import ModelParams, OptimizationParams
from gaussian_renderer import render_gsplat
from scene import Scene
from utils.general_utils import safe_state, seed_everything
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.time_utils import _Timer, _GpuMem, flush_stats

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F

from encoders.feature_extractor import FeatureExtractor

from utils.keypoints_sample import keypoints_vote
from utils.gsfeature_fusion import feature_fusion

os.environ['CUDA_VISIBLE_DEVICES']="0" # GPU set


timer_stats = defaultdict(float)  
mem_stat = defaultdict(float)  
timer = lambda name, stat: _Timer(name, stat)
gpu_mem   = lambda name, stat: _GpuMem(name, stat)


def training(
    dataset,
    opt,
    testing_iterations,
    saving_iterations,
    checkpoint,
    sample_kpts=True,
    config = None
):
    print(opt)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    print("Feature type:", dataset.feature_type)
    print("Gaussian type:", dataset.gaussian_type)
    if dataset.gaussian_type == "3dgs":
        from scene.gaussian_model import GaussianModel

        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        from scene.gaussian_model import GaussianModel_2dgs

        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError("Gaussian type not supported")

    scene = Scene(dataset, gaussians)

    # load masks
    masks = None
    if os.path.exists(os.path.join(dataset.source_path, dataset.images, "masks.pkl")):
        print(
            "Loading masks from",
            os.path.join(dataset.source_path, dataset.images, "masks.pkl"),
        )
        masks = pickle.load(
            open(os.path.join(dataset.source_path, dataset.images, "masks.pkl"), "rb")
        )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="train Gaussian")
    first_iter += 1

    # Training loop
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        render_pkg = render_gsplat(
            viewpoint_cam,
            gaussians,
            background,
            rgb_only=True,
            norm_feat_bf_render=dataset.norm_before_render,
            longest_edge=dataset.longest_edge,
            rasterize_mode="antialiased",
        )

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        # make sure ground truth image is the same size as the rendered image
        original_image = viewpoint_cam.original_image.cuda()
        gt_image = F.interpolate(
            original_image.unsqueeze(0),
            size=(image.shape[1], image.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if masks is not None:
            # use mask
            obj_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            sky_mask = masks[viewpoint_cam.image_name][1].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]

            # mask obj and distort border
            mask = obj_mask & distort_mask

            image = image * mask
            gt_image = gt_image * mask
            # mask sky
            gt_image[sky_mask.repeat(3, 1, 1) == False] = 1  # 全白

        Ll1 = l1_loss(image, gt_image)

        # overall loss
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        )

        # regularization for 2DGS
        if dataset.gaussian_type == "2dgs":
            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            rend_alpha = render_pkg['rend_alpha']
            # ------------------------------
            surf_normal *= rend_alpha.squeeze(0).detach()
            rend_normal = rend_normal.squeeze(0).permute(2, 0, 1)
            if len(surf_normal.shape) == 4:
                surf_normal = surf_normal.squeeze(0)
            surf_normal = surf_normal.permute(2, 0, 1)
            # ------------------------------
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            if masks is not None:
                normal_error *= mask
                rend_dist = rend_dist.squeeze(-1)
                rend_dist *= mask
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            loss = loss + dist_loss + normal_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                background,
                dataset,
                masks=masks,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                visibility_filter = visibility_filter[:,0]
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter, 0]
                )
                gaussians.add_densification_stats_gsplat(
                    viewspace_point_tensor,
                    visibility_filter,
                    image.shape[2],
                    image.shape[1],
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    if sample_kpts:
        feature_extractor = FeatureExtractor(config["feature_type"]).cuda().eval()
        output_path = os.path.join(dataset.model_path, config["log_name"])
        os.makedirs(output_path, exist_ok=True)  
           
        sampled_idx = keypoints_vote(feature_extractor, 
                                    scene, 
                                    gaussians, 
                                    masks, 
                                    output_path, 
                                    config
                                    )
        
        feature_fusion(feature_extractor,  
                    scene, 
                    gaussians, 
                    sampled_idx, 
                    masks, 
                    output_path, 
                    config
                    )



def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    background,
    dataset,
    masks=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = render_gsplat(
                        viewpoint,
                        scene.gaussians,
                        background,
                        rgb_only=True,
                        norm_feat_bf_render=dataset.norm_before_render,
                        longest_edge=dataset.longest_edge,
                        rasterize_mode="antialiased",
                    )

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)

                    original_image = viewpoint.original_image.cuda()

                    gt_image = F.interpolate(
                        original_image.unsqueeze(0),
                        size=(image.shape[1], image.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)


                    if masks is not None:
                        mask = masks[viewpoint.image_name][0].cuda()[None]
                        sky_mask = masks[viewpoint.image_name][1].cuda()[None]
                        distort_mask = masks[viewpoint.image_name][2].cuda()[None]
                        mask = mask & distort_mask

                        image_loss = image * mask
                        gt_image_loss = gt_image * mask
                        # mask sky
                        gt_image_loss[sky_mask.repeat(3, 1, 1) == False] = 1 
                    else:
                        image_loss = image
                        gt_image_loss = gt_image

                    l1_test += l1_loss(image_loss, gt_image_loss).mean().double()
                    psnr_test += psnr(image_loss, gt_image_loss).mean().double()

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                            if masks is not None:
                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}/mask".format(viewpoint.image_name),
                                    mask[None],
                                    global_step=iteration,
                                )
                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}/sky_mask".format(viewpoint.image_name),
                                    sky_mask[None],
                                    global_step=iteration,
                                )

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} ".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )


        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    seed_everything(2025)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[]) ##########
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--sample_kpts", action="store_true", default=False)
    parser.add_argument("--cfg", default=None, type=str)

    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    
    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.start_checkpoint,
        args.sample_kpts,
        config
    )

    # All done
    print("\nTraining complete.")

