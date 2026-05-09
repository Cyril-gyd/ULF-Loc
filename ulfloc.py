import datetime
import json
import os
import pickle
from argparse import ArgumentParser
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import cv2
from arguments import ModelParams, PipelineParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_from_pose_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from utils.graphics_utils import fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.pose_utils import cal_pose_error, solve_pose
from utils.LGCV import *
from utils.loc_utils import *
import logging

os.environ['CUDA_VISIBLE_DEVICES']="0" # GPU set

def get_intrinsic(fovx, fovy, width, height):
    focalX = fov2focal(fovx, width)
    focalY = fov2focal(fovy, height)
    K = np.array(
        [
            [focalX, 0.0, width / 2],
            [0.0, focalY, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K

def get_sampled_gaussians(gaussians: GaussianModel, idx_sampled, feature_dim):
    sampled_gaussians = GaussianModel(gaussians.max_sh_degree)
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    
    if hasattr(gaussians, '_loc_feature') and gaussians._loc_feature is not None:
        sampled_gaussians._loc_feature = gaussians._loc_feature[idx_sampled]
    else:
        num_sampled = len(idx_sampled)
        device = sampled_gaussians._xyz.device
        sampled_gaussians._loc_feature = torch.zeros((num_sampled, feature_dim), dtype=torch.float32, device=device)
    return sampled_gaussians

class ULFLoc:
    def __init__(self, scene, gaussians, masks, output_path, config):
        self.gaussians = gaussians
        self.config = config
        self.output_path = output_path
        
        self.feature_extractor = FeatureExtractor(config["feature_type"]).cuda().eval()
        self.sampled_idx = pickle.load(
            open(
                os.path.join(output_path, config["sample"]["landmark_file_name"]),
                "rb",
            )
        )
        
        landmarks = get_sampled_gaussians(
            gaussians, self.sampled_idx, self.feature_extractor.feature_dim
        )
        landmarks._loc_feature = pickle.load(
            open(
                os.path.join(output_path, config["sample"]["kpts_feature_file_name"]),
                "rb",
            )
        )
        
        self.landmarks = landmarks
        self.longest_edge = config["longest_edge"]
        
    def get_feature_map(self, image):
        """
        image: torch.Tensor, shape (3, H, W)
        """
        fine_resolution = get_resolution_from_longest_edge(
            image.shape[-2], image.shape[-1], self.longest_edge
        )
        coarse_resolution = (fine_resolution[0] // 8, fine_resolution[1] // 8)

        # Get feature
        feature_map, _ = self.feature_extractor.detectAndComputeDense(image[None])  # 1, C, H, W

        coarse_feature_map = F.interpolate(
            feature_map, size=coarse_resolution, mode="bilinear", align_corners=False
        )[0]
        coarse_feature_map = F.normalize(coarse_feature_map, p=2, dim=0)
        fine_feature_map = F.interpolate(
            feature_map, size=fine_resolution, mode="bilinear", align_corners=False
        )[0]
        fine_feature_map = F.normalize(fine_feature_map, p=2, dim=0)

        return fine_feature_map, coarse_feature_map
    
    @torch.no_grad()
    def localize(self, camera, query_img, fovx, fovy):

        query_fine_feature_map, query_coarse_feature_map = self.get_feature_map(
            query_img
        )

        sparse_result = self.loc_coarse(camera, query_img, fovx, fovy)

        pose_w2c = sparse_result["pose_w2c"]
        dense_results = []
        for iter in range(self.config["dense"]["iters"]):
            dense_result = self.loc_fine(
                iter, camera.image_name, query_img, query_coarse_feature_map, query_fine_feature_map, pose_w2c, fovx, fovy
            )
            pose_w2c = dense_result["pose_w2c"]
            
            dense_results.append(dense_result)

        return {"sparse": sparse_result, "dense": dense_results}

    @torch.no_grad()
    def loc_coarse(self, camera, query_img, fovx, fovy):

        sampled_keypoints, sampled_score, sampled_features = self.feature_extractor.detectAndCompute(query_img[None], 
                                            top_k=config["sparse"]["kpts_num"])[0].values()
        sampled_features = F.normalize(sampled_features, dim=-1)
        landmark_features = F.normalize(
            self.landmarks._loc_feature.squeeze(), dim=-1
        )

        corr_matrix = torch.matmul(sampled_features, landmark_features.T)

        if config["sparse"]["dual_softmax"] is True:
            corr_matrix = dual_softmax(
                corr_matrix=corr_matrix, temp=config["sparse"]["dual_softmax_temp"]
            )

        if config["sparse"]["mnn_match"] is True:
            b_ids, im_idx, gs_ids = mnn_match(
                corr_matrix[None], thr=self.config["sparse"]["threshold"]
            )
        else:
            im_idx, gs_ids, val = topk_match(
                corr_matrix[None],
                self.config["sparse"]["topk"],
                thr=self.config["sparse"]["threshold"],
            )

        p2d = sampled_keypoints[im_idx, :2].cpu().numpy()
        p3d = self.landmarks.get_xyz[gs_ids].cpu().numpy()
 
        gt_R = camera.R
        gt_t = camera.T
        H, W = query_img.shape[-2:]
        K = get_intrinsic(fovx, fovy, W, H)
        
        projected_2d = cv2.projectPoints(p3d, gt_R.T, gt_t.T, K, None)
        projected_2d = np.array(projected_2d[0]).reshape(-1, 2)
        dis = np.linalg.norm(projected_2d - p2d, axis=1)
        inliers = np.sum(dis < 5.0)
        total_matches = len(p3d)
        inlier_rate = inliers / total_matches

        pose_w2c, inliers = solve_pose(
            p2d + 0.5,
            p3d,
            K,
            self.config["sparse"]["solver"],
            self.config["sparse"]["reprojection_error"],
            self.config["sparse"]["confidence"],
            self.config["sparse"]["max_iterations"],
            self.config["sparse"]["min_iterations"],
        )

        return {
            "pose_w2c": pose_w2c,
            "inliers": inliers.shape[0],
            "inlier_rate": inlier_rate
        }

    @torch.no_grad()
    def loc_fine(
        self, iter, image_name, query_img, coarse_query_feature_map, fine_query_feature_map, pose_w2c, fovx, fovy
    ):

        H_ori, W_ori = query_img.shape[-2:]
        Hf, Wf = fine_query_feature_map.shape[-2:]
        Hc, Wc = coarse_query_feature_map.shape[-2:]
        W = Hf // Hc  # window size
        C = coarse_query_feature_map.shape[0]
        WW = W * W
        overlap_size = 0  
        K = get_intrinsic(fovx, fovy, Wf, Hf)

        render_pkg = render_from_pose_gsplat(
            self.gaussians,
            torch.tensor(pose_w2c, device="cuda"),
            fovx,
            fovy,
            Wf,
            Hf,
            render_mode="RGB+ED",
            rgb_only=True,
            norm_feat_bf_render=self.config["dense"]["norm_before_render"],
            rasterize_mode="antialiased",
        )
        
        fine_rendered_img = render_pkg["render"]
        fine_rendered_img = torch.clamp(fine_rendered_img, 0.0, 1.0)  
        fine_rendered_img = F.interpolate(fine_rendered_img.unsqueeze(0), size=(H_ori, W_ori), mode="bilinear", align_corners=False)[0]
        depth = render_pkg["depth"].squeeze(0)
        
        if (fine_rendered_img == 0).all():
            print("[skip] Rendered img is all zero")
            return {"pose_w2c": pose_w2c, "inliers": 0}
        
        rendered_feature_map, _ = self.feature_extractor.detectAndComputeDense(fine_rendered_img[None])

        coarse_rendered_feature_map = F.interpolate(rendered_feature_map, size=(Hc, Wc), mode='bilinear', align_corners=False)[0]
        fine_rendered_feature_map = F.interpolate(rendered_feature_map, size=(Hf, Wf), mode='bilinear', align_corners=False)[0]
        
        coarse_rendered_feature_map = F.normalize(coarse_rendered_feature_map, dim=0)
        fine_rendered_feature_map = F.normalize(fine_rendered_feature_map, dim=0)

        coarse_corr_matrix = torch.matmul(
            coarse_query_feature_map.permute(1, 2, 0).reshape(1, -1, C),
            coarse_rendered_feature_map.reshape(1, C, -1),
        ) 

        coarse_corr_matrix = dual_softmax(
            coarse_corr_matrix, temp=self.config["dense"]["coarse_dual_softmax_temp"]
        )

        c_b_ids, c_i_ids, c_j_ids = mnn_match(
            coarse_corr_matrix, thr=self.config["dense"]["coarse_threshold"]
        )

        if c_i_ids.dim() == 0:
            print("[skip] Failed in coarse match")
            return {"pose_w2c": pose_w2c, "inliers": 0}
        elif c_i_ids.shape[0] < 3:
            print("[skip] Failed in coarse match")
            return {"pose_w2c": pose_w2c, "inliers": 0}

        if config["dense"]["remove_mismatches"]:
            coarse_query_p2d = torch.stack([
                c_i_ids % Wc, 
                c_i_ids // Wc
            ], dim=1).float() * W  
            
            coarse_rendered_p2d = torch.stack([
                c_j_ids % Wc, 
                c_j_ids // Wc
            ], dim=1).float() * W  

            support_score = compute_geometric_support(
                coarse_query_p2d, 
                coarse_rendered_p2d, 
                self.config["dense"]["k"], 
                angle_thresh_cos=config["dense"]["tau_a"], 
                scale_thresh=config["dense"]["tau_s"]
            )
            
            valid_mask = support_score > self.config["dense"]["geometric_support_thres"]
            if valid_mask.sum() == 0 :
                print("[skip] Not enough geometrically consistent matches after coarse filtering")
                return {"pose_w2c": pose_w2c, "inliers": 0}
            
            c_b_ids = c_b_ids[valid_mask]
            c_i_ids = c_i_ids[valid_mask]
            c_j_ids = c_j_ids[valid_mask]

        query_feature_windows = (
            F.unfold(
                fine_query_feature_map, (W, W), stride=W, padding=overlap_size // 2
            )
            .reshape(1, C, WW, -1)[c_b_ids, :, :, c_i_ids]
            .permute(0, 2, 1)
        )  # B, N, C
        rendered_feature_windows = (
            F.unfold(
                fine_rendered_feature_map, (W, W), stride=W, padding=overlap_size // 2
            )
            .reshape(1, C, WW, -1)[c_b_ids, :, :, c_j_ids]
            .permute(0, 2, 1)
        )  # B, M, C
        
        fine_corr_matrix = torch.matmul(
            query_feature_windows, rendered_feature_windows.transpose(-2, -1)
        )  # B, N, M

        fine_corr_matrix = dual_softmax(
            fine_corr_matrix, temp=self.config["dense"]["fine_dual_softmax_temp"]
        )

        f_b_ids, f_i_ids, f_j_ids = mnn_match(
            fine_corr_matrix, thr=self.config["dense"]["fine_threshold"]
        )

        if f_i_ids.dim() == 0:
            print("[skip] Failed in fine match")
            return {"pose_w2c": pose_w2c, "inliers": 0}
        elif f_i_ids.shape[0] < 3:
            print("[skip] Failed in fine match")
            return {"pose_w2c": pose_w2c, "inliers": 0}

        query_p2d = torch.stack(
            [
                c_i_ids[f_b_ids] % Wc * W + f_i_ids % W,
                c_i_ids[f_b_ids] // Wc * W + f_i_ids // W,
            ],
            dim=1,
        ).float()
        rendered_p2d = torch.stack(
            [
                c_j_ids[f_b_ids] % Wc * W + f_j_ids % W,
                c_j_ids[f_b_ids] // Wc * W + f_j_ids // W,
            ],
            dim=1,
        ).float()

        pose_c2w = np.linalg.inv(pose_w2c)
        p3d = lift_2d_to_3d(rendered_p2d, torch.tensor(K, device="cuda"), torch.tensor(pose_c2w, device="cuda"), depth)

        # Solve pose
        query_p2d = query_p2d.cpu().numpy()
        p3d = p3d.cpu().numpy()

        pose_w2c, inliers = solve_pose(
            query_p2d + 0.5,
            p3d,
            K,
            self.config["dense"]["solver"],
            self.config["dense"]["reprojection_error"],
            self.config["dense"]["confidence"],
            self.config["dense"]["max_iterations"],
            self.config["dense"]["min_iterations"],
        )
        
        
        return {
            "pose_w2c": pose_w2c,
            "inliers": inliers.shape[0],
        }


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--cfg", default=None, type=str)
    parser.add_argument("--prefix", default=None, type=str)
    args = get_combined_args(parser)
    args.eval = True

    dataset = model.extract(args)

    if dataset.gaussian_type == "3dgs":
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError("Gaussian type not supported")

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        preload_cameras=False,
    )

    # Set up config
    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
        
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path

    output_path = os.path.join(dataset.model_path, config["log_name"])
    os.makedirs(output_path, exist_ok=True)                    
    print("Result output path:", output_path)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_path, "output.log"), mode='w'),
            logging.StreamHandler()
        ]
    )
    yaml.dump(config, open(os.path.join(output_path, os.path.basename(args.cfg)), "w"))

    # load masks
    masks = None
    if os.path.exists(os.path.join(dataset.images, "masks.pkl")):
        print(
            "Loading masks from",
            os.path.join(dataset.images, "masks.pkl"),
        )
        masks = pickle.load(
            open(os.path.join(dataset.images, "masks.pkl"), "rb")
        )

    # loc inference
    ulfloc = ULFLoc(scene, gaussians, masks, output_path, config)

    test_cameras = scene.getTestCameras()

    results = []
    sparse_aes = []
    sparse_tes = []
    sparse_inliers = []
    dense_aes = []
    dense_tes = []
    dense_inliers = []

    for idx, camera_info in enumerate(tqdm(test_cameras, desc="ULFLoc")):
        logging.info("Localize image:{}".format( camera_info.image_name))
        gt_w2c = camera_info.world_view_transform.transpose(0, 1).cpu().numpy()
        
        query_image = camera_info.original_image.to("cuda")
        
        if masks is not None:
            with torch.no_grad():
                obj_mask = masks[camera_info.image_name][0].cuda()[None]
                sky_mask = masks[camera_info.image_name][1].cuda()[None]
                distort_mask = masks[camera_info.image_name][2].cuda()[None]
                mask = obj_mask & distort_mask
                query_image = query_image * mask
                query_image[sky_mask.repeat(3, 1, 1) == False] = 0
        
        fovx = camera_info.FoVx
        fovy = camera_info.FoVy

        # localization
        loc_res = ulfloc.localize(camera_info, query_image, fovx, fovy)

        # evaluation
        sparse_ae, sparse_te = cal_pose_error(loc_res["sparse"]["pose_w2c"], gt_w2c)
        sparse_aes.append(sparse_ae)
        sparse_tes.append(sparse_te)
        sparse_inliers.append(loc_res["sparse"]["inliers"])
        loc_res["sparse_AE"] = sparse_ae
        loc_res["sparse_TE"] = sparse_te
        logging.info("sparse: AE: {:.3f}deg, TE: {:.3f}cm".format(sparse_ae, sparse_te))

        dense_ae, dense_te = cal_pose_error(loc_res["dense"][-1]["pose_w2c"], gt_w2c) 
        dense_aes.append(dense_ae)
        dense_tes.append(dense_te)
        dense_inliers.append(loc_res["dense"][-1]["inliers"])
        logging.info("dense: AE: {:.3f}deg, TE: {:.3f}cm".format(dense_ae, dense_te))

        loc_res["gt_pose_w2c"] = gt_w2c.tolist()
        loc_res["dense_AE"] = dense_ae
        loc_res["dense_TE"] = dense_te

        results.append(loc_res)

    # get summary
    sparse_aes = np.array(sparse_aes)
    sparse_tes = np.array(sparse_tes)
    dense_aes = np.array(dense_aes)
    dense_tes = np.array(dense_tes)

    results_summary = {
        "model_path": dataset.model_path,
        "sparse": {
            "median_ae": np.median(sparse_aes),
            "median_te": np.median(sparse_tes),
           "recall_50cm_5d": ((sparse_aes <= 5) & (sparse_tes <= 50)).sum()
            / len(sparse_aes),
           "recall_10cm_5d": ((sparse_aes <= 5) & (sparse_tes <= 10)).sum()
            / len(sparse_aes),
            "recall_5cm_5d": ((sparse_aes <= 5) & (sparse_tes <= 5)).sum()
            / len(sparse_aes),
            "recall_2cm_2d": ((sparse_aes <= 2) & (sparse_tes <= 2)).sum()
            / len(sparse_aes),
            "recall_1cm_1d": ((sparse_aes <= 1) & (sparse_tes <= 1)).sum()
            / len(sparse_aes),            
        },
        "dense": {
            "median_ae": np.median(dense_aes),
            "median_te": np.median(dense_tes),
            "recall_50cm_5d": ((dense_aes <= 5) & (dense_tes <= 50)).sum()
            / len(dense_aes),
            "recall_10cm_5d": ((dense_aes <= 5) & (dense_tes <= 10)).sum()
            / len(dense_aes),
            "recall_5cm_5d": ((dense_aes <= 5) & (dense_tes <= 5)).sum()
            / len(dense_aes),
            "recall_2cm_2d": ((dense_aes <= 2) & (dense_tes <= 2)).sum()
            / len(dense_aes),
            "recall_1cm_1d": ((dense_aes <= 1) & (dense_tes <= 1)).sum()
            / len(dense_aes),
        },
    }

    print("Result Summary:")
    print(json.dumps(results_summary, indent=4))

    json.dump(
        results_summary, open(os.path.join(output_path, "summary.json"), "w"), indent=4
    )

    for item in results:
        item["sparse"]["pose_w2c"] = item["sparse"]["pose_w2c"].tolist()
        for dense_item in item["dense"]:
            dense_item["pose_w2c"] = dense_item["pose_w2c"].tolist()
    json.dump(results, open(os.path.join(output_path, "results.json"), "w"), indent=4)

    print("Result are saved in", output_path)
