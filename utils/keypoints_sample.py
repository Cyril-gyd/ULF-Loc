import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from gaussian_renderer import GaussianModel
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import get_render_visible_mask
import os
from utils.graphics_utils import  fov2focal
import faiss
import pickle
import time

import faiss

def fast_random_knn_score(points, npoints, score, k=32):
    points_np = points.cpu().detach().numpy()
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(points_np)
    sampled_idx = np.random.choice(points.shape[0], npoints, replace=False)
    sampled_points = points_np[sampled_idx]
    
    _, knn_idx = index.search(sampled_points, k)
    knn_idx = torch.from_numpy(knn_idx).cuda()
    
    knn_score = score[knn_idx]
    score_knn_sort_idx = torch.argsort(knn_score, descending=True, dim=-1)
    
    final_sampled_idx = set()
    for i in range(npoints):
        for j in score_knn_sort_idx[i]:
            idx = knn_idx[i, j].item()
            if idx not in final_sampled_idx:
                final_sampled_idx.add(idx)
                break
    
    return torch.tensor(list(final_sampled_idx)).cuda()

def keypoints_vote(feature_extractor : FeatureExtractor, scene, gaussians, masks, output_path, config):
    if os.path.exists(os.path.join(output_path, config["sample"]["landmark_file_name"])):
        print("Loading sampled idx")
        sampled_idx = pickle.load(
            open(
                os.path.join(output_path, config["sample"]["landmark_file_name"]),
                "rb",
            )
        )
        return sampled_idx
        
    viewpoint_stack = scene.getTrainCameras()
    view_num = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.int, device="cuda")
    vote_sum = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.int16, device="cuda")
    

    index = faiss.IndexFlatL2(2)  
    start = time.time()
    for _, viewpoint_cam in enumerate(tqdm(viewpoint_stack, desc="K.C. Landmark Sampling")):

        render_visible_mask = get_render_visible_mask(
            gaussians,
            viewpoint_cam,
            viewpoint_cam.original_image.shape[2],
            viewpoint_cam.original_image.shape[1],
        )

        view_img = viewpoint_cam.original_image.cuda()
        
        if masks is not None:
            with torch.no_grad():
                obj_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
                sky_mask = masks[viewpoint_cam.image_name][1].cuda()[None]
                distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]

                # mask obj and distort border
                mask = obj_mask & distort_mask

                view_img = view_img * mask
                # mask sky
                view_img[sky_mask.repeat(3, 1, 1) == False] = 0
                
        gt_keypoints, gt_score, gt_feature = feature_extractor.detectAndCompute(view_img[None], 
                                                                 top_k= config["sample"]["2D_kpts_num"])[0].values()


        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda() 
        focalX = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.original_image.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, viewpoint_cam.original_image.shape[1])
        K = torch.tensor(
            [
                [focalX, 0.0, viewpoint_cam.original_image.shape[2] / 2],
                [0.0, focalY, viewpoint_cam.original_image.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        xyz = gaussians.get_xyz

        # project gaussians to image space
        xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=-1)
        xyz_cam = (viewmat @ xyz_homo.T)[:3]
        depths = xyz_cam[2]
        xyz_cam_homo = xyz_cam / depths

        xy = (K @ xyz_cam_homo)[:2].long()

        in_mask = (
            (xy[0] >= 0)
            & (xy[0] < viewpoint_cam.original_image.shape[2])
            & (xy[1] >= 0)
            & (xy[1] < viewpoint_cam.original_image.shape[1])
        )

        if render_visible_mask is not None:
            visible_mask = in_mask & render_visible_mask
        else:
            visible_mask = in_mask
            
        visible_indices = torch.where(visible_mask)[0]

        xy = xy[:, visible_mask]
        depths = depths[visible_mask]

        view_num[visible_mask] += 1
        

        xy_ = xy.transpose(1, 0)
        xy_np = xy_.cpu().numpy().astype('float32')
        gt_np = gt_keypoints.cpu().numpy().astype('float32')
        index.reset()
        index.add(gt_np)
        dists, _ = index.search(xy_np, k=1)  
        mask = (dists <= config["sample"]["thre_dis"]).flatten()  
        mask = torch.from_numpy(mask).to(xy_.device)
        
        vote_sum[visible_indices[mask]] += 1 

    k = config["sample"]["k"]
    sampled_idx= fast_random_knn_score(gaussians.get_xyz, config["sample"]["3D_kpts_num"], vote_sum, k)
    sampled_idx = torch.unique(sampled_idx)
    print(f"Sample 3D keypoints time:{time.time() - start:.2f} s")
    idx_path = os.path.join(output_path, config["sample"]["landmark_file_name"])
    print("Save sampled idx to:", idx_path)
    pickle.dump(sampled_idx, open(idx_path, "wb"))
    
    return sampled_idx