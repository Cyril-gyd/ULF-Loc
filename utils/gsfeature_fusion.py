import time
import torch
import torch.nn.functional as F
import os
from utils.graphics_utils import  fov2focal
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import GaussianModel
from tqdm import tqdm
import pickle


def get_sampled_gaussians(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = GaussianModel(gaussians.max_sh_degree)
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    return sampled_gaussians

@torch.no_grad()
def load_image_batch(views, masks, start_idx, end_idx, dtype=torch.float32):
    batch_img = []
    batch_mask = []
    for view in views[start_idx:end_idx]:
        img = view.original_image[0:3, :, :].to(dtype=dtype, device='cuda')  
        batch_img.append(img)
        if masks is not None:
            obj_mask = masks[view.image_name][0].cuda()[None]
            sky_mask = masks[view.image_name][1].cuda()[None]
            distort_mask = masks[view.image_name][2].cuda()[None]

            mask = obj_mask & distort_mask & sky_mask
            batch_mask.append(mask)
    if masks is not None:
        return torch.stack(batch_img) , torch.stack(batch_mask)  # [B,3,H,W]
    else:
        return torch.stack(batch_img), None

def get_cosine_weights(gaussians: GaussianModel, cam_centers, eps=1e-6):

    normal_global = F.normalize(gaussians.get_smallest_axis(), dim=-1, eps=eps)
    gaussian_to_cam = cam_centers.unsqueeze(1) - gaussians._xyz.unsqueeze(0)  # [B, N, 3]
    distance = torch.norm(gaussian_to_cam, dim=-1, keepdim=True)    
    view_dirs = F.normalize(gaussian_to_cam, dim=-1, eps=eps)
    
    cos_theta = (normal_global.unsqueeze(0) * view_dirs).sum(dim=-1).abs().clamp(min=0)  #[B, N]  
    geo_weight = cos_theta.unsqueeze(2) 
    return geo_weight.to(dtype = torch.float32, device = 'cuda')

@torch.no_grad()
def feature_fusion(feature_extractor : FeatureExtractor, scene, gaussians, sample_idx, masks, output_path, config):

    gaussians_sampled = get_sampled_gaussians(gaussians, sample_idx)
    
    views = scene.getTrainCameras()
    R_all = torch.stack([torch.tensor(view.R.T, dtype=torch.float32, device='cuda') for view in views])  # [C, 3, 3]
    t_all = torch.stack([torch.tensor(view.T, dtype=torch.float32, device='cuda') for view in views])  # [C, 3]
    K_all = torch.stack([torch.eye(3, 3, dtype=torch.float32, device='cuda') for view in views]) # [C, 3, 3]
    cam_center_all = torch.stack([view.camera_center.clone().detach().to(dtype=torch.float32, device='cuda') for view in views]) #[C ,3]

    for idx, view in enumerate(views):
        focal_length = fov2focal(view.FoVx, view.image_width)
        K_all[idx, 0, 0] = K_all[idx, 1, 1] = focal_length
        K_all[idx, 0, 2] = view.image_width / 2
        K_all[idx, 1, 2] = view.image_height / 2

    gaussian_pcd = gaussians_sampled.get_xyz.to(dtype = torch.float32, device = 'cuda') #[N, 3]
    N = gaussian_pcd.shape[0]
    V = len(views)
    C = feature_extractor.feature_dim

    image_width = views[0].image_width
    image_height = views[0].image_height

    weighted_sum = torch.zeros((N, C), dtype = torch.float32 ,device='cuda')  # [N, 64]
    sum_weights = torch.zeros((N, 1), dtype = torch.float32, device='cuda')  # [N, 1]

    batch_size = 10
    num_batches = (V + batch_size - 1) // batch_size

    start = time.time()
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Geometry-Weighted Feature Fusion"):

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, V)
            current_size = end_idx - start_idx

            R_batch = R_all[start_idx:end_idx] #[B, 3, 3]
            t_batch = t_all[start_idx:end_idx]
            K_batch = K_all[start_idx:end_idx]
            cam_center_batch = cam_center_all[start_idx:end_idx]
            
            pcd_trans = torch.einsum('cij,nj->cni', R_batch, gaussian_pcd) + t_batch[:, None, :] #  [N,3] -> [B,N,3]
            uv_homogeneous = torch.einsum('cij,cnj->cni', K_batch, pcd_trans) #  [B,N,3] -> [B,N,3]
            proj_batch = uv_homogeneous[..., :2] / uv_homogeneous[..., 2:] #  [B,N,3] -> [B,N,2]
            proj_batch = torch.round(proj_batch).to(torch.int16)

            in_image =  (proj_batch[..., 0] >= 0) & \
                        (proj_batch[..., 0] < image_width) & \
                        (proj_batch[..., 1] >= 0) & \
                        (proj_batch[..., 1] < image_height)
            in_image = in_image.unsqueeze(2) #[B,N,1]

            gt_imgs_batch, masks_batch= load_image_batch(views, masks, start_idx, end_idx, dtype=torch.float32) #[B, 3, H, W]
            
            if masks is not None:
                x_coords = torch.clamp(proj_batch[..., 0], 0, image_width - 1).long()  # [B, N]
                y_coords = torch.clamp(proj_batch[..., 1], 0, image_height - 1).long()  # [B, N]
                
                linear_indices = y_coords * image_width + x_coords  # [B, N]
                
                masks_flat = masks_batch.view(current_size, 1, -1).squeeze(1)  # [B, H*W]
                
                in_mask = torch.gather(masks_flat, 1, linear_indices)  # [B, N]
                in_mask = in_mask.unsqueeze(2).bool()  # [B, N, 1]
                
                visible_mask = in_image & in_mask # [B, N, 1]
            else:
                visible_mask = in_image
                
            feature_map, _ = feature_extractor.detectAndComputeDense(gt_imgs_batch)
            feature_map = F.interpolate(feature_map, size=(image_height, image_width), mode='bilinear', align_corners=False) #[B, 256, H, W]
    
            uv_norm = proj_batch.clone().to(torch.float32)  
            mask_2d = in_image.squeeze(-1) 
            uv_norm[~mask_2d] = 0
            uv_norm[..., 0] = 2.0 * uv_norm[..., 0] / image_width  - 1.0
            uv_norm[..., 1] = 2.0 * uv_norm[..., 1] / image_height - 1.0
            grid = uv_norm.unsqueeze(2)

            batch_feature = F.grid_sample(feature_map, grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False)  # [B, 256, N, 1]
            batch_feature = batch_feature.squeeze(-1).permute(0,2,1)    # [B,N,256]
            
            batch_cos_theta = get_cosine_weights(gaussians_sampled, cam_center_batch) #[B, N, 1]
            
            masked_features = batch_feature * visible_mask # [B, N, 64]
            masked_weights = batch_cos_theta * visible_mask # [B, N, 1]

            batch_weighted_sum = torch.sum(masked_features * masked_weights, dim=0)  # [N, 64]
            batch_sum_weights = torch.sum(masked_weights, dim=0)  # [N, 1]

            weighted_sum += batch_weighted_sum
            sum_weights += batch_sum_weights

    eps = 1e-6
    gaussian_feat = (weighted_sum /(sum_weights + eps)).to('cuda') #[N, 64]
    feat_path = os.path.join(output_path, config["sample"]["kpts_feature_file_name"])
    print("Save keypoints loc features to:", feat_path)
    pickle.dump(gaussian_feat, open(feat_path, "wb"))

