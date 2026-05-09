import torch
import faiss
import numpy as np

def knn_faiss(x, K):

    N, D = x.shape
    device = x.device
    
    x_np = x.cpu().numpy()
    
    index = faiss.IndexFlatL2(D)  
    index.add(x_np)
    
    distances, indices = index.search(x_np, K + 1)
    
    knn_idx = torch.from_numpy(indices[:, 1:]).to(device)  # [N, K]
    
    return knn_idx

@torch.no_grad()
def compute_geometric_support(x, y, K=16, angle_thresh_cos=0.9659, scale_thresh=0.1, scale_limit=3.0):

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device='cuda')
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, device='cuda')
        
    dist_x = torch.cdist(x, x)  # [N, N]
    knn_idx = dist_x.topk(K + 1, dim=-1, largest=False).indices[:, 1:]  # [N, K]

    x_knn = x[knn_idx]  # [N, K, 3]
    y_knn = y[knn_idx]  # [N, K, 2]

    x_rel = x_knn - x[:, None, :]  # [N, K, 3]
    y_rel = y_knn - y[:, None, :]  # [N, K, 2]

    # Angle consistency
    def normalize(v):  # v: [..., 2]
        return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    x_unit = normalize(x_rel)  # [N, K, 3]
    y_unit = normalize(y_rel)  # [N, K, 2]

    x_dot = torch.matmul(x_unit, x_unit.transpose(1, 2))  # (N, K, K)
    y_dot = torch.matmul(y_unit, y_unit.transpose(1, 2))
    angle_mask = (x_dot - y_dot).abs() < (1 - angle_thresh_cos)  # (N, K, K)

    xj = x_rel[:, :, None, :]  # [N, K, 1, 3]
    xk = x_rel[:, None, :, :]  # [N, 1, K, 3]
    cross_x = xj[..., 0] * xk[..., 1] - xj[..., 1] * xk[..., 0]  # [N, K, K]

    yj = y_rel[:, :, None, :]  # [N, K, 1, 2]
    yk = y_rel[:, None, :, :]  # [N, 1, K, 2]
    cross_y = yj[..., 0] * yk[..., 1] - yj[..., 1] * yk[..., 0]  # [N, K, K]
    is_norm_consis = (cross_x * cross_y) > 0

    # (N, K)
    is_zero_vec = (torch.linalg.norm(x_rel, dim=-1) < 1e-6) | (torch.linalg.norm(y_rel, dim=-1) < 1e-6)

    neightbor_threshold = 50
    is_not_neighbor = (torch.linalg.norm(x_rel, dim=-1) > neightbor_threshold) | (torch.linalg.norm(y_rel, dim=-1) > neightbor_threshold)
    is_zero_vec = is_zero_vec | is_not_neighbor
    
    # (N, K, 1) and (N, 1, K) -> (N, K, K)
    is_not_triangle = torch.logical_and(is_zero_vec[:, :, None], is_zero_vec[:, None, :])

    # (N, K, 2), (N, K, 2) -> (N, K, K, 2)
    x_jk_rel = x_rel[:, :, None, :] - x_rel[:, None, :, :]
    x_jk_dist = torch.linalg.norm(x_jk_rel, dim=-1)
    y_jk_rel = y_rel[:, :, None, :] - y_rel[:, None, :, :]
    y_jk_dist = torch.linalg.norm(y_jk_rel, dim=-1)
    is_zero_vec = (x_jk_dist < 1e-6) | (y_jk_dist < 1e-6)
    is_not_triangle = is_zero_vec | is_not_triangle

    is_not_triangle.sum() - len(is_not_triangle) * K
    is_triangle = ~ is_not_triangle
    
    #Scale consistency
    # xi, yi: [N, 2] → [N, 1, 1, 2]
    xi = x[:, None, None, :]  # [N, 1, 1, 2]
    yi = y[:, None, None, :]  # [N, 1, 1, 2]

    # x_knn, y_knn: [N, K, 2]
    xj = x_knn  # [N, K, 2]
    yj = y_knn  # [N, K, 2]

    xj_ = xj[:, :, None, :]  # [N, K, 1, 2]
    xk_ = xj[:, None, :, :]  # [N, 1, K, 2]

    yj_ = yj[:, :, None, :]  # [N, K, 1, 2]
    yk_ = yj[:, None, :, :]  # [N, 1, K, 2]

    a_s = (xj_ - xi).norm(dim=-1) + 1e-8  # 边 ij
    b_s = (xk_ - xi).norm(dim=-1) + 1e-8  # 边 ik
    c_s = (xk_ - xj_).norm(dim=-1) + 1e-8  # 边 jk

    a_t = (yj_ - yi).norm(dim=-1) + 1e-8
    b_t = (yk_ - yi).norm(dim=-1) + 1e-8
    c_t = (yk_ - yj_).norm(dim=-1) + 1e-8

    s_a = a_t / a_s  # [N, K, K]
    s_b = b_t / b_s
    s_c = c_t / c_s

    diff_ab = (s_a - s_b).abs()
    diff_ac = (s_a - s_c).abs()
    diff_bc = (s_b - s_c).abs()

    scale_similar = (diff_ab < scale_thresh) & (diff_ac < scale_thresh) & (diff_bc < scale_thresh)

    lower, upper = 1.0 / scale_limit, scale_limit
    scale_valid = (
        (s_a > lower) & (s_a < upper) &
        (s_b > lower) & (s_b < upper) &
        (s_c > lower) & (s_c < upper)
    )

    scale_mask = scale_similar & scale_valid  # [N, K, K]

    support_mask = angle_mask  & scale_mask & is_triangle & is_norm_consis  # [N, K, K]

    support_mask[:, torch.arange(K), torch.arange(K)] = False
    support_score = support_mask.float().sum(dim=(1, 2))  # (N, )

    return support_score  # (N, ), (N, K, K)
