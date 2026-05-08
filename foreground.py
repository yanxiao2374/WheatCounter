import cv2
import numpy as np
import torch


def foreground_generator(x_last,all_feats, B, C, H, W,x):
    for i in range(B):
        feat_map = x_last[i]  #  (N_patches, embed_dim)
        heatmap = torch.norm(feat_map, dim=1)  # L2范数 (N_patches,)

        # patch数假设为方形
        n_patches = feat_map.shape[0]
        n_h = n_w = int(n_patches ** 0.5)
        heatmap = heatmap.reshape(n_h, n_w).detach().cpu().numpy()

        # 上采样到原图大小
        heatmap = cv2.resize(heatmap, (W, H))
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化

        last = heatmap_norm

        # 原图归一化到0~255
        img_np = x[i].detach().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)

        # 计算“前 keep_ratio 百分比”阈值
        thresh = max(np.percentile(last, 100 *0.3), 0.3)

        mask = (last >= thresh).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel)

        # 去小点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 填充小洞
        mask_inv = 1 - mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
        min_hole_area = 100  # 小于此面积的洞会被填充
        for j in range(1, num_labels):  # 0 是外部背景
            if stats[j, cv2.CC_STAT_AREA] <= min_hole_area:
                mask[labels == j] = 1

        mask = np.expand_dims(mask, axis=2)  # (H,W,1)

        # 应用 mask
        overlay = img_np * mask  # 低关注度=0(黑)，高关注度保留原图像素

        return overlay
        save_path = f"layer_{i}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 融合热图（这里用最后一层的颜色）
        heatmap_color_first = cv2.applyColorMap(np.uint8(255 * first), cv2.COLORMAP_JET)
        heatmap_color_first = cv2.cvtColor(heatmap_color_first, cv2.COLOR_BGR2RGB)
        heatmap_color_last = cv2.applyColorMap(np.uint8(255 * last), cv2.COLORMAP_JET)
        heatmap_color_last = cv2.cvtColor(heatmap_color_last, cv2.COLOR_BGR2RGB)

        # 叠加显示
        overlay_first = cv2.addWeighted(img_np, 0.6, heatmap_color_first, 0.4, 0)
        overlay_last = cv2.addWeighted(img_np, 0.6, heatmap_color_last, 0.4, 0)
        save_path = f"red_layer_first_{i}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(overlay_first, cv2.COLOR_RGB2BGR))

        save_path = f"red_layer_last_{i}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(overlay_last, cv2.COLOR_RGB2BGR))

