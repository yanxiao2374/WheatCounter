import cv2
import numpy as np
import torch


def foreground_generator(all_feats, B, C, H, W,x):
    for i in range(B):
        # 保存第一层和最后一层的热图
        heatmaps = {}
        num_layers = len(all_feats)
        for layer_idx, feat in enumerate(all_feats):
            feat_map = feat[i][1:, :]  # 取第一个样本 (N_patches, embed_dim)
            heatmap = torch.norm(feat_map, dim=1)  # L2范数 (N_patches,)

            # patch数假设为方形
            n_patches = feat_map.shape[0]
            n_h = n_w = int(n_patches ** 0.5)
            heatmap = heatmap.reshape(n_h, n_w).detach().cpu().numpy()

            # 上采样到原图大小
            heatmap = cv2.resize(heatmap, (W, H))
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化

            # 只保存第1层和最后1层
            if layer_idx == 0:
                heatmaps['first'] = heatmap_norm
            elif layer_idx == num_layers - 1:
                heatmaps['last'] = heatmap_norm

        first = heatmaps['first']
        last = heatmaps['last']

        # 原图归一化到0~255
        img_np = x[i].detach().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)

        thresh1 = 1 #0.7
        thresh2 = 0.25 #0.6

        mask1 = (first >= thresh1).astype(np.uint8)
        mask2 = (last <= thresh2).astype(np.uint8)

        # 可以取并集或交集：
        mask_union = np.clip(mask1 + mask2, 0, 1)  # 并集

        #
        # heatmap = last  # 使用最后一层关注图
        # mask_expanded = np.zeros_like(mask_union)
        #
        # # 多级膨胀策略
        # levels = [
        #     (0.6, 3),  # 高关注 → 大膨胀核
        #     (0.4, 5),  # 中关注 → 中等膨胀核
        #     (0.2, 7),  # 低关注 → 小膨胀核
        # ]
        #
        # for thr, ksize in levels:
        #     temp_mask = (heatmap <= thr).astype(np.uint8)
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        #     temp_mask = cv2.dilate(temp_mask, kernel)
        #     mask_expanded = np.maximum(mask_expanded, temp_mask)

        # 将扩张结果与原 mask 结合
        mask_union = np.clip(mask_union, 0, 1)



        # --------------------------
        # 去小点（开运算） + 填洞（闭运算或连通域）
        # --------------------------
        # 去小点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_union = cv2.morphologyEx(mask_union, cv2.MORPH_OPEN, kernel)

        # 填充小洞
        # 方法1：闭运算
        # mask_union = cv2.morphologyEx(mask_union, cv2.MORPH_CLOSE, kernel)
        # 方法2：连通域填充小洞（可选，更精确）
        mask_inv = 1 - mask_union
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
        min_hole_area = 100  # 小于此面积的洞会被填充
        for j in range(1, num_labels):  # 0 是外部背景
            if stats[j, cv2.CC_STAT_AREA] <= min_hole_area:
                mask_union[labels == j] = 1

        mask_union = np.expand_dims(mask_union, axis=2)  # (H,W,1)

        # 应用 mask
        overlay = img_np * mask_union  # 低关注度=0(黑)，高关注度保留原图像素

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
        save_path = f"layer_first_{i}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(overlay_first, cv2.COLOR_RGB2BGR))

        save_path = f"layer_last_{i}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(overlay_last, cv2.COLOR_RGB2BGR))

