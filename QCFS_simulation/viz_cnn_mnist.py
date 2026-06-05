"""
CNN2MNIST：IF1 / IF2 特征图可视化（供 main_test --viz 或 run_viz_cnn_mnist 调用）。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Models.spike_temporal_adjust import SPIKE_SCHEDULE_MODES

VIZ_MODE_ORDER = sorted(SPIKE_SCHEDULE_MODES)


def _feature_2d(x_bchw):
    return x_bchw.mean(axis=1)


def _feat_to_numpy(feat_tb, T):
    x = feat_tb.detach().float().cpu().numpy()
    if T and T > 0:
        x = x / float(T)
    return x


def _logits_to_pred(logits):
    if logits.dim() == 3:
        return logits.mean(0).argmax(dim=1)
    return logits.argmax(dim=1)


def _save_dual_feature_grid(
    map_if1,
    map_if2,
    ann_if1_2d,
    ann_if2_2d,
    diff_if1,
    diff_if2,
    mode_order,
    suffix_titles,
    vis_idx,
    vmin1,
    vmax1,
    vmin2,
    vmax2,
    L_signed1,
    L_signed2,
    out_path_if1,
    out_path_if2,
    suptitle_if1,
    suptitle_if2,
):
    n_modes = len(mode_order)
    n_rows = 2 * n_modes + 1
    num_show = len(vis_idx)

    def draw_grid(map_2d, ann_2d, diff_2d, vmin, vmax, Ls, suptitle, out_path):
        fig, axes = plt.subplots(n_rows, num_show, figsize=(6 * num_show, 4.8 * n_rows))
        if num_show == 1:
            axes = axes.reshape(-1, 1)
        for col, s in enumerate(vis_idx):
            for r, m in enumerate(mode_order):
                axes[r, col].imshow(
                    map_2d[m][s], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax
                )
                axes[r, col].set_title("s%d %s%s" % (s, m, suffix_titles[m][s]))
                axes[r, col].axis("off")
            axes[n_modes, col].imshow(
                ann_2d[s], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax
            )
            axes[n_modes, col].set_title("s%d ANN T=0" % (s,))
            axes[n_modes, col].axis("off")
            for r, m in enumerate(mode_order):
                rr = n_modes + 1 + r
                axes[rr, col].imshow(
                    diff_2d[m][s],
                    cmap="gray",
                    aspect="equal",
                    vmin=-Ls,
                    vmax=Ls,
                )
                axes[rr, col].set_title("s%d %s - ANN" % (s, m))
                axes[rr, col].axis("off")
        plt.suptitle(suptitle, fontsize=10, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out_path)

    draw_grid(
        map_if1,
        ann_if1_2d,
        diff_if1,
        vmin1,
        vmax1,
        L_signed1,
        suptitle_if1,
        out_path_if1,
    )
    draw_grid(
        map_if2,
        ann_if2_2d,
        diff_if2,
        vmin2,
        vmax2,
        L_signed2,
        suptitle_if2,
        out_path_if2,
    )


@torch.no_grad()
def save_cnn_mnist_feature_maps(
    model,
    images,
    labels,
    T_snn,
    L,
    out_dir,
    file_tag="",
    logger=None,
    viz_diff_abs_max=None,
    viz_feat_vmin=None,
    viz_feat_vmax=None,
):
    """
    对已加载的 CNN2MNIST 在 images 上生成 IF1/IF2 特征大图并保存。
    需要 T_snn > 0；model 须含 forward_with_if_features。
    """
    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    if not hasattr(model, "forward_with_if_features"):
        _log("viz: 模型无 forward_with_if_features，跳过")
        return
    if T_snn <= 0:
        _log("viz: 需要 T>0，跳过")
        return

    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    images = images.to(device)
    B = images.shape[0]
    gt = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    feats_if1 = {}
    feats_if2 = {}
    preds = {}
    model.eval()

    for m in VIZ_MODE_ORDER:
        model.set_T(T_snn)
        model.set_L(L)
        model.set_spike_schedule(m)
        logits, f1, f2 = model.forward_with_if_features(images)
        feats_if1[m] = _feat_to_numpy(f1, T_snn)
        feats_if2[m] = _feat_to_numpy(f2, T_snn)
        preds[m] = _logits_to_pred(logits).cpu().numpy()

    model.set_T(0)
    model.set_L(L)
    model.set_spike_schedule("normal")
    _, f1_ann, f2_ann = model.forward_with_if_features(images)
    ann_if1_np = _feat_to_numpy(f1_ann, 0)
    ann_if2_np = _feat_to_numpy(f2_ann, 0)

    model.set_T(T_snn)
    model.set_L(L)

    map_if1_2d = {k: _feature_2d(v) for k, v in feats_if1.items()}
    map_if2_2d = {k: _feature_2d(v) for k, v in feats_if2.items()}
    ann_if1_2d = _feature_2d(ann_if1_np)
    ann_if2_2d = _feature_2d(ann_if2_np)
    diff_if1 = {k: map_if1_2d[k] - ann_if1_2d for k in VIZ_MODE_ORDER}
    diff_if2 = {k: map_if2_2d[k] - ann_if2_2d for k in VIZ_MODE_ORDER}

    suffix = {m: [""] * B for m in VIZ_MODE_ORDER}
    for m in VIZ_MODE_ORDER:
        p = preds[m]
        suffix[m] = [
            " | g=%d p=%d %s" % (int(gt[i]), int(p[i]), "OK" if p[i] == gt[i] else "ERR")
            for i in range(B)
        ]

    vis_idx = np.arange(B, dtype=np.int64)

    if viz_feat_vmin is not None and viz_feat_vmax is not None:
        fvmin = float(viz_feat_vmin)
        fvmax = float(viz_feat_vmax)
        vmin1 = vmin2 = fvmin
        vmax1 = vmax2 = fvmax
    else:
        vmin1 = min([map_if1_2d[k].min() for k in VIZ_MODE_ORDER] + [ann_if1_2d.min()])
        vmax1 = max([map_if1_2d[k].max() for k in VIZ_MODE_ORDER] + [ann_if1_2d.max(), 1e-6])
        vmin2 = min([map_if2_2d[k].min() for k in VIZ_MODE_ORDER] + [ann_if2_2d.min()])
        vmax2 = max([map_if2_2d[k].max() for k in VIZ_MODE_ORDER] + [ann_if2_2d.max(), 1e-6])

    diff_max1 = max(np.abs(diff_if1[k]).max() for k in VIZ_MODE_ORDER)
    diff_max2 = max(np.abs(diff_if2[k]).max() for k in VIZ_MODE_ORDER)
    L1 = float(viz_diff_abs_max) if viz_diff_abs_max else max(diff_max1, 0.01)
    L2 = float(viz_diff_abs_max) if viz_diff_abs_max else max(diff_max2, 0.01)

    tag = (file_tag + "_") if file_tag else ""
    tag += "T%d_L%d" % (T_snn, L)
    out1 = os.path.join(out_dir, "cnn_mnist_if1_feature_maps_%s.png" % tag)
    out2 = os.path.join(out_dir, "cnn_mnist_if2_feature_maps_%s.png" % tag)

    _save_dual_feature_grid(
        map_if1_2d,
        map_if2_2d,
        ann_if1_2d,
        ann_if2_2d,
        diff_if1,
        diff_if2,
        VIZ_MODE_ORDER,
        suffix,
        vis_idx,
        vmin1,
        vmax1,
        vmin2,
        vmax2,
        L1,
        L2,
        out1,
        out2,
        "CNN2MNIST after IF1 (pre-pool) | diff SNN-ANN grayscale +/- %.4f (mid=0)" % L1,
        "CNN2MNIST after IF2 (pre-pool) | diff SNN-ANN grayscale +/- %.4f (mid=0)" % L2,
    )

    _log(
        "viz: saved IF1/IF2 maps to %s (tag=%s) |diff|_max if1=%.4f if2=%.4f"
        % (out_dir, tag, diff_max1, diff_max2)
    )
