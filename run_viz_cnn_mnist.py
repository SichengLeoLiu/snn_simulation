"""
独立入口：加载 checkpoint 后调用 viz_cnn_mnist.save_cnn_mnist_feature_maps。
测试时推荐直接用：python main_test.py -T 8 -L 8 --viz
"""
import argparse
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from Models import modelpool
from Models.cnn_mnist import remap_legacy_cnn2_state_dict
from utils import get_torch_device, seed_all
from viz_cnn_mnist import save_cnn_mnist_feature_maps


def _checkpoint_candidates(model_dir, model, L, time_steps, suffix):
    base = "%s_L[%d]" % (model, L)

    def pth(name):
        return os.path.join(model_dir, name + ".pth")

    out = []
    if time_steps > 0:
        if suffix:
            out.append(pth(base + "_T[%d]_%s" % (time_steps, suffix)))
        out.append(pth(base + "_T[%d]" % (time_steps,)))
        if suffix:
            out.append(pth(base + "_%s" % (suffix,)))
        out.append(pth(base))
    else:
        if suffix:
            out.append(pth(base + "_%s" % (suffix,)))
        out.append(pth(base))
    seen, unique = set(), []
    for path in out:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def resolve_ckpt_path(ckpt, model_dir, model, L, T, suffix):
    if ckpt:
        path = os.path.expanduser(ckpt)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    for p in _checkpoint_candidates(model_dir, model, L, T, suffix):
        if os.path.isfile(p):
            print("加载权重:", p)
            return p
    raise FileNotFoundError(
        "未找到 checkpoint，已尝试:\n  "
        + "\n  ".join(_checkpoint_candidates(model_dir, model, L, T, suffix))
    )


def get_mnist_batch(batch_size, data_dir, device):
    data_dir = data_dir or os.path.expanduser("~/datasets")
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST(data_dir, train=False, download=True, transform=trans)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    return images.to(device), labels


@torch.no_grad()
def run_viz(args):
    device = get_torch_device(args.device)
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = resolve_ckpt_path(
        args.ckpt, args.model_dir, args.model, args.L, args.T, args.suffix
    )
    model = modelpool(args.model, "mnist").to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict, legacy = remap_legacy_cnn2_state_dict(state_dict)
    if legacy:
        print("已映射旧版 features.* 键名")
    model.load_state_dict(state_dict, strict=True)

    images, labels = get_mnist_batch(args.batch_size, args.data_dir, device)
    n = min(args.num_show, images.shape[0])
    images, labels = images[:n], labels[:n]

    tag = "standalone"
    save_cnn_mnist_feature_maps(
        model,
        images,
        labels,
        args.T,
        args.L,
        args.out_dir,
        file_tag=tag,
        logger=None,
        viz_diff_abs_max=args.viz_diff_abs_max,
        viz_feat_vmin=args.viz_feat_vmin,
        viz_feat_vmax=args.viz_feat_vmax,
    )


def main():
    p = argparse.ArgumentParser(
        description="CNN2MNIST IF1/IF2 特征图（独立脚本；测试请用 main_test.py --viz）"
    )
    p.add_argument("--T", type=int, default=8, help="SNN 时间步 T（>0）")
    p.add_argument("-L", "--L", type=int, default=8)
    p.add_argument("--batch_size", "-b", type=int, default=8)
    p.add_argument("--num_show", type=int, default=6)
    p.add_argument("--out_dir", type=str, default="./cnn_mnist_viz")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--model_dir", type=str, default="mnist-checkpoints")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--model", type=str, default="cnn2")
    p.add_argument("--suffix", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--viz_diff_abs_max", type=float, default=None)
    p.add_argument("--viz_feat_vmin", type=float, default=None)
    p.add_argument("--viz_feat_vmax", type=float, default=None)
    args = p.parse_args()
    if args.T <= 0:
        raise SystemExit("请设 --T>0")
    run_viz(args)


if __name__ == "__main__":
    main()
