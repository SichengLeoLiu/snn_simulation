#!/usr/bin/env python3
"""Summarize finished FG-MNE-U jobs (everything except ImageNet).

Looks in scratch first, then local important_results.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
LOCAL = ROOT / "important_results"
SEEDS = (40, 41, 42, 43, 44)


def first_card(*paths: Path) -> dict | None:
    for path in paths:
        if path.is_file():
            card = json.loads(path.read_text())
            card["_path"] = str(path)
            return card
    return None


def fget(card: dict | None, key: str, default=float("nan")) -> float:
    if not card:
        return float(default)
    try:
        return float(card.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def fmt(x: float, nd=2) -> str:
    return f"{x:.{nd}f}" if math.isfinite(x) else "miss"


def mean_std(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], float("nan")
    return statistics.mean(finite), statistics.stdev(finite)


def row(label: str, card: dict | None, extra="") -> str:
    return (
        f"  {label:<28} "
        f"clean={fmt(fget(card, 'test_clean')):>7}  "
        f"s5={fmt(fget(card, 'test_sigma5')):>7}  "
        f"aucH={fmt(fget(card, 'test_auc_high')):>8}  "
        f"||W||={fmt(fget(card, 'classifier_frobenius'), 3):>7}"
        f"{extra}"
    )


def vgg_fgmneu_seed(dataset: str, seed: int) -> dict | None:
    rel = f"cifar_mne_nodetach_unmatched_head_l2_seed42/{dataset}/vgg16_mne_head/seed{seed}/scorecard.json"
    return first_card(SCRATCH / rel, LOCAL / rel)


def print_vgg_5seed() -> None:
    print("\n=== 1. VGG-16 FG-MNE-U 五 seed（legacy map, no-detach + unmatched L2）===")
    print("对照五 seed 来自 cifar_vgg16_nodetach_5seed（无 unmatched L2）和 detach MNE。")
    baselines = {
        ("cifar10", "L2-wo"): (91.102, 0.074, 19.92, 4.293),
        ("cifar10", "MNE detach"): (90.378, 0.145, 72.056, 2.572),
        ("cifar10", "no-detach"): (89.924, 0.252, 80.748, 1.859),
        ("cifar100", "L2-wo"): (63.826, 0.264, 4.212, 0.245),
        ("cifar100", "MNE detach"): (59.620, 0.225, 41.138, 1.234),
        ("cifar100", "no-detach"): (59.326, 0.456, 53.600, 1.120),
    }
    for dataset in ("cifar10", "cifar100"):
        print(f"\n[{dataset}]")
        cleans, s5s, auchs = [], [], []
        for seed in SEEDS:
            card = vgg_fgmneu_seed(dataset, seed)
            print(row(f"FG-MNE-U seed{seed}", card))
            cleans.append(fget(card, "test_clean"))
            s5s.append(fget(card, "test_sigma5"))
            auchs.append(fget(card, "test_auc_high"))
        mc, sc = mean_std(cleans)
        m5, s5 = mean_std(s5s)
        mh, sh = mean_std(auchs)
        n = sum(1 for x in cleans if math.isfinite(x))
        print(
            f"  {'FG-MNE-U mean±std':<28} "
            f"clean={fmt(mc)}±{fmt(sc)}  s5={fmt(m5)}±{fmt(s5)}  "
            f"aucH={fmt(mh)}±{fmt(sh)}  n={n}"
        )
        for name in ("L2-wo", "MNE detach", "no-detach"):
            c, cs, s, ss = baselines[(dataset, name)]
            print(f"  {name+' 5-seed':<28} clean={c:.2f}±{cs:.2f}  s5={s:.2f}±{ss:.2f}")
        if n >= 5 and math.isfinite(m5):
            nd = baselines[(dataset, "no-detach")][2]
            det = baselines[(dataset, "MNE detach")][2]
            if m5 >= nd - 0.3:
                print("  -> 高噪声不劣于 plain no-detach，五 seed 主方法站得住。")
            elif m5 > det:
                print("  -> 仍优于 detach，但相对 plain no-detach 没有稳定增益。")
            else:
                print("  -> 未稳定超过 detach；不要扩主表口径。")


def print_resnet_samemap() -> None:
    print("\n=== 2. ResNet-18 同-map seed42：no-detach 无 unmatched L2（rmap）===")
    print("比较对象都应是 resnet map；legacy no-detach 只作附录。")
    for dataset in ("cifar10", "cifar100"):
        print(f"\n[{dataset}]")
        cards = {
            "L2-wo": first_card(
                LOCAL / f"cifar_mne_unmatched_head_l2_seed42/{dataset}/resnet18_l2wo/seed42/scorecard.json",
                SCRATCH / f"cifar_mne_unmatched_head_l2_seed42/{dataset}/resnet18_l2wo/seed42/scorecard.json",
            ),
            "detach resnet": first_card(
                LOCAL / f"cifar_mne_unmatched_head_l2_seed42/{dataset}/resnet18_mne/seed42/scorecard.json",
                SCRATCH / f"cifar_mne_unmatched_head_l2_seed42/{dataset}/resnet18_mne/seed42/scorecard.json",
            ),
            "detach+U": first_card(
                LOCAL / f"cifar_mne_unmatched_head_l2_seed42/{dataset}/resnet18_mne_head/seed42/scorecard.json",
                SCRATCH / f"cifar_mne_unmatched_head_l2_seed42/{dataset}/resnet18_mne_head/seed42/scorecard.json",
            ),
            "no-detach legacy": first_card(
                LOCAL / f"cifar_mne_nodetach_unmatched_head_l2_seed42/{dataset}/resnet18_mne/seed42/scorecard.json",
                SCRATCH / f"cifar_mne_nodetach_unmatched_head_l2_seed42/{dataset}/resnet18_mne/seed42/scorecard.json",
            ),
            "no-detach resnet": first_card(
                SCRATCH / f"cifar_resnet18_nodetach_resnetmap_seed42/{dataset}/resnet18_mne_body/seed42/scorecard.json",
                LOCAL / f"cifar_resnet18_nodetach_resnetmap_seed42/{dataset}/resnet18_mne_body/seed42/scorecard.json",
            ),
            "FG-MNE-U": first_card(
                LOCAL / f"cifar_mne_nodetach_unmatched_head_l2_seed42/{dataset}/resnet18_mne_head/seed42/scorecard.json",
                SCRATCH / f"cifar_mne_nodetach_unmatched_head_l2_seed42/{dataset}/resnet18_mne_head/seed42/scorecard.json",
            ),
        }
        for name, card in cards.items():
            print(row(name, card))
        body = cards["no-detach resnet"]
        hybrid = cards["FG-MNE-U"]
        legacy = cards["no-detach legacy"]
        if body and hybrid:
            d_head = fget(hybrid, "test_sigma5") - fget(body, "test_sigma5")
            d_map = fget(body, "test_sigma5") - fget(legacy, "test_sigma5")
            print(
                f"  Δs5(FG-MNE-U − same-map body)={fmt(d_head)}   "
                f"Δs5(same-map body − legacy ND)={fmt(d_map)}"
            )
            if abs(d_head) < 1.0 and d_map > 3:
                print("  -> 先前 +head 大跳主要来自 resnet map，不是 unmatched L2。")
            elif d_head > 1.5:
                print("  -> 同 map 下 unmatched L2 仍有增益，可以扩 ResNet 五 seed。")
            else:
                print("  -> unmatched L2 在同 map no-detach 上增益有限。")


def print_ssd() -> None:
    print("\n=== 3. SSD300-VGG16 / VOC seed42：FG-MNE-U ===")
    print("闸门：clean 下降不超过约 0.5 pp；AUC[3,5] 不劣于 no-detach，最好超过 L2-wo。")
    l2 = first_card(
        LOCAL / "voc_ssd300_five_regs_seed42/l2wo/scorecard.json",
        SCRATCH / "voc_ssd300_five_regs_seed42/l2wo/scorecard.json",
    )
    nd = first_card(
        LOCAL / "voc_ssd300_five_regs_seed42/nodetach/scorecard.json",
        SCRATCH / "voc_ssd300_five_regs_seed42/nodetach/scorecard.json",
    )
    hd = first_card(
        SCRATCH / "voc_ssd300_nodetach_unmatched_head_l2_seed42/mne_head/scorecard.json",
        LOCAL / "voc_ssd300_nodetach_unmatched_head_l2_seed42/mne_head/scorecard.json",
    )
    print(row("L2-wo", l2))
    print(row("no-detach", nd))
    print(row("FG-MNE-U", hd))
    if hd and l2 and nd:
        dclean = fget(hd, "test_clean") - fget(l2, "test_clean")
        print(
            f"  Δclean vs L2-wo={fmt(dclean)} pp   "
            f"s5 FG={fmt(fget(hd,'test_sigma5'))}  "
            f"ND={fmt(fget(nd,'test_sigma5'))}  "
            f"L2={fmt(fget(l2,'test_sigma5'))}   "
            f"aucH FG={fmt(fget(hd,'test_auc_high'))}  "
            f"ND={fmt(fget(nd,'test_auc_high'))}  "
            f"L2={fmt(fget(l2,'test_auc_high'))}"
        )
        clean_ok = dclean >= -0.55
        auc_ok = fget(hd, "test_auc_high") + 1e-9 >= fget(nd, "test_auc_high")
        if clean_ok and auc_ok:
            print("  -> 过闸门，可以扩五 seed。")
        elif clean_ok:
            print("  -> clean 可接受，但 robust 未超过 no-detach，先不要扩五 seed。")
        else:
            print("  -> clean 掉太多或结果缺失，不要扩五 seed。")


def print_grad() -> None:
    print("\n=== 4. VGG FG-MNE-U 梯度 ablation seed42（始终开 unmatched L2）===")
    reuse = {
        "detach": (
            LOCAL / "cifar_mne_unmatched_head_l2_seed42/{ds}/vgg16_mne_head/seed42/scorecard.json",
            SCRATCH / "cifar_mne_unmatched_head_l2_seed42/{ds}/vgg16_mne_head/seed42/scorecard.json",
        ),
        "full": (
            LOCAL / "cifar_mne_nodetach_unmatched_head_l2_seed42/{ds}/vgg16_mne_head/seed42/scorecard.json",
            SCRATCH / "cifar_mne_nodetach_unmatched_head_l2_seed42/{ds}/vgg16_mne_head/seed42/scorecard.json",
        ),
    }
    for dataset in ("cifar10", "cifar100"):
        print(f"\n[{dataset}]  ∇γ  ∇λ")
        cells = {
            "detach": (
                False,
                False,
                first_card(*[Path(str(path).format(ds=dataset)) for path in reuse["detach"]]),
            ),
            "gamma": (
                True,
                False,
                first_card(
                    SCRATCH / f"cifar_fgmneu_grad_ablation_seed42/{dataset}/vgg16_fgmneu_gamma/seed42/scorecard.json",
                    LOCAL / f"cifar_fgmneu_grad_ablation_seed42/{dataset}/vgg16_fgmneu_gamma/seed42/scorecard.json",
                ),
            ),
            "lambda": (
                False,
                True,
                first_card(
                    SCRATCH / f"cifar_fgmneu_grad_ablation_seed42/{dataset}/vgg16_fgmneu_lambda/seed42/scorecard.json",
                    LOCAL / f"cifar_fgmneu_grad_ablation_seed42/{dataset}/vgg16_fgmneu_lambda/seed42/scorecard.json",
                ),
            ),
            "full": (
                True,
                True,
                first_card(*[Path(str(path).format(ds=dataset)) for path in reuse["full"]]),
            ),
        }
        for name, (dg, dl, card) in cells.items():
            extra = (
                f"  λ={fmt(fget(card, 'if_thresh_mean'), 3)}  "
                f"γ={fmt(fget(card, 'bn_gamma_mean'), 3)}  "
                f"gain={fmt(fget(card, 'folded_gain_geomean'), 3)}  "
                f"gap={fmt(fget(card, 'ann_snn_gap'))}  "
                f"ρ5={fmt(fget(card, 'rho_true_sigma5_median'), 3)}"
            )
            print(f"  {name:<8} ∇γ={int(dg)} ∇λ={int(dl)} |{row('', card, extra)[30:]}")
        full = cells["full"][2]
        lam = cells["lambda"][2]
        gam = cells["gamma"][2]
        det = cells["detach"][2]
        if full and lam and gam and det:
            if abs(fget(lam, "test_sigma5") - fget(full, "test_sigma5")) < 1.5 and fget(gam, "test_sigma5") < fget(lam, "test_sigma5") - 2:
                print("  -> 高噪声主要来自 ∇λ，不是 ∇γ。")
            elif abs(fget(gam, "test_sigma5") - fget(full, "test_sigma5")) < 1.5:
                print("  -> ∇γ 已接近 full，λ 不是唯一因素。")
            if fget(full, "if_thresh_mean") > 1.5 * max(fget(det, "if_thresh_mean"), 1e-8):
                print("  -> full/λ 的 IF 阈值明显增大，检查是否靠抬 λ 减小 MNE loss。")


def print_effective() -> None:
    print("\n=== 5. VGG 五行 component：补 Effective L2 + unmatched L2 ===")
    for dataset in ("cifar10", "cifar100"):
        print(f"\n[{dataset}]")
        cards = {
            "L2-wo": first_card(
                LOCAL / f"cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_l2wo_fixed/scorecard.json",
                SCRATCH / f"cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_l2wo_fixed/scorecard.json",
            ),
            "Effective (body only)": first_card(
                LOCAL / f"cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_effective_fixed/scorecard.json",
                SCRATCH / f"cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_effective_fixed/scorecard.json",
            ),
            "Effective + U": first_card(
                SCRATCH / f"cifar_vgg16_effective_l2_head_seed42/{dataset}/vgg16_effective_head/seed42/scorecard.json",
                LOCAL / f"cifar_vgg16_effective_l2_head_seed42/{dataset}/vgg16_effective_head/seed42/scorecard.json",
            ),
            "detach + U": first_card(
                LOCAL / f"cifar_mne_unmatched_head_l2_seed42/{dataset}/vgg16_mne_head/seed42/scorecard.json",
                SCRATCH / f"cifar_mne_unmatched_head_l2_seed42/{dataset}/vgg16_mne_head/seed42/scorecard.json",
            ),
            "FG-MNE-U": first_card(
                LOCAL / f"cifar_mne_nodetach_unmatched_head_l2_seed42/{dataset}/vgg16_mne_head/seed42/scorecard.json",
                SCRATCH / f"cifar_mne_nodetach_unmatched_head_l2_seed42/{dataset}/vgg16_mne_head/seed42/scorecard.json",
            ),
            "∇λγ, no U": first_card(
                LOCAL / f"cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_nodetach_fixed/scorecard.json",
                SCRATCH / f"cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_nodetach_fixed/scorecard.json",
            ),
        }
        for name, card in cards.items():
            print(row(name, card))
        eff_u = cards["Effective + U"]
        eff = cards["Effective (body only)"]
        if eff_u and eff:
            d = fget(eff_u, "test_sigma5") - fget(eff, "test_sigma5")
            print(f"  Δs5(Effective+U − body-only Effective)={fmt(d)}")


def main() -> None:
    print(f"SCRATCH exists={SCRATCH.is_dir()}  LOCAL={LOCAL}")
    print_vgg_5seed()
    print_resnet_samemap()
    print_ssd()
    print_grad()
    print_effective()
    print("\nImageNet 三份仍在排队，未汇总。")


if __name__ == "__main__":
    sys.exit(main())
