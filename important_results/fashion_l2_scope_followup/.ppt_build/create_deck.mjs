import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";


const RESULT_ROOT = path.resolve(
  process.env.RESULT_ROOT ?? "important_results/fashion_l2_scope_followup",
);
const REPO_ROOT = path.resolve(RESULT_ROOT, "../..");
const OUTPUT = path.resolve(
  process.env.FINAL_PPTX ?? path.join(RESULT_ROOT, "fashion_l2_scope_mechanism_2026-08-03.pptx"),
);
const RENDER_DIR = path.join(RESULT_ROOT, ".ppt_build", "rendered");

const W = 1280;
const H = 720;
const FONT = "Heiti SC";
const COLORS = {
  ink: "#000000",
  muted: "#555B66",
  panel: "#EDEDED",
  panel2: "#F5F5F5",
  rule: "#B8BCC4",
  blue: "#3D8DFF",
  lightBlue: "#D0EDFA",
  orange: "#D55E00",
  green: "#009E73",
  purple: "#6A3D9A",
  white: "#FFFFFF",
};


function parseCsv(text) {
  const records = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      records.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    records.push(row);
  }
  const [headers, ...data] = records.filter((item) => item.some((value) => value !== ""));
  return data.map((values) => Object.fromEntries(headers.map((key, index) => [key, values[index] ?? ""])));
}


async function readCsv(relativePath) {
  return parseCsv(await fs.readFile(path.join(REPO_ROOT, relativePath), "utf8"));
}


function pick(rows, predicate, label) {
  const row = rows.find(predicate);
  if (!row) throw new Error(`Missing row: ${label}`);
  return row;
}


function number(value) {
  return Number.parseFloat(value);
}


function fixed(value, digits = 2) {
  return number(value).toFixed(digits);
}


function signed(value, digits = 2) {
  const parsed = number(value);
  return `${parsed >= 0 ? "+" : ""}${parsed.toFixed(digits)}`;
}


function addText(slide, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position,
    fill: options.fill ?? "none",
    line: options.line ?? { style: "solid", fill: "none", width: 0 },
  });
  shape.text = String(text);
  shape.text.style = {
    fontSize: options.fontSize ?? 22,
    typeface: options.typeface ?? FONT,
    color: options.color ?? COLORS.ink,
    bold: options.bold ?? false,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "shrinkText",
    insets: options.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}


function addRect(slide, position, options = {}) {
  return slide.shapes.add({
    geometry: "rect",
    name: options.name,
    position,
    fill: options.fill ?? COLORS.panel,
    line: options.line ?? { style: "solid", fill: options.lineFill ?? "none", width: options.lineWidth ?? 0 },
  });
}


function addRule(slide, left, top, width, fill = COLORS.rule, height = 1) {
  return addRect(slide, { left, top, width, height }, { fill });
}


function addHeader(slide, title, page, kicker = "FASHION-MNIST / QCFS ANN-to-SNN") {
  addText(slide, kicker, { left: 42, top: 28, width: 500, height: 30 }, {
    fontSize: 16,
    bold: true,
    color: COLORS.muted,
  });
  addText(slide, title, { left: 42, top: 67, width: 1196, height: 78 }, {
    fontSize: 48,
    bold: true,
    autoFit: "shrinkText",
  });
  addRule(slide, 42, 154, 1196, COLORS.rule, 1);
  addText(slide, String(page), { left: 1184, top: 668, width: 54, height: 20 }, {
    fontSize: 14,
    alignment: "right",
    color: COLORS.muted,
  });
}


function addProtocolFooter(slide, text = "Protocol: ANN T=0 training, SNN T=L=16, rate_uniform, post-input-IF Gaussian noise") {
  addText(slide, text, { left: 42, top: 662, width: 1050, height: 22 }, {
    fontSize: 14,
    color: COLORS.muted,
  });
}


function setNotes(slide, talkTrack, sources) {
  slide.speakerNotes.textFrame.setText(
    `${talkTrack}\n\n[Sources]\n${sources.map((source) => `- ${source}`).join("\n")}`,
  );
  slide.speakerNotes.setVisible(true);
}


async function addImage(slide, filePath, position, alt) {
  const data = await fs.readFile(filePath);
  return slide.images.add({
    blob: data,
    contentType: "image/png",
    alt,
    fit: "contain",
    geometry: "rect",
    position,
  });
}


function addStatPanel(slide, position, stat, body, color = COLORS.blue) {
  addRect(slide, position, { fill: COLORS.panel2, lineFill: COLORS.rule, lineWidth: 1 });
  addRect(slide, { left: position.left, top: position.top, width: 7, height: position.height }, { fill: color });
  const compact = position.height < 160;
  const statTop = position.top + (compact ? 14 : 22);
  const statHeight = compact ? 48 : Math.min(96, position.height * 0.45);
  const bodyTop = position.top + (compact ? 70 : 115);
  addText(slide, stat, {
    left: position.left + 24,
    top: statTop,
    width: position.width - 46,
    height: statHeight,
  }, { fontSize: compact ? 32 : 42, bold: true, color });
  addText(slide, body, {
    left: position.left + 24,
    top: bodyTop,
    width: position.width - 46,
    height: position.top + position.height - bodyTop - 12,
  }, { fontSize: compact ? 16 : 19, color: COLORS.ink });
}


async function loadData() {
  return {
    prior: await readCsv("important_results/mne_adv_5seed/c2c4_fashion_mnist_mnist_T4_T8_T16_summary.csv"),
    equivalence: await readCsv("important_results/manual_l2_vs_wd_pilot/fashion_c2c4_T16_post_input_if/matched_current_summary.csv"),
    multiSummary: await readCsv("important_results/fashion_l2_scope_followup/multiseed_summary.csv"),
    multiDelta: await readCsv("important_results/fashion_l2_scope_followup/multiseed_paired_delta.csv"),
    scopeSummary: await readCsv("important_results/fashion_l2_scope_followup/parameter_scope_summary.csv"),
    relativeSummary: await readCsv("important_results/fashion_l2_scope_followup/absolute_vs_relative_summary.csv"),
    parameterScale: await readCsv("important_results/fashion_l2_scope_followup/parameter_scale_multiseed_summary.csv"),
    layerwise: await readCsv("important_results/fashion_l2_scope_followup/layerwise_mechanism_mean_std.csv"),
    margins: await readCsv("important_results/fashion_l2_scope_followup/output_margin_mean_std.csv"),
    architecture: await readCsv("important_results/fashion_l2_scope_followup/architecture_2x2_summary.csv"),
  };
}


async function buildDeck() {
  const data = await loadData();
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  const priorMethods = Object.fromEntries(
    ["old_detach", "orthogonal", "no_reg", "l2"].map((method) => [
      method,
      pick(data.prior, (row) => row.dataset === "fashion_mnist" && row.T === "16" && row.method === method, method),
    ]),
  );
  const multi = Object.fromEntries(
    ["wd_weight_only", "manual_l2_all"].map((method) => [
      method,
      pick(data.multiSummary, (row) => row.method === method, method),
    ]),
  );
  const delta1 = pick(data.multiDelta, (row) => row.sigma === "1.0", "multi sigma=1 delta");
  const scope = Object.fromEntries(
    data.scopeSummary.map((row) => [row.method, row]),
  );
  const relative = Object.fromEntries(
    data.relativeSummary.map((row) => [`${row.sigma_scale}:${row.method}`, row]),
  );
  const scales = Object.fromEntries(
    data.parameterScale
      .map((row) => [row.method, row]),
  );
  const layerW = pick(data.layerwise, (row) => row.method === "wd_weight_only" && row.sigma === "1.0" && row.layer === "if6", "weights if6");
  const layerAll = pick(data.layerwise, (row) => row.method === "manual_l2_all" && row.sigma === "1.0" && row.layer === "if6", "all if6");
  const marginW = pick(data.margins, (row) => row.method === "wd_weight_only" && row.sigma === "1.0", "weights margin");
  const marginAll = pick(data.margins, (row) => row.method === "manual_l2_all" && row.sigma === "1.0", "all margin");
  const architecture = Object.fromEntries(
    data.architecture.map((row) => [`${row.model}:${row.method}`, row]),
  );

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addText(slide, "REPRESENTATION STABILITY", { left: 42, top: 38, width: 500, height: 34 }, {
      fontSize: 18,
      bold: true,
      color: COLORS.muted,
    });
    addText(slide, "为什么 all-parameter L2\n会损害 SNN 鲁棒性？", { left: 42, top: 180, width: 1030, height: 235 }, {
      fontSize: 72,
      bold: true,
      verticalAlignment: "bottom",
    });
    addRect(slide, { left: 42, top: 454, width: 170, height: 8 }, { fill: COLORS.blue });
    addText(slide, "从 MNE-L2 对比到参数范围、尺度敏感性与结构交互", { left: 42, top: 500, width: 930, height: 64 }, {
      fontSize: 30,
      color: COLORS.muted,
    });
    addText(slide, "Fashion-MNIST | ANN T=0 -> SNN T=16 | 2026-08-03", { left: 42, top: 630, width: 720, height: 28 }, {
      fontSize: 18,
      color: COLORS.muted,
    });
    setNotes(slide,
      "这次汇报不再问哪一种正则化看起来最稳健，而是追问为什么 optimizer weight decay 在某些 ANN 到 SNN 转换中会失去鲁棒性。我们把统计复现、参数范围、尺度归一化、逐层传播和结构因素串成一条因果链。",
      [path.join(RESULT_ROOT, "README.md")],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "MNE-L2 不是唯一稳健方法", 2);
    addText(slide, "前期 5-seed 小模型结果表明，MNE-L2 的确优于 optimizer L2，但 Orthogonal 与它几乎重合，No Reg 也保持较强。", { left: 42, top: 176, width: 1160, height: 64 }, {
      fontSize: 24,
      color: COLORS.muted,
    });
    const entries = [
      ["old_detach", "MNE-L2", COLORS.blue],
      ["orthogonal", "Orthogonal", COLORS.green],
      ["no_reg", "No Reg", COLORS.purple],
      ["l2", "Optimizer L2", COLORS.orange],
    ];
    entries.forEach(([key, label, color], index) => {
      const row = priorMethods[key];
      const left = 42 + index * 298;
      addStatPanel(slide, { left, top: 275, width: 270, height: 300 }, `${fixed(row.sigma_0p5_acc)}%`, `${label}\nT=16, sigma=0.5\nclean ${fixed(row.clean_acc)}%`, color);
    });
    addText(slide, "研究问题因此转向：MNE-L2 的优势有多少来自目标函数本身，有多少来自对照组 weight decay 的参数范围？", { left: 42, top: 600, width: 1130, height: 44 }, {
      fontSize: 24,
      bold: true,
    });
    setNotes(slide,
      "这里先主动削弱一个过强叙事。MNE-L2 在 Fashion-MNIST 的低延迟转换中比 optimizer L2 更稳健，但 Orthogonal 基本相同，No Reg 也不差。因此不能把结果写成 MNE-L2 独有的稳健性，只能说它避开了某类 L2 训练产生的失败模式。",
      [path.join(REPO_ROOT, "important_results/mne_adv_5seed/c2c4_fashion_mnist_mnist_T4_T8_T16_summary.csv")],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "实现形式不是主要矛盾，parameter scope 才是", 3);
    const panels = [
      [42, 205, "同一 scope", "WD weights-only 与 explicit L2 weights-only 的 checkpoint 最大差为 0。\n\n数学等价在代码中得到验证。", COLORS.blue],
      [650, 205, "同一 scope", "WD all-params 与 explicit L2 all-params 的 checkpoint 最大差同样为 0。\n\n不是 optimizer API 本身造成差异。", COLORS.green],
      [42, 430, "Weights only", "只约束 Conv/Linear weight。\n不衰减 BN gamma/beta、IF threshold 与 bias。", COLORS.purple],
      [650, 430, "All parameters", "同时衰减 BN affine、IF threshold 与 bias。\nANN 可通过重参数化维持 clean accuracy。", COLORS.orange],
    ];
    panels.forEach(([left, top, heading, body, color]) => {
      addRect(slide, { left, top, width: 570, height: 175 }, { fill: COLORS.panel2, lineFill: COLORS.rule, lineWidth: 1 });
      addRect(slide, { left, top, width: 8, height: 175 }, { fill: color });
      addText(slide, heading, { left: left + 26, top: top + 22, width: 500, height: 38 }, { fontSize: 28, bold: true, color });
      addText(slide, body, { left: left + 26, top: top + 70, width: 505, height: 85 }, { fontSize: 20 });
    });
    setNotes(slide,
      "这一步排除了一个常见混淆。只要参数集合一致，SGD 的 coupled weight decay 与 loss 中的二次惩罚给出相同 checkpoint。真正发生变化的是 all-parameter 设置额外压缩了 BN、IF threshold 和 bias。",
      [path.join(REPO_ROOT, "important_results/manual_l2_vs_wd_pilot/fashion_c2c4_T16_post_input_if/matched_current_summary.csv")],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "深度不构成单调解释，VGG-like 结构才放大差距", 4);
    await addImage(slide,
      path.join(REPO_ROOT, "important_results/fashion_vgglike_l2_scope_seed42/cnn6_architecture_scope_comparison.png"),
      { left: 42, top: 174, width: 780, height: 445 },
      "Narrow and VGG-like CNN6 robustness curves",
    );
    addStatPanel(slide, { left: 855, top: 190, width: 350, height: 185 }, "+11.49 pp", "VGG-like CNN6\nsigma=1 raw gap", COLORS.orange);
    addStatPanel(slide, { left: 855, top: 405, width: 350, height: 185 }, "+0.59 pp", "Narrow CNN6\nsigma=1 raw gap", COLORS.blue);
    addProtocolFooter(slide);
    setNotes(slide,
      "把 CNN 从 2 层加深到 10 层没有产生单调差距。相同的六个 Conv-BN-IF 层，在窄网络中差异很小，在分阶段加宽的 VGG-like 结构中却达到 11.49 个百分点。因此 BN 层数不是充分解释，宽度、pooling 位置与表示 margin 必须一起考虑。",
      [
        path.join(REPO_ROOT, "important_results/fashion_deep_l2_scope_seed42/l2_scope_delta_by_depth.csv"),
        path.join(REPO_ROOT, "important_results/fashion_vgglike_l2_scope_seed42/cnn6_architecture_scope_comparison.csv"),
      ],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "后续实验把统计复现与机制定位分开", 5);
    const steps = [
      ["01", "5 training seeds", "复现 weights-only 与 all-params 的 VGG-like 差距"],
      ["02", "Parameter scope", "拆分 W+BN、W+IF、W+BN+IF，定位参数家族"],
      ["03", "Scale controls", "比较绝对噪声、按阈值归一化与按 RMS 归一化噪声"],
      ["04", "Mechanism + structure", "逐层扰动、margin，以及 width x pooling 的 2 x 2 对照"],
    ];
    addRule(slide, 115, 374, 1048, COLORS.rule, 3);
    steps.forEach(([numberText, heading, body], index) => {
      const left = 42 + index * 304;
      addRect(slide, { left: left + 58, top: 346, width: 24, height: 24 }, { fill: COLORS.blue });
      addText(slide, numberText, { left, top: 218, width: 120, height: 48 }, { fontSize: 36, bold: true, color: COLORS.blue });
      addText(slide, heading, { left, top: 280, width: 260, height: 46 }, { fontSize: 27, bold: true });
      addText(slide, body, { left, top: 415, width: 260, height: 120 }, { fontSize: 20, color: COLORS.muted });
    });
    addText(slide, "所有新增实验仍限定为 post-input-IF fixed Gaussian noise，不代表 raw-image noise。", { left: 42, top: 590, width: 1100, height: 42 }, { fontSize: 23, bold: true });
    setNotes(slide,
      "实验顺序很重要。先用五个训练种子确认主现象，再用单种子的参数范围拆分定位原因。尺度归一化检查 fixed absolute noise 是否构成关键条件，最后用逐层指标和结构对照解释为什么相同参数缩放只在某些网络中传到输出。",
      [path.join(REPO_ROOT, "QCFS_simulation/noise3_exp/RUN_fashion_l2_scope_followup.sh")],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "VGG-like 鲁棒性差距在 5 个训练 seed 上复现", 6);
    await addImage(slide, path.join(RESULT_ROOT, "figure_multiseed.png"), { left: 42, top: 172, width: 790, height: 455 }, "Five-seed absolute-noise comparison");
    addStatPanel(slide, { left: 860, top: 185, width: 345, height: 190 }, `${fixed(multi.wd_weight_only.end_mean)}%`, `Weights only at sigma=1\nstd ${fixed(multi.wd_weight_only.end_std)} pp`, COLORS.blue);
    addStatPanel(slide, { left: 860, top: 400, width: 345, height: 190 }, `${fixed(multi.manual_l2_all.end_mean)}%`, `All params at sigma=1\nstd ${fixed(multi.manual_l2_all.end_std)} pp`, COLORS.orange);
    addText(slide, `Paired gap at sigma=1: ${signed(delta1.paired_delta_mean)} +/- ${fixed(delta1.paired_delta_std)} pp; clean-adjusted ${signed(delta1.clean_adjusted_delta_mean)} pp`, { left: 42, top: 625, width: 1120, height: 30 }, { fontSize: 22, bold: true });
    setNotes(slide,
      `五个训练种子确认了主现象。sigma 等于 1 时，weights-only 平均为 ${fixed(multi.wd_weight_only.end_mean)}%，all-params 为 ${fixed(multi.manual_l2_all.end_mean)}%，paired gap 为 ${signed(delta1.paired_delta_mean)} 个百分点。clean-adjusted gap 仍为 ${signed(delta1.clean_adjusted_delta_mean)}，所以不是 clean 起点造成。`,
      [path.join(RESULT_ROOT, "multiseed_summary.csv"), path.join(RESULT_ROOT, "multiseed_paired_delta.csv")],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "IF threshold 还是 BN：谁在造成损失？", 7);
    await addImage(slide, path.join(RESULT_ROOT, "figure_parameter_scope.png"), { left: 42, top: 165, width: 820, height: 455 }, "Selective L2 parameter-scope curves");
    const scopeOrder = ["wd_weight_only", "manual_l2_w_bn", "manual_l2_w_if", "manual_l2_w_bn_if", "manual_l2_all"];
    scopeOrder.forEach((method, index) => {
      const row = scope[method];
      addText(slide, `${row.method_label}\n${fixed(row.end_accuracy)}%`, { left: 895, top: 175 + index * 87, width: 310, height: 67 }, {
        fontSize: 21,
        bold: true,
        color: index === 0 ? COLORS.blue : (index === scopeOrder.length - 1 ? COLORS.orange : COLORS.ink),
      });
    });
    addText(slide, "Numbers: SNN accuracy at absolute sigma=1", { left: 895, top: 622, width: 310, height: 24 }, { fontSize: 14, color: COLORS.muted });
    setNotes(slide,
      `参数范围拆分给出直接定位。sigma 等于 1 时，W+BN 为 ${fixed(scope.manual_l2_w_bn.end_accuracy)}%，W+IF 为 ${fixed(scope.manual_l2_w_if.end_accuracy)}%，W+BN+IF 为 ${fixed(scope.manual_l2_w_bn_if.end_accuracy)}%，all-params 为 ${fixed(scope.manual_l2_all.end_accuracy)}%。直接惩罚 IF threshold 的 W+IF 最差，BN 单独加入也造成约 6 个百分点损失。非单调排序说明 BN、IF、bias 之间存在补偿，不能把它们解释为简单相加。`,
      [path.join(RESULT_ROOT, "parameter_scope_summary.csv"), path.join(RESULT_ROOT, "parameter_scale_summary.csv")],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "固定绝对噪声是否放大了低阈值模型的劣势？", 8);
    await addImage(slide, path.join(RESULT_ROOT, "figure_absolute_vs_relative.png"), { left: 42, top: 170, width: 900, height: 435 }, "Absolute versus scale-normalized noise curves");
    const absGap = number(relative["absolute:wd_weight_only"].end_accuracy) - number(relative["absolute:manual_l2_all"].end_accuracy);
    const thresholdGap = number(relative["input_if_threshold:wd_weight_only"].end_accuracy) - number(relative["input_if_threshold:manual_l2_all"].end_accuracy);
    const rmsGap = number(relative["post_input_if_rms:wd_weight_only"].end_accuracy) - number(relative["post_input_if_rms:manual_l2_all"].end_accuracy);
    addStatPanel(slide, { left: 965, top: 180, width: 250, height: 125 }, signed(absGap), "Absolute end gap", COLORS.orange);
    addStatPanel(slide, { left: 965, top: 325, width: 250, height: 125 }, signed(thresholdGap), "Threshold-normalized", COLORS.blue);
    addStatPanel(slide, { left: 965, top: 470, width: 250, height: 125 }, signed(rmsGap), "RMS-normalized", COLORS.green);
    addText(slide, `Input threshold: weights ${fixed(scales.wd_weight_only.input_if_threshold_mean)} vs all ${fixed(scales.manual_l2_all.input_if_threshold_mean)} (5-seed means)`, { left: 42, top: 615, width: 900, height: 30 }, { fontSize: 21, bold: true });
    setNotes(slide,
      `All-params 把 5-seed mean input threshold 从 ${fixed(scales.wd_weight_only.input_if_threshold_mean)} 压到 ${fixed(scales.manual_l2_all.input_if_threshold_mean)}。绝对噪声末端 gap 为 ${signed(absGap)}，按 threshold 归一化后反转为 ${signed(thresholdGap)}，按 clean activation RMS 归一化后接近持平并为 ${signed(rmsGap)}。因此当前结论应写成 fixed-absolute-noise scale sensitivity，而不是无条件的内在不稳定。`,
      [path.join(RESULT_ROOT, "absolute_vs_relative_summary.csv"), path.join(RESULT_ROOT, "parameter_scale_multiseed_summary.csv")],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "逐层扰动最终表现为分类 margin 损失", 9);
    await addImage(slide, path.join(RESULT_ROOT, "figure_layerwise_mechanism.png"), { left: 42, top: 170, width: 805, height: 455 }, "Layerwise relative representation perturbation");
    addStatPanel(slide, { left: 875, top: 185, width: 330, height: 185 }, `${fixed(layerW.relative_l2_mean)}`, `Weights only\nIF6 relative L2 at sigma=1`, COLORS.blue);
    addStatPanel(slide, { left: 875, top: 395, width: 330, height: 185 }, `${fixed(layerAll.relative_l2_mean)}`, `All parameters\nIF6 relative L2 at sigma=1`, COLORS.orange);
    addText(slide, `IF6 spike mismatch: ${fixed(number(layerW.spike_mismatch_rate_mean) * 100)}% vs ${fixed(number(layerAll.spike_mismatch_rate_mean) * 100)}%; prediction flips: ${fixed(number(marginW.prediction_flip_rate_mean) * 100)}% vs ${fixed(number(marginAll.prediction_flip_rate_mean) * 100)}%`, { left: 42, top: 620, width: 1160, height: 35 }, { fontSize: 20, bold: true });
    setNotes(slide,
      `逐层曲线说明差距不是只存在于参数统计。sigma 等于 1 时，IF6 的相对扰动只从 ${fixed(layerW.relative_l2_mean)} 增到 ${fixed(layerAll.relative_l2_mean)}，但 spike mismatch 从 ${fixed(number(layerW.spike_mismatch_rate_mean) * 100)}% 增到 ${fixed(number(layerAll.spike_mismatch_rate_mean) * 100)}%，预测翻转率从 ${fixed(number(marginW.prediction_flip_rate_mean) * 100)}% 增到 ${fixed(number(marginAll.prediction_flip_rate_mean) * 100)}%。这说明单一表示范数不能概括失效，量化决策与分类 margin 同样关键。`,
      [path.join(RESULT_ROOT, "layerwise_mechanism_mean_std.csv"), path.join(RESULT_ROOT, "output_margin_mean_std.csv")],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "宽度与 pooling 位置共同决定差距是否暴露", 10);
    await addImage(slide, path.join(RESULT_ROOT, "figure_architecture_2x2.png"), { left: 42, top: 165, width: 890, height: 470 }, "Six-layer width and pooling two-by-two control");
    const architectureNames = ["cnn6", "cnn6_narrow_staged", "cnn6_wide_early", "cnn6_vgg"];
    architectureNames.forEach((model, index) => {
      const weights = architecture[`${model}:wd_weight_only`];
      const all = architecture[`${model}:manual_l2_all`];
      const gap = number(weights.end_accuracy) - number(all.end_accuracy);
      addText(slide, `${weights.label}\n${signed(gap)} pp`, { left: 965, top: 175 + index * 108, width: 245, height: 80 }, {
        fontSize: 21,
        bold: true,
        color: Math.abs(gap) > 5 ? COLORS.orange : COLORS.ink,
      });
    });
    addText(slide, "Displayed value: weights-only minus all-params at sigma=1", { left: 965, top: 610, width: 250, height: 40 }, { fontSize: 14, color: COLORS.muted });
    setNotes(slide,
      "六层网络的 2 x 2 对照把通道宽度与 pooling 位置拆开。窄加早池化的 gap 只有 0.59，窄加分阶段池化为 5.88，宽加早池化为 7.00，宽加分阶段池化达到 11.49 个百分点。因此宽度和 pooling 位置都在放大差距，而不是 BN 层数本身。这里仍是 seed 42，结构效应还需要更多训练种子。",
      [path.join(RESULT_ROOT, "architecture_2x2_summary.csv")],
    );
  }

  {
    const slide = presentation.slides.add();
    slide.background.fill = COLORS.white;
    addHeader(slide, "可辩护的结论更窄，也更强", 11);
    const conclusions = [
      ["01", "已确认", "VGG-like CNN6 中，all-parameter L2 的 post-input-IF absolute-noise 鲁棒性平均低于 weights-only，且在 5 seeds 上复现。", COLORS.blue],
      ["02", "机制定位", `直接正则 IF threshold 是最强负面因素；5-seed mean input threshold 从 ${fixed(scales.wd_weight_only.input_if_threshold_mean)} 降到 ${fixed(scales.manual_l2_all.input_if_threshold_mean)}。`, COLORS.orange],
      ["03", "条件边界", "阈值/RMS 归一化后 gap 消失或反转；深度和 BN 数量本身不足以解释，结构也会调制结果。", COLORS.green],
      ["04", "对 MNE-L2 的含义", "MNE-L2 应与 tuned weights-only L2、Orthogonal 和 No Reg 比较。当前证据支持它避开失败模式，不支持其稳健性独一无二。", COLORS.purple],
    ];
    conclusions.forEach(([index, heading, body, color], itemIndex) => {
      const left = itemIndex % 2 === 0 ? 42 : 650;
      const top = itemIndex < 2 ? 190 : 420;
      addText(slide, index, { left, top, width: 70, height: 48 }, { fontSize: 38, bold: true, color });
      addText(slide, heading, { left: left + 90, top, width: 440, height: 42 }, { fontSize: 28, bold: true });
      addText(slide, body, { left: left + 90, top: top + 58, width: 470, height: 120 }, { fontSize: 20, color: COLORS.muted });
    });
    addText(slide, "下一步：把 causal scopes 的关键两组扩展到 5 seeds，并在 CIFAR VGG 上用相同 parameter scope 与 scale-normalized protocol 复现。", { left: 42, top: 625, width: 1160, height: 34 }, { fontSize: 22, bold: true });
    setNotes(slide,
      "最终结论需要保持边界。我们确认的是特定转换和噪声协议下的参数范围效应，不是 raw-image robustness，也不是所有深网络的普遍定律。对论文最有价值的下一步，是用 weights-only L2 作为公平基线重新评价 MNE-L2，并把最关键的 causal scope 在 CIFAR 上复现。",
      [
        path.join(RESULT_ROOT, "multiseed_summary.csv"),
        path.join(RESULT_ROOT, "parameter_scope_summary.csv"),
        path.join(RESULT_ROOT, "absolute_vs_relative_summary.csv"),
        path.join(RESULT_ROOT, "architecture_2x2_summary.csv"),
      ],
    );
  }

  await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
  await fs.mkdir(RENDER_DIR, { recursive: true });
  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(RENDER_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(RENDER_DIR, `${stem}.layout.json`), await layout.text());
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(RENDER_DIR, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));
  const inspection = await presentation.inspect({ kind: "slide,textbox,shape,image,chart,table,notes", maxChars: 60000 });
  await fs.writeFile(path.join(RENDER_DIR, "inspection.ndjson"), inspection.ndjson);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT);
  console.log(`[DONE] ${OUTPUT}`);
}


buildDeck().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
