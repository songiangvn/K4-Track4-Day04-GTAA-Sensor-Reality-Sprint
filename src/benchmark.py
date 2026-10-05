"""T1 · Camera degradation health score — benchmark nhỏ trên BDD100K + YOLOv8n.

Giữ nguyên ảnh và pipeline, mỗi lần chỉ đổi MỘT loại lỗi camera ở một mức cụ thể, rồi đo:
  • sức khỏe ảnh (không cần nhãn): blur score, saturation ratio, entropy
  • thuật toán (cần nhãn GT của BDD100K): mAP@0.5, mAP@0.5:0.95 của YOLOv8n
  • độ tự tin của detector: uncertainty = 1 − max_confidence (predict ở conf 0,05) — tín hiệu
    "least confidence" hay dùng để chọn ảnh cần gán nhãn trong active learning

Dữ liệu: biến môi trường BDD100K_ROOT (mặc định data/bdd100k), xem README.md.
Chạy:  python src/benchmark.py --n 200            (đầy đủ)
       python src/benchmark.py --n 8 --smoke      (kiểm tra nhanh trên CPU)
       python src/benchmark.py --families overexpose --no-night --tag glare   (phần mở rộng)
"""

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # không để ultralytics tự pip install
os.environ.setdefault("YOLO_VERBOSE", "False")      # gọn log: không in banner / thanh tiến độ của ultralytics

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
# Thư mục BDD100K bản FiftyOne (HuggingFace dgural/bdd100k): chứa samples.json và data/*.jpg.
BDD100K_ROOT = Path(os.environ.get("BDD100K_ROOT", ROOT / "data" / "bdd100k"))
# Weights COCO của YOLOv8n; để tên trần thì ultralytics tự tải về lần chạy đầu.
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")

# Lớp BDD -> id COCO để dùng thẳng weights COCO; lớp không có tương ứng (traffic sign, train,
# other vehicle...) bị bỏ. Tên lớp theo bản FiftyOne của BDD100K.
BDD_TO_COCO = {
    "car": 2, "truck": 7, "bus": 5, "pedestrian": 0, "rider": 0,
    "traffic light": 9, "bicycle": 1, "motorcycle": 3,
}

# Mỗi họ lỗi: tên tham số, [(giá trị, severity theo nguồn S5 hoặc None)]. Mức 0 = baseline sạch.
# Nguồn S5: Dong et al., CVPR 2023 — github.com/thu-ml/3D_Corruptions_AD @ 48c23f7, Camera_corruptions.py.
# Hai họ đầu lấy ĐÚNG định nghĩa severity 1/3/5 của S5; hai họ sau S5 không có, nhóm tự thêm.
FAMILIES = {
    # ImageMotionBlurFrontBack (lỗi motion blur camera mặc định của S5 trên KITTI): zoom blur khi xe
    # lao về phía trước, phóng to tới 1 + 0,02·severity
    "motion_blur": ("zoom", [(0.02, 1), (0.06, 3), (0.10, 5)]),
    # ImageAddGaussianNoise -> imagecorruptions gaussian_noise (ImageNet-C): σ = [.08 .12 .18 .26 .38]
    # trên thang 0..1; ghi ở đây theo thang 0..255 (.08·255, .18·255, .38·255)
    "noise": ("sigma", [(20.4, 1), (45.9, 3), (96.9, 5)]),
    # Nhóm tự thêm — thiếu sáng: nhân giá trị pixel (S5 không có họ này)
    "dark": ("gain", [(0.5, None), (0.25, None), (0.1, None)]),
    # Nhóm tự thêm (phần mở rộng) — chói / thừa sáng: nhân rồi cắt ở 255
    "overexpose": ("gain", [(1.5, None), (2.5, None), (4.0, None)]),
}
DEFAULT_FAMILIES = ["motion_blur", "noise", "dark"]
JPEG_QUALITY = 95  # mọi điều kiện, kể cả baseline, đều qua cùng một lần nén JPEG


# ---------------------------------------------------------------- tạo lỗi
def zoom_blur(image: np.ndarray, max_zoom: float) -> np.ndarray:
    """Viết lại ImageMotionBlurFrontBack.zoom_blur của S5 bằng OpenCV (bản gốc dùng scipy zoom):
    trung bình ảnh gốc với các bản phóng to từ tâm, hệ số 1, 1,005, ... < 1 + max_zoom."""
    h, w = image.shape[:2]
    x = image.astype(np.float32) / 255.0
    out = np.zeros_like(x)
    factors = np.arange(1, 1 + max_zoom, 0.005)
    for z in factors:
        ch, cw = int(np.ceil(h / z)), int(np.ceil(w / z))
        top, left = (h - ch) // 2, (w - cw) // 2
        crop = x[top:top + ch, left:left + cw]
        out += cv2.resize(crop, (round(cw * z), round(ch * z)), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return (np.clip((x + out) / (len(factors) + 1), 0, 1) * 255).astype(np.uint8)


def degrade(image: np.ndarray, family: str, level: float, rng: np.random.Generator) -> np.ndarray:
    if family == "motion_blur":
        return zoom_blur(image, level)
    if family == "noise":
        return np.clip(image.astype(np.float32) + rng.normal(0, level, image.shape), 0, 255).astype(np.uint8)
    if family in ("dark", "overexpose"):
        return np.clip(image.astype(np.float32) * level, 0, 255).astype(np.uint8)
    raise ValueError(family)


# ---------------------------------------------------------------- sức khỏe ảnh
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)


def health_metrics(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
    nonzero = hist[hist > 0]
    return {
        # variance của Laplacian: cao = nhiều cạnh sắc nét, thấp = nhòe (đơn vị: mức xám²)
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        # tỉ lệ pixel bị cắt ở hai đầu (≤5 hoặc ≥250): mất thông tin do quá tối / quá sáng
        "saturation_ratio": float(((gray <= 5) | (gray >= 250)).mean()),
        # entropy Shannon của histogram xám (bit, tối đa 8): thấp = ảnh nghèo thông tin
        "entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
        # CẢI TIẾN đề xuất: ước lượng độ lệch chuẩn nhiễu (Immerkær 1996, "Fast noise variance
        # estimation"): lọc bằng mặt nạ Laplacian bậc hai triệt tiêu cạnh/vùng phẳng, chỉ giữ nhiễu
        # (đơn vị: mức xám 0-255, cao = nhiễu)
        "noise_est": float(np.sqrt(np.pi / 2) / (6 * (gray.shape[0] - 2) * (gray.shape[1] - 2))
                           * np.abs(cv2.filter2D(gray.astype(np.float64), -1, IMMERKAER)[1:-1, 1:-1]).sum()),
    }


# ---------------------------------------------------------------- dữ liệu
def load_split(bdd_root: Path, timeofday: str, n: int, seed: int) -> list[dict]:
    samples = json.load(open(bdd_root / "samples.json"))["samples"]
    pool = [s for s in samples if s["timeofday"]["label"] == timeofday and s["weather"]["label"] == "clear"]
    pool.sort(key=lambda s: s["filepath"])  # thứ tự cố định trước khi bốc ngẫu nhiên
    return random.Random(seed).sample(pool, min(n, len(pool)))


def write_labels(samples: list[dict], label_dir: Path) -> int:
    label_dir.mkdir(parents=True, exist_ok=True)
    boxes = 0
    for s in samples:
        lines = []
        for det in s["detections"]["detections"]:
            cls = BDD_TO_COCO.get(det["label"])
            if cls is None:
                continue
            x, y, w, h = det["bounding_box"]  # FiftyOne: góc trên-trái, chuẩn hoá 0..1
            lines.append(f"{cls} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
        boxes += len(lines)
        (label_dir / (Path(s["filepath"]).stem + ".txt")).write_text("\n".join(lines))
    return boxes


def build_condition(name: str, samples, bdd_root, label_dir, work, family, level, seed) -> Path:
    cond_dir = work / name
    image_dir = cond_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    # chép chứ không symlink: Windows không cho tạo symlink nếu thiếu quyền admin
    if not (cond_dir / "labels").exists():
        shutil.copytree(label_dir, cond_dir / "labels")
    for i, s in enumerate(samples):
        out = image_dir / Path(s["filepath"]).name
        if out.exists():
            continue
        image = cv2.imread(str(bdd_root / s["filepath"]))
        if family in FAMILIES:  # "night_real" là ảnh đêm thật, không gây lỗi thêm
            image = degrade(image, family, level, np.random.default_rng(seed + i))
        cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return cond_dir


# ---------------------------------------------------------------- đo một điều kiện
def evaluate(model, cond_dir: Path, names: dict, device, batch: int) -> tuple[list[dict], dict]:
    rows = []
    images = sorted((cond_dir / "images").glob("*.jpg"))
    for path in images:
        image = cv2.imread(str(path))
        # ngưỡng thấp 0,05 để ảnh khó vẫn hiện box mờ, phân biệt được với ảnh thật sự trống
        pred = model.predict(image, conf=0.05, device=device, verbose=False)[0]
        confs = pred.boxes.conf.cpu().numpy()
        max_conf = float(confs.max()) if len(confs) else None
        rows.append({
            "image": path.name,
            **health_metrics(image),
            "max_conf": max_conf,
            "boxes_conf_ge_0.25": int((confs >= 0.25).sum()),
            # least confidence; ảnh không có box nào -> 0 (ảnh trống, không phải ảnh khó)
            "uncertainty": 0.0 if max_conf is None else 1.0 - max_conf,
        })
    yaml_path = cond_dir / "data.yaml"
    yaml_path.write_text(json.dumps({"path": str(cond_dir), "train": "images", "val": "images", "names": names}))
    metrics = model.val(data=str(yaml_path), imgsz=640, batch=batch, device=device, plots=False,
                        verbose=False, project=str(cond_dir / "val"), name="run", exist_ok=True)
    return rows, {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)}


# ---------------------------------------------------------------- tổng hợp
def topk_share(per_image: pd.DataFrame, family: str, n: int) -> list[dict]:
    """Trộn baseline + 3 mức lỗi của một họ thành một pool, xếp theo uncertainty như khi chọn
    ảnh để gán nhãn, lấy top-n. Nếu uncertainty không liên quan tới lỗi camera, mỗi mức ~25 %."""
    pool = per_image[(per_image.family == family) | (per_image.condition == "clean")]
    top = pool.sort_values("uncertainty", ascending=False).head(n)
    share = top.condition.value_counts(normalize=True)
    return [{"family": family, "condition": c, "share_of_topk": float(share.get(c, 0.0))}
            for c in pool.condition.unique()]


def plot_family(summary: pd.DataFrame, family: str, out: Path) -> None:
    rows = summary[(summary.family == family) | (summary.condition == "clean")].sort_values("rank")
    labels = rows.condition.str.replace(f"{family}_", "", regex=False)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, col, title in zip(axes, ["blur_score", "saturation_ratio", "mAP50", "uncertainty"],
                              ["Blur score (var Laplacian)", "Saturation ratio", "mAP@0.5 (YOLOv8n)",
                               "Uncertainty (1 − max conf)"]):
        ax.plot(range(len(rows)), rows[col], marker="o")
        ax.set_xticks(range(len(rows)), labels, rotation=20)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{family}: mức lỗi tăng từ trái sang phải (n = {int(rows.n_images.iloc[0])} ảnh/điều kiện)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def example_grid(work: Path, conditions: list[str], image_name: str, out: Path) -> None:
    tiles = []
    for c in conditions:
        img = cv2.resize(cv2.imread(str(work / c / "images" / image_name)), (426, 240))
        cv2.putText(img, c, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(out), grid)


def run_benchmark(n: int = 200, seed: int = 0, families: list[str] | None = None, night: bool = True,
                  device=None, batch: int = 16, tag: str = "main", bdd_root: Path = BDD100K_ROOT,
                  weights: str = YOLO_WEIGHTS, command: str | None = None) -> dict:
    """Chạy toàn bộ benchmark, ghi kết quả vào results/<tag>/ và trả về các bảng (dùng được từ notebook)."""
    families = families or DEFAULT_FAMILIES
    out_dir = ROOT / "results" / tag
    work = ROOT / "work" / f"{tag}_n{n}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    day = load_split(bdd_root, "daytime", n, seed)
    day_boxes = write_labels(day, work / "labels_day")
    # (tên, họ, giá trị tham số, thứ hạng mức lỗi 0..3, severity theo S5)
    conditions = [("clean", None, 0.0, 0, None)]
    for fam in families:
        param, levels = FAMILIES[fam]
        conditions += [(f"{fam}_{param}{lv:g}", fam, lv, i + 1, sev) for i, (lv, sev) in enumerate(levels)]

    model = YOLO(str(weights))
    names = {int(k): v for k, v in model.names.items()}
    per_image, summary = [], []

    def run(name, samples, label_dir, family, level, rank, s5_severity):
        cond_dir = build_condition(name, samples, bdd_root, label_dir, work, family, level, seed)
        rows, maps = evaluate(model, cond_dir, names, device, batch)
        df = pd.DataFrame(rows).assign(condition=name, family=family or "none", level=level, rank=rank)
        per_image.append(df)
        summary.append({
            "condition": name, "family": family or "none", "level": level, "rank": rank, "s5_severity": s5_severity,
            "n_images": len(df),
            "blur_score": df.blur_score.median(), "saturation_ratio": df.saturation_ratio.mean(),
            "entropy_bits": df.entropy_bits.mean(),
            "noise_est": df.noise_est.median(),
            "mean_max_conf": df.max_conf.mean(), "frac_no_box": df.max_conf.isna().mean(),
            "boxes_conf_ge_0.25": df["boxes_conf_ge_0.25"].mean(),
            "uncertainty": df.uncertainty.mean(), **maps,
        })
        print(f"[{time.time() - t0:6.0f}s] {name:24} blur={summary[-1]['blur_score']:8.1f} "
              f"sat={summary[-1]['saturation_ratio']:.3f} mAP50={maps['mAP50']:.3f} "
              f"unc={summary[-1]['uncertainty']:.3f}", flush=True)

    for name, family, level, rank, s5_severity in conditions:
        run(name, day, work / "labels_day", family, level, rank, s5_severity)
    night_boxes = 0
    if night:  # lỗi THẬT, không mô phỏng — nhưng là ảnh khác, không cùng baseline
        night_samples = load_split(bdd_root, "night", n, seed)
        night_boxes = write_labels(night_samples, work / "labels_night")
        run("night_real", night_samples, work / "labels_night", "night_real", 0.0, 0, None)

    per_image_df = pd.concat(per_image, ignore_index=True)
    summary_df = pd.DataFrame(summary)
    # RCE theo S5: suy giảm tương đối so với baseline sạch (ảnh đêm khác ảnh nên RCE của nó chỉ để tham khảo)
    clean_map = summary_df.loc[summary_df.condition == "clean", "mAP50"].iloc[0]
    summary_df["RCE"] = (clean_map - summary_df.mAP50) / clean_map
    per_image_df.to_csv(out_dir / "per_image.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    topk = pd.DataFrame([r for fam in families for r in topk_share(per_image_df, fam, n)])
    topk.to_csv(out_dir / "topk_share.csv", index=False, float_format="%.3f")
    for fam in families:
        plot_family(summary_df, fam, out_dir / f"plot_{fam}.png")
    example_grid(work, [c[0] for c in conditions], Path(day[0]["filepath"]).name, out_dir / "examples.jpg")

    run_info = {
        "command": command or f"run_benchmark(n={n}, seed={seed}, families={families}, night={night}, tag={tag!r})",
        "seed": seed, "n_per_condition": n,
        "day_gt_boxes": day_boxes, "night_gt_boxes": night_boxes, "jpeg_quality": JPEG_QUALITY,
        "families": {f: FAMILIES[f] for f in families}, "weights": str(weights),
        "versions": {"python": platform.python_version(), "ultralytics": ultralytics.__version__,
                     "opencv": cv2.__version__, "numpy": np.__version__},
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False))
    print(f"Xong sau {run_info['seconds']} s -> {out_dir.relative_to(ROOT)}")
    return {"summary": summary_df, "per_image": per_image_df, "topk": topk, "out_dir": out_dir, "info": run_info}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bdd-root", type=Path, default=BDD100K_ROOT, help="mặc định: $BDD100K_ROOT")
    p.add_argument("--weights", default=YOLO_WEIGHTS, help="mặc định: $YOLO_WEIGHTS hoặc yolov8n.pt")
    p.add_argument("--n", type=int, default=200, help="số ảnh mỗi điều kiện")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES, choices=list(FAMILIES))
    p.add_argument("--no-night", action="store_true", help="bỏ lát cắt ban đêm thật của BDD100K")
    p.add_argument("--device", default=None, help="vd. 0 hoặc cpu; mặc định để ultralytics tự chọn")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--tag", default="main", help="tên thư mục kết quả trong results/")
    p.add_argument("--smoke", action="store_true", help="chạy thử: ghi vào results/smoke")
    args = p.parse_args()
    result = run_benchmark(n=args.n, seed=args.seed, families=args.families, night=not args.no_night,
                           device=args.device, batch=args.batch, tag="smoke" if args.smoke else args.tag,
                           bdd_root=args.bdd_root, weights=args.weights, command=" ".join(sys.argv))
    print(result["summary"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nTop-k theo uncertainty:\n" + result["topk"].to_string(index=False))


if __name__ == "__main__":
    main()
