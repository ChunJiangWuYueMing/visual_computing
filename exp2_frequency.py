"""
实验二 频域滤波
"""

import os, numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import MAIN_IMAGE

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def savefig(fig, name):
    fig.savefig(os.path.join(RESULTS, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


# ---------- DFT 工具 ----------

def dft2(img):
    f = np.fft.fft2(img.astype(np.float64))
    return np.fft.fftshift(f)

def idft2(fshift):
    f = np.fft.ifftshift(fshift)
    out = np.fft.ifft2(f)
    return np.clip(np.real(out), 0, 255).astype(np.uint8)

def spectrum(fshift):
    mag = np.log1p(np.abs(fshift))
    return (mag / mag.max() * 255).astype(np.uint8)

def dist_matrix(rows, cols):
    u = np.arange(rows).reshape(-1,1) - rows//2
    v = np.arange(cols).reshape(1,-1) - cols//2
    return np.sqrt(u**2 + v**2)


# ---------- 低通滤波器 ----------

def ilpf(rows, cols, d0):
    D = dist_matrix(rows, cols)
    H = np.zeros((rows, cols))
    H[D <= d0] = 1.0
    return H

def blpf(rows, cols, d0, n=2):
    D = dist_matrix(rows, cols)
    return 1.0 / (1.0 + (D/d0)**(2*n))

def glpf(rows, cols, d0):
    D = dist_matrix(rows, cols)
    return np.exp(-D**2 / (2*d0**2))


# ---------- 高通滤波器 = 1 - 对应低通 ----------

def ihpf(rows, cols, d0):
    return 1.0 - ilpf(rows, cols, d0)

def bhpf(rows, cols, d0, n=2):
    return 1.0 - blpf(rows, cols, d0, n)

def ghpf(rows, cols, d0):
    return 1.0 - glpf(rows, cols, d0)


# ---------- 频域滤波流程 ----------

def freq_filter(img, H):
    F = dft2(img)
    G = F * H
    return idft2(G), F, G


# ---------- 实验 ----------

def run_lowpass():
    print("\n[2.1] Frequency-domain Lowpass Filtering")

    img = load_gray(f"images/{MAIN_IMAGE}")
    rows, cols = img.shape
    cutoffs = [20, 50, 100]

    filters = [
        ("ILPF", ilpf),
        ("BLPF", blpf),
        ("GLPF", glpf),
    ]

    for fname, ffunc in filters:
        fig, ax = plt.subplots(2, len(cutoffs)+1, figsize=(17, 8))
        fig.suptitle(f"{fname} - different cutoff", fontsize=14, fontweight="bold")

        F0 = dft2(img)
        ax[0,0].imshow(img, cmap="gray"); ax[0,0].set_title("Original"); ax[0,0].axis("off")
        ax[1,0].imshow(spectrum(F0), cmap="gray"); ax[1,0].set_title("Spectrum"); ax[1,0].axis("off")

        for i, d0 in enumerate(cutoffs):
            H = ffunc(rows, cols, d0, 2) if fname == "BLPF" else ffunc(rows, cols, d0)
            result, _, G = freq_filter(img, H)
            ax[0,i+1].imshow(result, cmap="gray")
            ax[0,i+1].set_title(f"$D_0$ = {d0}"); ax[0,i+1].axis("off")
            ax[1,i+1].imshow(spectrum(G), cmap="gray")
            ax[1,i+1].set_title(f"Filtered spectrum"); ax[1,i+1].axis("off")

        plt.tight_layout()
        savefig(fig, f"freq_lowpass_{fname}.png")

    # 三种低通在D0=50下的对比
    d0 = 50
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"Lowpass Comparison ($D_0$ = {d0})", fontsize=14, fontweight="bold")

    ax[0,0].imshow(img, cmap="gray"); ax[0,0].set_title("Original"); ax[0,0].axis("off")
    ax[1,0].imshow(spectrum(dft2(img)), cmap="gray"); ax[1,0].set_title("Spectrum"); ax[1,0].axis("off")

    lp_list = [
        ("ILPF", ilpf(rows, cols, d0)),
        ("BLPF (n=2)", blpf(rows, cols, d0, 2)),
        ("GLPF", glpf(rows, cols, d0)),
    ]
    for i, (n, H) in enumerate(lp_list):
        res, _, _ = freq_filter(img, H)
        ax[0,i+1].imshow(res, cmap="gray"); ax[0,i+1].set_title(f"{n} result"); ax[0,i+1].axis("off")
        ax[1,i+1].imshow(H, cmap="gray"); ax[1,i+1].set_title(f"{n} filter"); ax[1,i+1].axis("off")

    plt.tight_layout()
    savefig(fig, "freq_lowpass_compare.png")


def run_highpass():
    print("\n[2.2] Frequency-domain Highpass Filtering")

    img = load_gray(f"images/{MAIN_IMAGE}")
    rows, cols = img.shape
    cutoffs = [10, 30, 60]

    filters = [
        ("IHPF", ihpf),
        ("BHPF", bhpf),
        ("GHPF", ghpf),
    ]

    for fname, ffunc in filters:
        fig, ax = plt.subplots(2, len(cutoffs)+1, figsize=(17, 8))
        fig.suptitle(f"{fname} - different cutoff", fontsize=14, fontweight="bold")

        F0 = dft2(img)
        ax[0,0].imshow(img, cmap="gray"); ax[0,0].set_title("Original"); ax[0,0].axis("off")
        ax[1,0].imshow(spectrum(F0), cmap="gray"); ax[1,0].set_title("Spectrum"); ax[1,0].axis("off")

        for i, d0 in enumerate(cutoffs):
            H = ffunc(rows, cols, d0, 2) if fname == "BHPF" else ffunc(rows, cols, d0)
            result, _, G = freq_filter(img, H)
            ax[0,i+1].imshow(result, cmap="gray")
            ax[0,i+1].set_title(f"$D_0$ = {d0}"); ax[0,i+1].axis("off")
            ax[1,i+1].imshow(spectrum(G), cmap="gray")
            ax[1,i+1].set_title(f"Filtered spectrum"); ax[1,i+1].axis("off")

        plt.tight_layout()
        savefig(fig, f"freq_highpass_{fname}.png")

    # 三种高通对比 D0=30
    d0 = 30
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"Highpass Comparison ($D_0$ = {d0})", fontsize=14, fontweight="bold")

    ax[0,0].imshow(img, cmap="gray"); ax[0,0].set_title("Original"); ax[0,0].axis("off")
    ax[1,0].imshow(spectrum(dft2(img)), cmap="gray"); ax[1,0].set_title("Spectrum"); ax[1,0].axis("off")

    hp_list = [
        ("IHPF", ihpf(rows, cols, d0)),
        ("BHPF (n=2)", bhpf(rows, cols, d0, 2)),
        ("GHPF", ghpf(rows, cols, d0)),
    ]
    for i, (n, H) in enumerate(hp_list):
        res, _, _ = freq_filter(img, H)
        ax[0,i+1].imshow(res, cmap="gray"); ax[0,i+1].set_title(f"{n} result"); ax[0,i+1].axis("off")
        ax[1,i+1].imshow(H, cmap="gray"); ax[1,i+1].set_title(f"{n} filter"); ax[1,i+1].axis("off")

    plt.tight_layout()
    savefig(fig, "freq_highpass_compare.png")

    # 高频提升
    k = 1.5
    H_boost = (k - 1) + ghpf(rows, cols, 30)
    res_boost, _, _ = freq_filter(img, H_boost)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("High-Boost Filtering", fontsize=14, fontweight="bold")
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(res_boost, cmap="gray"); ax[1].set_title(f"High-boost k={k}"); ax[1].axis("off")

    # 空域Sobel做参考
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.clip(np.abs(sx)+np.abs(sy), 0, 255).astype(np.uint8)
    ax[2].imshow(edge, cmap="gray"); ax[2].set_title("Sobel edge (spatial ref)"); ax[2].axis("off")
    plt.tight_layout()
    savefig(fig, "freq_highboost.png")


if __name__ == "__main__":
    print("=" * 50)
    print("  Exp2: Frequency-domain Filtering")
    print("=" * 50)
    run_lowpass()
    run_highpass()
    print("\ndone.")
