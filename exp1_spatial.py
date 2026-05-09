"""
实验一 直方图均衡及空间滤波
"""

import os, numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import MAIN_IMAGE, LOW_CONTRAST_IMAGE, EXTRA_IMAGE

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


# ---------- 1. 直方图均衡 ----------

def hist_equalize(img):
    h, w = img.shape
    total = h * w

    hist = np.zeros(256, dtype=np.int64)
    for v in img.ravel():
        hist[v] += 1

    cdf = np.cumsum(hist).astype(np.float64)
    cdf_min = cdf[cdf > 0].min()

    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255).astype(np.uint8)
    return lut[img]


def run_hist_eq():
    print("\n[1.1] Histogram Equalization")

    for name, path in [("low_contrast", f"images/{LOW_CONTRAST_IMAGE}"),
                        ("main", f"images/{MAIN_IMAGE}")]:
        img = load_gray(path)
        eq_my = hist_equalize(img)
        eq_cv = cv2.equalizeHist(img)

        fig, ax = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f"Histogram Equalization - {name}", fontsize=14, fontweight="bold")

        ax[0,0].imshow(img, cmap="gray", vmin=0, vmax=255)
        ax[0,0].set_title("Original"); ax[0,0].axis("off")
        ax[0,1].imshow(eq_my, cmap="gray", vmin=0, vmax=255)
        ax[0,1].set_title("My impl"); ax[0,1].axis("off")
        ax[0,2].imshow(eq_cv, cmap="gray", vmin=0, vmax=255)
        ax[0,2].set_title("cv2.equalizeHist"); ax[0,2].axis("off")

        ax[1,0].hist(img.ravel(), 256, range=(0,256), color="steelblue", alpha=.7)
        ax[1,0].set_title("Hist (original)"); ax[1,0].set_xlim(0,256)
        ax[1,1].hist(eq_my.ravel(), 256, range=(0,256), color="coral", alpha=.7)
        ax[1,1].set_title("Hist (my impl)"); ax[1,1].set_xlim(0,256)
        ax[1,2].hist(eq_cv.ravel(), 256, range=(0,256), color="mediumseagreen", alpha=.7)
        ax[1,2].set_title("Hist (OpenCV)"); ax[1,2].set_xlim(0,256)

        plt.tight_layout()
        savefig(fig, f"hist_eq_{name}.png")


# ---------- 2. 平滑滤波 ----------

def mean_filter(img, k=3):
    p = k // 2
    pad = np.pad(img, p, mode="reflect").astype(np.float64)
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.mean(pad[i:i+k, j:j+k])
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_filter(img, k=5, sigma=1.0):
    ax = np.arange(k) - k//2
    xx, yy = np.meshgrid(ax, ax)
    kern = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    kern /= kern.sum()

    p = k // 2
    pad = np.pad(img, p, mode="reflect").astype(np.float64)
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.sum(pad[i:i+k, j:j+k] * kern)
    return np.clip(out, 0, 255).astype(np.uint8)


def median_filter(img, k=3):
    p = k // 2
    pad = np.pad(img, p, mode="reflect")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.median(pad[i:i+k, j:j+k])
    return out.astype(np.uint8)


def max_filter(img, k=3):
    p = k // 2
    pad = np.pad(img, p, mode="reflect")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.max(pad[i:i+k, j:j+k])
    return out.astype(np.uint8)


def min_filter(img, k=3):
    p = k // 2
    pad = np.pad(img, p, mode="reflect")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.min(pad[i:i+k, j:j+k])
    return out.astype(np.uint8)


def run_smoothing():
    print("\n[1.2] Smoothing Filters")

    stem = os.path.splitext(MAIN_IMAGE)[0]
    ref = load_gray(f"images/{MAIN_IMAGE}")

    # 线性平滑 -> 高斯噪声
    noisy = load_gray(f"images/{stem}_gaussian.png")
    m3 = mean_filter(noisy, 3)
    m5 = mean_filter(noisy, 5)
    g1 = gaussian_filter(noisy, 5, 1.0)
    g2 = gaussian_filter(noisy, 5, 2.0)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Linear Smoothing (Gaussian noise)", fontsize=14, fontweight="bold")
    for idx, (im, t) in enumerate([
        (noisy, "Noisy"), (m3, "Mean 3x3"), (m5, "Mean 5x5"),
        (ref, "Original"), (g1, r"Gaussian $\sigma$=1"), (g2, r"Gaussian $\sigma$=2"),
    ]):
        ax[idx//3, idx%3].imshow(im, cmap="gray")
        ax[idx//3, idx%3].set_title(t); ax[idx//3, idx%3].axis("off")
    plt.tight_layout()
    savefig(fig, "smooth_linear.png")

    # 顺序统计 -> 椒盐噪声
    sp = load_gray(f"images/{stem}_salt_pepper.png")
    med3 = median_filter(sp, 3)
    med5 = median_filter(sp, 5)
    mx = max_filter(sp, 3)
    mn = min_filter(sp, 3)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Order-Statistic Filtering (S&P noise)", fontsize=14, fontweight="bold")
    for idx, (im, t) in enumerate([
        (sp, "Salt & Pepper"), (med3, "Median 3x3"), (med5, "Median 5x5"),
        (ref, "Original"), (mx, "Max 3x3"), (mn, "Min 3x3"),
    ]):
        ax[idx//3, idx%3].imshow(im, cmap="gray")
        ax[idx//3, idx%3].set_title(t); ax[idx//3, idx%3].axis("off")
    plt.tight_layout()
    savefig(fig, "smooth_order_stat.png")

    # PSNR
    print("\n  PSNR (dB)")
    for tag, res in [("noisy(gauss)", noisy), ("mean3", m3), ("mean5", m5),
                     ("gauss_s1", g1), ("gauss_s2", g2)]:
        print(f"    {tag:<15} {cv2.PSNR(ref, res):.2f}")
    for tag, res in [("noisy(sp)", sp), ("median3", med3), ("median5", med5),
                     ("max3", mx), ("min3", mn)]:
        print(f"    {tag:<15} {cv2.PSNR(ref, res):.2f}")


# ---------- 3. 锐化滤波 ----------

def conv2d(img, kernel):
    """手动卷积"""
    k = kernel.shape[0]
    p = k // 2
    pad = np.pad(img, p, mode="reflect").astype(np.float64)
    out = np.zeros(img.shape, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.sum(pad[i:i+k, j:j+k] * kernel)
    return out


def to_uint8(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def laplacian_sharpen(img):
    kern = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
    lap = conv2d(img, kern)
    sharp = np.clip(img.astype(np.float64) - lap, 0, 255).astype(np.uint8)
    return to_uint8(lap), sharp


def sobel_gradient(img):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float64)
    gx = conv2d(img, kx)
    gy = conv2d(img, ky)
    mag = np.abs(gx) + np.abs(gy)
    return to_uint8(gx), to_uint8(gy), to_uint8(mag), mag


def prewitt_gradient(img):
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float64)
    ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float64)
    gx = conv2d(img, kx)
    gy = conv2d(img, ky)
    mag = np.abs(gx) + np.abs(gy)
    return to_uint8(gx), to_uint8(gy), to_uint8(mag), mag


def roberts_gradient(img):
    f = img.astype(np.float64)
    h, w = f.shape
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    # Roberts 2x2 交叉差分
    for i in range(h-1):
        for j in range(w-1):
            gx[i,j] = f[i,j] - f[i+1,j+1]
            gy[i,j] = f[i,j+1] - f[i+1,j]
    mag = np.abs(gx) + np.abs(gy)
    return to_uint8(gx), to_uint8(gy), to_uint8(mag), mag


def sharpen_with_gradient(img, mag_raw):
    """用梯度做锐化增强"""
    scale = mag_raw / (mag_raw.max() + 1e-8) * 128
    out = np.clip(img.astype(np.float64) + scale, 0, 255)
    return out.astype(np.uint8)


def run_sharpening():
    print("\n[1.3] Sharpening Filters")

    img = load_gray(f"images/{LOW_CONTRAST_IMAGE}")

    # 拉普拉斯
    lap_vis, lap_sharp = laplacian_sharpen(img)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Laplacian Sharpening", fontsize=14, fontweight="bold")
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(lap_vis, cmap="gray"); ax[1].set_title(r"$\nabla^2 f$"); ax[1].axis("off")
    ax[2].imshow(lap_sharp, cmap="gray"); ax[2].set_title(r"$f - \nabla^2 f$"); ax[2].axis("off")
    plt.tight_layout()
    savefig(fig, "sharpen_laplacian.png")

    # 三种一阶微分
    s_gx, s_gy, s_mag, s_raw = sobel_gradient(img)
    p_gx, p_gy, p_mag, p_raw = prewitt_gradient(img)
    r_gx, r_gy, r_mag, r_raw = roberts_gradient(img)

    s_sharp = sharpen_with_gradient(img, s_raw)
    p_sharp = sharpen_with_gradient(img, p_raw)
    r_sharp = sharpen_with_gradient(img, r_raw)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("First-Order Gradient & Sharpening", fontsize=14, fontweight="bold")
    ax[0,0].imshow(s_mag, cmap="gray"); ax[0,0].set_title("Sobel |G|"); ax[0,0].axis("off")
    ax[0,1].imshow(p_mag, cmap="gray"); ax[0,1].set_title("Prewitt |G|"); ax[0,1].axis("off")
    ax[0,2].imshow(r_mag, cmap="gray"); ax[0,2].set_title("Roberts |G|"); ax[0,2].axis("off")
    ax[1,0].imshow(s_sharp, cmap="gray"); ax[1,0].set_title("Sobel sharpened"); ax[1,0].axis("off")
    ax[1,1].imshow(p_sharp, cmap="gray"); ax[1,1].set_title("Prewitt sharpened"); ax[1,1].axis("off")
    ax[1,2].imshow(r_sharp, cmap="gray"); ax[1,2].set_title("Roberts sharpened"); ax[1,2].axis("off")
    plt.tight_layout()
    savefig(fig, "sharpen_first_order.png")

    # Sobel分量细节
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Sobel Components", fontsize=13)
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(s_gx, cmap="gray"); ax[1].set_title("Gx"); ax[1].axis("off")
    ax[2].imshow(s_gy, cmap="gray"); ax[2].set_title("Gy"); ax[2].axis("off")
    ax[3].imshow(s_mag, cmap="gray"); ax[3].set_title("|Gx|+|Gy|"); ax[3].axis("off")
    plt.tight_layout()
    savefig(fig, "sharpen_sobel_detail.png")

    # astronaut 多方法对比
    img2 = load_gray(f"images/{EXTRA_IMAGE}")
    l2_v, l2_s = laplacian_sharpen(img2)
    _, _, sm2, sr2 = sobel_gradient(img2)
    _, _, pm2, pr2 = prewitt_gradient(img2)
    _, _, rm2, rr2 = roberts_gradient(img2)

    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    fig.suptitle("Sharpening Comparison - astronaut", fontsize=14, fontweight="bold")
    ax[0,0].imshow(img2, cmap="gray"); ax[0,0].set_title("Original"); ax[0,0].axis("off")
    ax[0,1].imshow(l2_v, cmap="gray"); ax[0,1].set_title("Laplacian"); ax[0,1].axis("off")
    ax[0,2].imshow(sm2, cmap="gray"); ax[0,2].set_title("Sobel |G|"); ax[0,2].axis("off")
    ax[0,3].imshow(pm2, cmap="gray"); ax[0,3].set_title("Prewitt |G|"); ax[0,3].axis("off")
    ax[1,0].imshow(rm2, cmap="gray"); ax[1,0].set_title("Roberts |G|"); ax[1,0].axis("off")
    ax[1,1].imshow(l2_s, cmap="gray"); ax[1,1].set_title("Lap sharp"); ax[1,1].axis("off")
    ax[1,2].imshow(sharpen_with_gradient(img2, sr2), cmap="gray")
    ax[1,2].set_title("Sobel sharp"); ax[1,2].axis("off")
    ax[1,3].imshow(sharpen_with_gradient(img2, pr2), cmap="gray")
    ax[1,3].set_title("Prewitt sharp"); ax[1,3].axis("off")
    plt.tight_layout()
    savefig(fig, "sharpen_astronaut.png")


if __name__ == "__main__":
    print("=" * 50)
    print("  Exp1: Histogram Equalization & Spatial Filtering")
    print("=" * 50)
    run_hist_eq()
    run_smoothing()
    run_sharpening()
    print("\ndone.")
