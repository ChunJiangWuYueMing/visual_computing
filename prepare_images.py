"""
准备实验用的测试图像
生成带噪声的版本用于滤波实验
"""
import os
import numpy as np
import cv2
from config import MAIN_IMAGE, LOW_CONTRAST_IMAGE, EXTRA_IMAGE

IMG_DIR = "images"


def add_salt_pepper(img, ratio=0.02, seed=42):
    rng = np.random.default_rng(seed)
    out = img.copy().astype(np.float64)
    out[rng.random(img.shape) < ratio] = 255
    out[rng.random(img.shape) < ratio] = 0
    return out.astype(np.uint8)


def add_gaussian_noise(img, sigma=25, seed=42):
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def ensure_gray(path):
    """读入图片, 如果是彩色就自动转灰度"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, img)  # 覆盖保存为灰度版
        print(f"  {os.path.basename(path)} 已自动转为灰度")
    return img


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    needed = {MAIN_IMAGE, LOW_CONTRAST_IMAGE, EXTRA_IMAGE}
    existing = set(os.listdir(IMG_DIR))

    # 缺少的图片从 scikit-image 下载
    missing = needed - existing
    if missing:
        from skimage import data
        from skimage.color import rgb2gray
        from skimage.util import img_as_ubyte

        defaults = {
            "camera.png": lambda: data.camera(),
            "moon.png": lambda: data.moon(),
            "astronaut.png": lambda: img_as_ubyte(rgb2gray(data.astronaut())),
        }
        for fname in missing:
            if fname in defaults:
                img = defaults[fname]()
                cv2.imwrite(os.path.join(IMG_DIR, fname), img)
                print(f"  下载默认图: {fname}")
            else:
                print(f"  WARNING: {fname} 不在默认库中, 请手动放到 images/ 下")

    # 确保全部是灰度
    for fname in needed:
        p = os.path.join(IMG_DIR, fname)
        if os.path.exists(p):
            ensure_gray(p)

    # 对主图生成噪声版本
    main_path = os.path.join(IMG_DIR, MAIN_IMAGE)
    camera = cv2.imread(main_path, cv2.IMREAD_GRAYSCALE)
    if camera is None:
        print(f"ERROR: {main_path} 不存在, 无法生成噪声图")
        return

    stem = os.path.splitext(MAIN_IMAGE)[0]  # 去掉扩展名
    sp = add_salt_pepper(camera)
    cv2.imwrite(os.path.join(IMG_DIR, f"{stem}_salt_pepper.png"), sp)
    print(f"  {stem}_salt_pepper.png")

    gs = add_gaussian_noise(camera)
    cv2.imwrite(os.path.join(IMG_DIR, f"{stem}_gaussian.png"), gs)
    print(f"  {stem}_gaussian.png")

    print("图像准备完毕")


if __name__ == "__main__":
    main()
