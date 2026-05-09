# 视觉计算课程设计

SCAI011112 图像处理实验：直方图均衡、空间滤波、频域滤波。

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | **图片配置文件**，在这里更改所选图片文件名 |
| `prepare_images.py` | 准备测试图像，对主图生成椒盐噪声和高斯噪声版本 |
| `exp1_spatial.py` | 实验一：直方图均衡化、线性平滑/顺序统计滤波、锐化滤波（Laplacian + Sobel/Prewitt/Roberts） |
| `exp2_frequency.py` | 实验二：频域低通滤波（ILPF/BLPF/GLPF）和高通滤波（IHPF/BHPF/GHPF） |
| `requirements.txt` | Python 依赖包列表 |

## 环境配置

Windows + VSCode + Python 3.11

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 使用自己的图片

打开 `config.py`，把文件名改成你自己的：

```python
MAIN_IMAGE = "my_photo.png"          # 随便叫什么都行
LOW_CONTRAST_IMAGE = "dark_scene.jpg"
EXTRA_IMAGE = "building.png"
```

然后把对应的图片文件丢到 `images/` 目录下，运行 `prepare_images.py` 即可。彩色图会自动转灰度。

如果你不放自己的图片，默认会从 scikit-image 下载 camera/moon/astronaut。

## 运行

```powershell
python prepare_images.py    # 先跑这个
python exp1_spatial.py       # 实验一
python exp2_frequency.py     # 实验二
```

## 输出结果说明

所有结果图保存在 `results/` 目录下：

### 实验一：空间域

| 文件 | 内容 |
|------|------|
| `hist_eq_low_contrast.png` | 低对比度图的直方图均衡前后对比 + 直方图变化 |
| `hist_eq_main.png` | 主图的直方图均衡前后对比 + 直方图变化 |
| `smooth_linear.png` | 均值滤波(3x3, 5x5)和高斯滤波(σ=1, σ=2)处理高斯噪声的效果 |
| `smooth_order_stat.png` | 中值滤波(3x3, 5x5)、最大值、最小值滤波处理椒盐噪声的效果 |
| `sharpen_laplacian.png` | 拉普拉斯算子响应 + 锐化结果 |
| `sharpen_first_order.png` | Sobel / Prewitt / Roberts 三种一阶微分的梯度幅度和锐化对比 |
| `sharpen_sobel_detail.png` | Sobel 的 Gx、Gy 分量和梯度幅度细节 |
| `sharpen_astronaut.png` | 四种锐化方法在 astronaut 图上的综合对比 |

### 实验二：频域

| 文件 | 内容 |
|------|------|
| `freq_lowpass_ILPF.png` | 理想低通在 D0=20/50/100 下的滤波结果和频谱 |
| `freq_lowpass_BLPF.png` | 巴特沃斯低通在 D0=20/50/100 下的滤波结果和频谱 |
| `freq_lowpass_GLPF.png` | 高斯低通在 D0=20/50/100 下的滤波结果和频谱 |
| `freq_lowpass_compare.png` | 三种低通滤波器在 D0=50 下的横向对比（含滤波器可视化） |
| `freq_highpass_IHPF.png` | 理想高通在 D0=10/30/60 下的滤波结果和频谱 |
| `freq_highpass_BHPF.png` | 巴特沃斯高通在 D0=10/30/60 下的滤波结果和频谱 |
| `freq_highpass_GHPF.png` | 高斯高通在 D0=10/30/60 下的滤波结果和频谱 |
| `freq_highpass_compare.png` | 三种高通滤波器在 D0=30 下的横向对比（含滤波器可视化） |
| `freq_highboost.png` | 高频提升滤波效果，与空域 Sobel 边缘做参考对比 |

## 技术栈

- Python 3.11
- NumPy — 矩阵运算和 FFT
- OpenCV — 图像读写和对照验证
- Matplotlib — 结果可视化
- scikit-image — 提供默认测试图像
