"""
在这里配置实验用的图片文件名
如果你要用自己的图片, 只需要改下面的文件名即可
图片放到 images/ 目录下, 要求是灰度图(或者会被自动转灰度)
"""

# 滤波实验主图 (会对这张图加噪声做平滑/频域实验)
MAIN_IMAGE = "test_photo.png"

# 锐化/直方图均衡用的低对比度图
LOW_CONTRAST_IMAGE = "moon.png"

# 额外的对比测试图
EXTRA_IMAGE = "astronaut.png"
