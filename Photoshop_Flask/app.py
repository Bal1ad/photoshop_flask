import os
import uuid
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
from scipy.signal import convolve2d
from skimage.restoration import richardson_lucy
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import kornia

# # 加载模型，只加载一次，避免每次请求都反复加载
# beauty_model = torch.load('models/GFPGANv1.pth', map_location='cpu').eval()
# style_model = torch.load('models/GFPGANv1.3.pth', map_location='cpu').eval()

# 全局提前加载 Mask R-CNN 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn_model.to(device)
maskrcnn_model.eval()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('file')
    filenames = []
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            filenames.append(filename)
    return jsonify({'filenames': filenames})

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

def get_unique_filename(prefix, ext=".png"):
    return f"{prefix}_{uuid.uuid4().hex}{ext}"

# ----------- 基础图像处理函数 -----------
# 灰度转化
def gray_transform(img, method):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'linear':
        result = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    elif method == 'log':
        c = 255 / np.log(1 + np.max(gray))
        result = c * (np.log(1 + gray.astype(np.float64)))
        result = np.uint8(np.clip(result, 0, 255))
    elif method == 'exp':
        c = 255 / (np.exp(1) - 1)
        result = c * (np.exp(gray / 255.0) - 1)
        result = np.uint8(np.clip(result, 0, 255))
    else:
        result = gray
    return result

# 二值图
def binary(img, threshold=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return result

# 直方图均衡化
def hist_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.equalizeHist(gray)
    return result

# 平滑处理
def smooth(img, method):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'mean':
        result = cv2.blur(gray, (5, 5))
    elif method == 'gaussian':
        result = cv2.GaussianBlur(gray, (5, 5), 1)
    else:
        result = gray
    return result

# 锐化处理
def sharpen(img, method):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'roberts':
        kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        img_x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        img_y = cv2.filter2D(gray, cv2.CV_16S, kernely)
        abs_x = cv2.convertScaleAbs(img_x)
        abs_y = cv2.convertScaleAbs(img_y)
        result = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    elif method == 'sobel':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        result = cv2.magnitude(sobelx, sobely)
        result = np.uint8(np.clip(result, 0, 255))
    else:
        result = gray
    return result

# 滤波器
def filter_img(img, method):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'mean':
        result = cv2.blur(gray, (5, 5))
    elif method == 'median':
        result = cv2.medianBlur(gray, 5)
    elif method == 'max':
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.dilate(gray, kernel)
    elif method == 'min':
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.erode(gray, kernel)
    else:
        result = gray
    return result

# 傅里叶变换
def fft_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
    return magnitude_spectrum

# ----------- 低通/高通滤波器 -----------
def get_filter_mask(shape, kind, d0, n=2):
    """生成低通/高通滤波器的掩模"""
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt((U - P // 2) ** 2 + (V - Q // 2) ** 2)
    if kind == 'ilpf':
        H = (D <= d0).astype(np.float32)
    elif kind == 'blpf':
        H = 1 / (1 + (D / d0) ** (2 * n))
    elif kind == 'glpf':
        H = np.exp(-(D**2) / (2 * (d0**2)))
    elif kind == 'ihpf':
        H = 1 - (D <= d0).astype(np.float32)
    elif kind == 'bhpf':
        H = 1 - 1 / (1 + (D / d0) ** (2 * n))
    elif kind == 'ghpf':
        H = 1 - np.exp(-(D**2) / (2 * (d0**2)))
    else:
        H = np.ones_like(D, dtype=np.float32)
    return H

def freq_filter(img, H):
    """频域滤波（修复全白问题）"""
    img_float = img.astype(np.float32)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=(0, 1))

    # 应用滤波器（同时处理实部和虚部）
    dft_shift[:, :, 0] *= H
    dft_shift[:, :, 1] *= H

    f_ishift = np.fft.ifftshift(dft_shift, axes=(0, 1))
    img_back = cv2.idft(f_ishift)
    mag = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 关键修复：归一化到 [0, 255]
    mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag_normalized.astype(np.uint8)


def ideal_lowpass_filter(shape, d0):
    P, Q = shape
    H = np.zeros((P, Q), dtype=np.float32)
    center = (P // 2, Q // 2)
    for u in range(P):
        for v in range(Q):
            if np.sqrt((u-center[0])**2 + (v-center[1])**2) <= d0:
                H[u, v] = 1
    return H

def butterworth_lowpass_filter(shape, d0, n):
    P, Q = shape
    H = np.zeros((P, Q), dtype=np.float32)
    center = (P // 2, Q // 2)
    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u-center[0])**2 + (v-center[1])**2)
            H[u, v] = 1 / (1 + (D/d0)**(2*n))
    return H

def gaussian_lowpass_filter(shape, d0):
    P, Q = shape
    H = np.zeros((P, Q), dtype=np.float32)
    center = (P // 2, Q // 2)
    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u-center[0])**2 + (v-center[1])**2)
            H[u, v] = np.exp(-(D**2)/(2*(d0**2)))
    return H

def ideal_highpass_filter(shape, d0):
    return 1 - ideal_lowpass_filter(shape, d0)

def butterworth_highpass_filter(shape, d0, n):
    return 1 - butterworth_lowpass_filter(shape, d0, n)

def gaussian_highpass_filter(shape, d0):
    return 1 - gaussian_lowpass_filter(shape, d0)



# ----------- 图像分割 -----------
def segmentation_threshold(img, threshold=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return result

def segmentation_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.Canny(gray, 100, 200)
    return result

def maskrcnn_infer(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_pil).to(device)
    with torch.no_grad():
        outputs = maskrcnn_model([img_tensor])[0]
    masks = outputs['masks']  # (N, 1, H, W)
    scores = outputs['scores']
    threshold = 0.5
    mask_img = img_cv.copy()
    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0].cpu().numpy()
            mask = (mask > 0.5)
            color = np.random.randint(0, 255, (3,))
            mask_img[mask] = mask_img[mask] * 0.4 + color * 0.6
    return mask_img.astype(np.uint8)

# ----------- 拼接与融合（SIFT/SURF+RANSAC）-----------
def sift_ransac_stitch(img1, img2):
    # 检测特征点
    try:
        sift = cv2.SIFT_create()
    except:
        try:
            sift = cv2.xfeatures2d.SIFT_create()
        except:
            sift = None
    if sift is None:
        try:
            sift = cv2.xfeatures2d.SURF_create()
        except:
            raise Exception("SIFT/SURF不可用，请安装opencv-contrib-python")
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 匹配特征点
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        # RANSAC求单应性
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners = np.concatenate(
            (np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2), warped_corners), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation = [-xmin, -ymin]
        H_translation = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])
        result = cv2.warpPerspective(img2, H_translation.dot(H), (xmax-xmin, ymax-ymin))
        result[translation[1]:h1+translation[1], translation[0]:w1+translation[0]] = img1
        return result
    else:
        raise Exception("匹配点太少，无法拼接")

# ----------- 拼接与融合（金字塔融合）-----------
def pyramid_blending(img1, img2, num_levels=4):
    # 1. 统一为三通道、uint8
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    # 2. resize到一致
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    # 3. 构建高斯金字塔
    gpA, gpB = [img1], [img2]
    for i in range(num_levels):
        gpA.append(cv2.pyrDown(gpA[-1]))
        gpB.append(cv2.pyrDown(gpB[-1]))
    # 4. 构建拉普拉斯金字塔
    lpA, lpB = [], []
    for i in range(num_levels):
        size = (gpA[i].shape[1], gpA[i].shape[0])
        GA_up = cv2.pyrUp(gpA[i+1], dstsize=size)
        LA = cv2.subtract(gpA[i], GA_up)
        lpA.append(LA)
        size = (gpB[i].shape[1], gpB[i].shape[0])
        GB_up = cv2.pyrUp(gpB[i+1], dstsize=size)
        LB = cv2.subtract(gpB[i], GB_up)
        lpB.append(LB)
    lpA.append(gpA[-1])
    lpB.append(gpB[-1])
    # 5. 拼接每层
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, :cols // 2], lb[:, cols // 2:]))
        LS.append(ls)
    # 6. 重建
    result = LS[-1]
    for i in range(num_levels-1, -1, -1):
        size = (LS[i].shape[1], LS[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        if result.shape != LS[i].shape:
            result = cv2.resize(result, (LS[i].shape[1], LS[i].shape[0]))
        result = cv2.add(result, LS[i])
    return np.clip(result, 0, 255).astype(np.uint8)


# ----------- 拼接与融合（SuperGlue）-----------
def superglue_stitch(img1, img2, resize_width=800):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 图像预处理（统一尺寸并转为灰度）
    def preprocess_image(img, width):
        h, w = img.shape[:2]
        scale = width / w
        img_resized = cv2.resize(img, (width, int(h * scale)))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        return img_gray, scale

    img1_gray, scale1 = preprocess_image(img1, resize_width)
    img2_gray, scale2 = preprocess_image(img2, resize_width)

    # 2. 使用SuperPoint提取特征
    superpoint = kornia.feature.SuperPoint(1000).to(device)

    # 转换图像为tensor
    def image_to_tensor(img_gray):
        img_tensor = torch.from_numpy(img_gray / 255.).float()[None, None].to(device)
        return img_tensor

    img1_tensor = image_to_tensor(img1_gray)
    img2_tensor = image_to_tensor(img2_gray)

    # 提取特征
    with torch.no_grad():
        features1 = superpoint(img1_tensor)
        features2 = superpoint(img2_tensor)

    # 3. 使用SuperGlue进行特征匹配
    superglue = kornia.feature.SuperGlue().to(device)

    # 构建输入数据
    data = {
        "keypoints0": features1["keypoints"],
        "descriptors0": features1["descriptors"],
        "keypoints1": features2["keypoints"],
        "descriptors1": features2["descriptors"],
        "image0": img1_tensor,
        "image1": img2_tensor,
    }

    # 运行匹配
    with torch.no_grad():
        matches = superglue(data)

    # 4. 获取匹配点对
    mkpts0 = features1["keypoints"][0].cpu().numpy() / scale1
    mkpts1 = features2["keypoints"][0].cpu().numpy() / scale2
    matches = matches["matches0"][0].cpu().numpy()
    confidence = matches["matching_scores0"][0].cpu().numpy()

    # 筛选高置信度匹配
    valid = matches > -1
    mkpts0 = mkpts0[valid]
    mkpts1 = mkpts1[matches[valid]]
    conf = confidence[valid]

    if len(mkpts0) < 4:
        raise ValueError(f"匹配点不足，仅找到 {len(mkpts0)} 对有效匹配")

    # 5. 计算单应性矩阵
    H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)

    # 6. 图像变换与拼接
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 计算拼接后尺寸
    corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_corners = np.concatenate((np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]), warped_corners), axis=0)

    # 计算平移量
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]
    H_trans = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    # 执行透视变换
    result = cv2.warpPerspective(img2, H_trans.dot(H), (xmax - xmin, ymax - ymin))
    result[translation[1]:translation[1] + h1, translation[0]:translation[0] + w1] = img1

    return result

# ----------- 多场景图像复原（传统）  -----------
# 暗通道先验
def get_dark_channel(im, size=15):
    min_img = cv2.min(cv2.min(im[:,:,0], im[:,:,1]), im[:,:,2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    dark = cv2.erode(min_img, kernel)
    return dark

def get_atmosphere(im, dark):
    flat_im = im.reshape(-1, 3)
    flat_dark = dark.flatten()
    search_idx = flat_dark.argsort()[-int(0.001*len(flat_dark)):]
    A = flat_im[search_idx].mean(axis=0)
    return A

def get_transmission(im, A, size=15, omega=0.95):
    normed = im / A
    transmission = 1 - omega * get_dark_channel(normed, size)
    return transmission

def guided_filter(p, I, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r,r))
    mean_Ip = cv2.boxFilter(I*p, cv2.CV_64F, (r,r))
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = cv2.boxFilter(I*I, cv2.CV_64F, (r,r))
    var_I = mean_II - mean_I*mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r,r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r,r))
    q = mean_a*I + mean_b
    return q

def dehaze(img):
    img = img.astype(np.float64)/255
    dark = get_dark_channel(img)
    A = get_atmosphere(img, dark)
    raw_t = get_transmission(img, A)
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)/255
    t = guided_filter(raw_t, gray, 40, 1e-3)
    t = np.clip(t, 0.1, 1)
    J = np.empty_like(img)
    for c in range(3):
        J[:,:,c] = (img[:,:,c] - A[c]) / t + A[c]
    result = np.clip(J*255, 0, 255).astype(np.uint8)
    return result

# Wiener滤波
def wiener_filter(img, kernel_size=5, K=0.01):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 构造 kernel，并用 np.zeros 填充到和图像一样大
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    pad_h = h - kernel_size
    pad_w = w - kernel_size
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    kernel_padded = np.pad(kernel, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')

    # 确保 kernel_padded 和 gray 同 shape
    kernel_padded = kernel_padded[:h, :w]

    # 傅里叶变换
    dummy = np.fft.fft2(gray)
    kernel_fft = np.fft.fft2(kernel_padded)

    # Wiener滤波
    kernel_wiener = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    result = np.fft.ifft2(dummy * kernel_wiener)
    result = np.abs(result)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# 盲卷积 Lucy-Richardson
def motion_psf(length=15, angle=0):
    psf = np.zeros((length, length))
    center = length // 2
    slope = np.tan(np.deg2rad(angle)) if angle != 90 else 0
    for i in range(length):
        offset = int(center + (i-center)*slope)
        if 0 <= offset < length:
            psf[i, offset] = 1
    psf /= psf.sum() if psf.sum() != 0 else 1
    return psf

def blind_deconv_lucy(img, psf_length=15, psf_angle=0, iterations=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    psf = motion_psf(psf_length, psf_angle)
    # 注意这里使用 num_iter
    deconvolved = richardson_lucy(gray.astype(np.float64)/255.0, psf, num_iter=iterations)
    deconvolved = np.clip(deconvolved * 255, 0, 255).astype(np.uint8)
    return deconvolved

# ----------------- 核心路由 ----------------
@app.route('/process', methods=['POST'])
def process():
    action = request.form.get('action')
    sub_action = request.form.get('sub_action')
    param = request.form.get('param', 'linear')
    filenames = request.form.get('filenames', '').split(',')
    if not filenames or not filenames[0]:
        return jsonify({'error': 'No file uploaded'}), 400
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filenames[0])
    img = cv2.imread(img_path)

    processed_filename = None
    result = None

    if action == 'processing':
        if sub_action == 'gray':
            result = gray_transform(img, param)
            processed_filename = get_unique_filename('gray')
        elif sub_action == 'binary':
            threshold = int(request.form.get('threshold', 128))
            result = binary(img, threshold=threshold)
            processed_filename = get_unique_filename('binary')
        elif sub_action == 'hist':
            result = hist_equalization(img)
            processed_filename = get_unique_filename('hist')
        elif sub_action == 'smooth':
            result = smooth(img, param)
            processed_filename = get_unique_filename('smooth')
        elif sub_action == 'sharpen':
            result = sharpen(img, param)
            processed_filename = get_unique_filename('sharpen')
        elif sub_action == 'filter':
            result = filter_img(img, param)
            processed_filename = get_unique_filename('filter')
        elif sub_action == 'fft':
            result = fft_transform(img)
            processed_filename = get_unique_filename('fft')
        elif sub_action == 'lowpass':
            method = param
            d0 = int(request.form.get('d0', 30))
            n = int(request.form.get('n', 2))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shape = gray.shape
            H = get_filter_mask(shape, method, d0, n)
            result = freq_filter(gray, H)
            processed_filename = get_unique_filename('lowpass')
        elif sub_action == 'highpass':
            method = param
            d0 = int(request.form.get('d0', 30))
            n = int(request.form.get('n', 2))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shape = gray.shape
            H = get_filter_mask(shape, method, d0, n)
            result = freq_filter(gray, H)
            processed_filename = get_unique_filename('highpass')
        # elif sub_action == 'dl':
        #     if param == 'beauty':
        #         from gfpgan_utils import face_beautify  # 如果上面没导入这里导入也可
        #         processed_filename = get_unique_filename('beauty')
        #         output_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        #         face_beautify(img_path, output_path)
        #         return jsonify({'processed_filename': processed_filename})
        # elif sub_action == 'dl':
        #     if param == 'beauty':
        #         result = beautify_face(img)
        #         processed_filename = get_unique_filename('beauty')
        #     elif param == 'style':
        #         result = style_transfer(img)
        #         processed_filename = get_unique_filename('style')
        #     else:
        #         return jsonify({'error': '未知深度学习操作'}), 400
        #
        # else:
        #     return jsonify({'error': 'Not implemented'}), 400

    elif action == 'segmentation':
        if sub_action == 'threshold':
            threshold = int(request.form.get('threshold', 128))
            result = segmentation_threshold(img, threshold)
            processed_filename = get_unique_filename('seg_threshold')
        elif sub_action == 'edge':
            result = segmentation_edge(img)
            processed_filename = get_unique_filename('seg_edge')
        elif action == 'segmentation':
            if sub_action == 'maskrcnn':
                result = maskrcnn_infer(img)
                processed_filename = get_unique_filename('maskrcnn')
        else:
            return jsonify({'error': 'Not implemented'}), 400

    elif action == 'restoration':
        if sub_action == 'dehaze':
            result = dehaze(img)
            processed_filename = get_unique_filename('dehaze')
        elif sub_action == 'deblur':
            method = request.form.get('method', 'wiener')
            if method == 'wiener':
                result = wiener_filter(img)
                processed_filename = get_unique_filename('deblur_wiener')
            elif method == 'blind':
                psf_length = int(request.form.get('psf_length', 15))
                psf_angle = int(request.form.get('psf_angle', 0))
                iterations = int(request.form.get('iterations', 30))
                result = blind_deconv_lucy(img, psf_length, psf_angle, iterations)
                processed_filename = get_unique_filename('deblur_blind')
            else:
                return jsonify({'error': 'Unknown deblur method'}), 400
        else:
            return jsonify({'error': 'Not implemented'}), 400

    elif action == 'stitching':
        if len(filenames) < 2:
            return jsonify({'error': '请上传两张图片'}), 400
        img1 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filenames[0]))
        img2 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filenames[1]))
        if sub_action == 'sift_ransac':
            try:
                result = sift_ransac_stitch(img1, img2)
                processed_filename = get_unique_filename('stitching')
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        elif sub_action == 'blend':
            try:
                result = pyramid_blending(img1, img2)
                processed_filename = get_unique_filename('blend')
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        elif sub_action == 'superglue':
            try:
                result = superglue_stitch(img1, img2)
                processed_filename = get_unique_filename('superglue')
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Not implemented'}), 400

    if result is not None and processed_filename is not None:
        save_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(save_path, result)
        return jsonify({'processed_filename': processed_filename})
    else:
        return jsonify({'error': 'Unknown processing type'}), 400

if __name__ == '__main__':
    app.run(debug=True)