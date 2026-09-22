import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== ĐANG CHẠY TOÀN BỘ BÀI LAB DAY 5 (ĐỦ 7 BÀI) ===")

# ==========================================
# 1. GRAYLEVEL THRESHOLDING (Phân ngưỡng mức xám)
# ==========================================
# Đọc ảnh xám
img_gray = cv2.imread("Day 5/2. Global Thresholding/paper.png", 0)
if img_gray is not None:
    # Phân ngưỡng mức xám cố định với ngưỡng T = 128
    _, graylevel_thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
else:
    graylevel_thresh = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# 2. GLOBAL THRESHOLDING (Phân ngưỡng toàn cục - Otsu)
# ==========================================
if img_gray is not None:
    _, global_thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
else:
    global_thresh = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# 3. LOCALLY ADAPTIVE THRESHOLDING (Thích nghi cục bộ)
# ==========================================
img_adaptive = cv2.imread(
    "Day 5/3. Locally Adaptive Thresholding/paper.png", 0
)
if img_adaptive is not None:
    adaptive_thresh = cv2.adaptiveThreshold(
        img_adaptive,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
else:
    adaptive_thresh = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# 4. MAP SKIN DETECTOR (Phát hiện vùng da)
# ==========================================
test_skin_img = cv2.imread("Day 5/4. MAP Skin Detector/Face_Test_2.jpg")
if test_skin_img is not None:
    test_ycrcb = cv2.cvtColor(test_skin_img, cv2.COLOR_BGR2YCrCb)
    # Ngưỡng màu da tiêu chuẩn trong không gian YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(test_ycrcb, lower_skin, upper_skin)
else:
    skin_mask = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# 5. REGION LABELING (Gắn nhãn vùng liên thông)
# ==========================================
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    skin_mask, connectivity=8
)
# Chuẩn hóa nhãn và áp dụng colormap để hiển thị màu sắc từng vùng (Đã fix lỗi kích thước)
norm_labels = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX).astype(
    np.uint8
)
labeled_img = cv2.applyColorMap(norm_labels, cv2.COLORMAP_JET)
labeled_img[labels == 0] = [0, 0, 0]  # Đặt nền về màu đen

# ==========================================
# 6 & 7. HOLE FILLING (Lấp đầy lỗ trống - 2 phần)
# ==========================================
img_hole1 = cv2.imread("Day 5/6. Hole Filling/paper.png", 0)
img_hole2 = cv2.imread("Day 5/7. Hole Filling/book.png", 0)


def perform_hole_filling(binary_img):
    if binary_img is None:
        return None
    _, bin_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    h, w = bin_img.shape
    flood_filled = bin_img.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask, (0, 0), 255)
    inverted_flood = cv2.bitwise_not(flood_filled)
    return cv2.bitwise_or(bin_img, inverted_flood)


filled_1 = perform_hole_filling(img_hole1)
filled_2 = perform_hole_filling(img_hole2)

# ==========================================
# HIỂN THỊ TẤT CẢ KẾT QUẢ TRỰC QUAN (ĐỦ 7 BÀI)
# ==========================================
plt.figure(figsize=(16, 10))

plt.subplot(2, 4, 1)
plt.imshow(img_gray, cmap="gray")
plt.title("1. Graylevel Thresh")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(global_thresh, cmap="gray")
plt.title("2. Global Thresh (Otsu)")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(adaptive_thresh, cmap="gray")
plt.title("3. Adaptive Thresh")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(skin_mask, cmap="gray")
plt.title("4. MAP Skin Mask")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
plt.title("5. Region Labeling")
plt.axis("off")

plt.subplot(2, 4, 6)
if filled_1 is not None:
    plt.imshow(filled_1, cmap="gray")
plt.title("6. Hole Filling (Paper)")
plt.axis("off")

plt.subplot(2, 4, 7)
if filled_2 is not None:
    plt.imshow(filled_2, cmap="gray")
plt.title("7. Hole Filling (Book)")
plt.axis("off")

plt.tight_layout()
plt.show()