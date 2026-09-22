import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== ĐANG CHẠY TOÀN BỘ BÀI LAB DAY 6 ===")

# Định nghĩa phần tử cấu trúc (Structuring Element / Kernel) dùng chung
kernel = np.ones((3, 3), np.uint8)

# ==========================================
# 1. BINARY DILATION (Phép giãn nở nhị phân)
# ==========================================
img_dil = cv2.imread("Day 6/1. Binary Dilation/bacteria.png", 0)
if img_dil is not None:
    _, bin_dil = cv2.threshold(img_dil, 127, 255, cv2.THRESH_BINARY)
    dilated_res = cv2.dilate(bin_dil, kernel, iterations=1)
else:
    dilated_res = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# 2. BINARY EROSION (Phép co nhị phân)
# ==========================================
img_circles = cv2.imread(
    "Day 6/3. Blob Separation by Binary Erosion/circles.png", 0
)
if img_circles is not None:
    _, bin_circles = cv2.threshold(img_circles, 127, 255, cv2.THRESH_BINARY)
    eroded_res = cv2.erode(bin_circles, kernel, iterations=1)
else:
    eroded_res = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# 3. BLOB SEPARATION BY BINARY EROSION (Phân tách vật thể)
# ==========================================
if img_circles is not None:
    # Tăng số lần lặp (iterations=2) để bẻ gãy các liên kết dính nhau giữa các đối tượng
    separated_res = cv2.erode(bin_circles, kernel, iterations=2)
else:
    separated_res = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# 4. HOLE DETECTION BY BINARY EROSION (Phát hiện lỗ trống)
# ==========================================
img_fence = cv2.imread(
    "Day 6/4. Hole Detection by Binary Erosion/Binary_Erosion_Fence_bw.png",
    0,
)
if img_fence is not None:
    _, bin_fence = cv2.threshold(img_fence, 127, 255, cv2.THRESH_BINARY)
    inv_fence = cv2.bitwise_not(bin_fence)
    hole_detected = cv2.erode(inv_fence, kernel, iterations=1)
else:
    hole_detected = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# 5. SMALL HOLE REMOVAL (Xóa lỗ nhỏ bằng phép Closing)
# ==========================================
img_peter = cv2.imread("Day 6/5. Small Hole Removal/peter.png", 0)
if img_peter is not None:
    _, bin_peter = cv2.threshold(img_peter, 127, 255, cv2.THRESH_BINARY)
    # Phép Closing giúp lấp đầy các lỗ thủng nhỏ bên trong đối tượng
    hole_removed = cv2.morphologyEx(
        bin_peter, cv2.MORPH_CLOSE, kernel, iterations=2
    )
else:
    hole_removed = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# 6. BINARY MORPHOLOGICAL EDGE DETECTOR (Phát hiện biên)
# ==========================================
img_edge = cv2.imread(
    "Day 6/6. Binary Morphological Edge Detector/cliparts.png", 0
)
if img_edge is not None:
    _, bin_edge = cv2.threshold(img_edge, 127, 255, cv2.THRESH_BINARY)
    ero_e = cv2.erode(bin_edge, kernel, iterations=1)
    dil_e = cv2.dilate(bin_edge, kernel, iterations=1)

    # Biên kết hợp (Morphological Gradient) = Dilation - Erosion
    combined_edge = cv2.absdiff(dil_e, ero_e)
else:
    combined_edge = np.zeros((100, 100), dtype=np.uint8)

# ==========================================
# TRỰC QUAN HÓA TẤT CẢ CÁC BÀI LAB DAY 6
# ==========================================
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(dilated_res, cmap="gray")
plt.title("1. Binary Dilation")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(eroded_res, cmap="gray")
plt.title("2. Binary Erosion")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(separated_res, cmap="gray")
plt.title("3. Blob Separation")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(hole_detected, cmap="gray")
plt.title("4. Hole Detection")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(hole_removed, cmap="gray")
plt.title("5. Small Hole Removal")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(combined_edge, cmap="gray")
plt.title("6. Morphological Edge")
plt.axis("off")

plt.tight_layout()
plt.show()