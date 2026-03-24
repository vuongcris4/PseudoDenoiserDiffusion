# Dataset Analysis Report: Pseudo-label vs Ground Truth

**Date:** Auto-generated
**Dataset:** OEM_v2_aDanh (2189 images)
**Classes:** 8 (Barren, Rangeland, Developed, Road, Tree, Water, Agriculture, Building)

## 1. Dataset Overview

| Split | Images | Total Pixels | Correct | Noisy | Pixel Acc | Noise Ratio |
|-------|--------|-------------|---------|-------|-----------|-------------|
| train | 1751 | 1,688,350,342 | 1,182,470,391 | 505,879,951 | 0.7004 | 0.2996 |
| val | 218 | 213,059,367 | 149,791,838 | 63,267,529 | 0.7031 | 0.2969 |
| test | 220 | 216,179,804 | 152,097,900 | 64,081,904 | 0.7036 | 0.2964 |
| **Total** | **2189** | **2,117,589,513** | **1,484,360,129** | **633,229,384** | **0.7010** | **0.2990** |

## 2. Global Metrics

- **mIoU:** 0.4917
- **Mean Precision:** 0.6729
- **Mean Recall:** 0.6223
- **Pixel Accuracy:** 0.7010
- **Edge Error Rate:** 0.5435 (130,196,352 / 239,544,055 edge pixels)
- **Non-edge Error Rate:** 0.2678 (503,033,032 / 1,878,045,458 non-edge pixels)

## 3. Per-class Performance

| Class | GT Pixels | Pseudo Pixels | GT Ratio | IoU | Precision | Recall | F1 | Noise Rate |
|-------|-----------|---------------|----------|-----|-----------|--------|----|------------|
| Barren | 40,679,210 | 9,580,452 | 0.0192 | 0.0710 | 0.3479 | 0.0819 | 0.1326 | 0.9181 |
| Rangeland | 448,404,305 | 393,060,067 | 0.2118 | 0.4448 | 0.6591 | 0.5777 | 0.6157 | 0.4223 |
| Developed | 387,276,302 | 511,403,137 | 0.1829 | 0.4465 | 0.5424 | 0.7163 | 0.6174 | 0.2837 |
| Road | 140,711,683 | 122,555,864 | 0.0664 | 0.4881 | 0.7046 | 0.6137 | 0.6560 | 0.3863 |
| Tree | 397,090,787 | 412,814,835 | 0.1875 | 0.6317 | 0.7595 | 0.7896 | 0.7743 | 0.2104 |
| Water | 69,034,855 | 59,767,311 | 0.0326 | 0.5043 | 0.7225 | 0.6255 | 0.6705 | 0.3745 |
| Agriculture | 281,300,307 | 259,615,098 | 0.1328 | 0.6478 | 0.8191 | 0.7559 | 0.7862 | 0.2441 |
| Building | 353,092,064 | 348,792,749 | 0.1667 | 0.6993 | 0.8281 | 0.8181 | 0.8231 | 0.1819 |

## 4. Confusion Matrix (GT rows → Pseudo columns)

| GT \\ Pred | Barren | Rangeland | Developed | Road | Tree | Water | Agriculture | Building |
|---|---|---|---|---|---|---|---|---|
| Barren | **3,332,678** (8.2%) | 4,489,658 (11.0%) | 27,772,717 (68.3%) | 393,195 (1.0%) | 626,301 (1.5%) | 1,970,778 (4.8%) | 1,967,512 (4.8%) | 126,371 (0.3%) |
| Rangeland | 2,061,896 (0.5%) | **259,060,862** (57.8%) | 65,556,515 (14.6%) | 4,055,159 (0.9%) | 74,158,009 (16.5%) | 5,904,398 (1.3%) | 29,304,385 (6.5%) | 8,303,081 (1.9%) |
| Developed | 2,364,604 (0.6%) | 21,559,827 (5.6%) | **277,404,257** (71.6%) | 26,424,192 (6.8%) | 9,078,454 (2.3%) | 1,422,882 (0.4%) | 4,581,340 (1.2%) | 44,440,746 (11.5%) |
| Road | 89,305 (0.1%) | 3,892,391 (2.8%) | 44,057,100 (31.3%) | **86,354,423** (61.4%) | 2,341,024 (1.7%) | 475,626 (0.3%) | 1,279,926 (0.9%) | 2,221,888 (1.6%) |
| Tree | 394,722 (0.1%) | 52,612,446 (13.2%) | 17,183,731 (4.3%) | 1,704,890 (0.4%) | **313,538,201** (79.0%) | 3,809,024 (1.0%) | 5,081,662 (1.3%) | 2,766,111 (0.7%) |
| Water | 190,689 (0.3%) | 8,191,289 (11.9%) | 10,046,764 (14.6%) | 1,195,764 (1.7%) | 1,875,552 (2.7%) | **43,180,841** (62.5%) | 3,491,319 (5.1%) | 862,637 (1.2%) |
| Agriculture | 1,130,063 (0.4%) | 40,045,324 (14.2%) | 13,662,933 (4.9%) | 1,248,397 (0.4%) | 8,600,956 (3.1%) | 2,747,992 (1.0%) | **212,640,797** (75.6%) | 1,223,845 (0.4%) |
| Building | 16,495 (0.0%) | 3,208,270 (0.9%) | 55,719,120 (15.8%) | 1,179,844 (0.3%) | 2,596,338 (0.7%) | 255,770 (0.1%) | 1,268,157 (0.4%) | **288,848,070** (81.8%) |

## 5. Top Confusion Pairs (GT class → wrongly predicted as)

| Rank | GT Class | Predicted As | Pixels | % of GT Class |
|------|----------|-------------|--------|---------------|
| 1 | Rangeland | Tree | 74,158,009 | 16.54% |
| 2 | Rangeland | Developed | 65,556,515 | 14.62% |
| 3 | Building | Developed | 55,719,120 | 15.78% |
| 4 | Tree | Rangeland | 52,612,446 | 13.25% |
| 5 | Developed | Building | 44,440,746 | 11.48% |
| 6 | Road | Developed | 44,057,100 | 31.31% |
| 7 | Agriculture | Rangeland | 40,045,324 | 14.24% |
| 8 | Rangeland | Agriculture | 29,304,385 | 6.54% |
| 9 | Barren | Developed | 27,772,717 | 68.27% |
| 10 | Developed | Road | 26,424,192 | 6.82% |

## 6. Noise Distribution Statistics

- **Mean noise ratio:** 0.3052 ± 0.1293
- **Median noise ratio:** 0.2895
- **Min / Max noise ratio:** 0.0086 / 0.9843
- **Mean mIoU per image:** 0.3947 ± 0.1046
- **Min / Max mIoU:** 0.0447 / 0.7762

### Noise Ratio Distribution

| Noise Ratio Range | Count | Percentage |
|-------------------|-------|------------|
| 0.00 - 0.05 | 11 | 0.5% |
| 0.05 - 0.10 | 54 | 2.5% |
| 0.10 - 0.15 | 104 | 4.8% |
| 0.15 - 0.20 | 202 | 9.2% |
| 0.20 - 0.30 | 822 | 37.6% |
| 0.30 - 0.50 | 858 | 39.2% |
| 0.50 - 1.00 | 138 | 6.3% |

## 7. Suggested Additional Metrics for Denoising Research

Các chỉ số gợi ý thêm để phân tích toàn diện cho bài toán denoising pseudo-label:

1. **Boundary Error Ratio** (đã tính ở trên): Tỷ lệ lỗi tại biên vs. không phải biên → xác định noise chủ yếu ở boundary hay interior
2. **Spatial Noise Clustering**: Dùng connected-component analysis trên error mask → đo kích thước vùng noise trung bình
3. **Per-Region Noise**: Tên khu vực (city/region trong filename) → xem region nào CISC-R predict kém nhất
4. **Class-conditional Transition Matrix**: Ma trận chuyển đổi chuẩn hóa → dùng làm prior cho D3PM transition
5. **Noise Spatial Heatmap**: Aggregate error positions → xem noise tập trung ở góc, biên hay giữa ảnh
6. **Small Object Error Rate**: Tính riêng cho các vùng GT nhỏ (< 100px) → pseudo-label thường nhầm object nhỏ
7. **Label Smoothness Score**: Entropy(pseudo) vs Entropy(GT) → đo mức "noisy" của pseudo-label
8. **Confidence Map Analysis**: Nếu CISC-R có confidence output → correlation giữa confidence thấp và error
9. **Temporal/Geographic Bias**: Group theo city → phát hiện bias theo geographic region
10. **Class Imbalance Impact**: Correlation giữa class frequency và noise rate → minority class bị ảnh hưởng nhiều hơn?

## 8. Geographic Analysis (Per-Region)

| Region | Images | Noise Ratio | Total Pixels |
|--------|--------|-------------|-------------|
| western | 16 | 0.5712 | 16,289,960 |
| paris | 42 | 0.4888 | 17,720,956 |
| shanghai | 42 | 0.4063 | 17,741,757 |
| ulaanbaatar | 28 | 0.4042 | 29,036,698 |
| khartoum | 42 | 0.4026 | 17,743,047 |
| chiclayo | 30 | 0.4010 | 30,917,225 |
| ica | 30 | 0.3980 | 31,319,942 |
| ngaoundere | 42 | 0.3950 | 43,924,144 |
| kagera | 42 | 0.3947 | 44,002,350 |
| pisco | 30 | 0.3900 | 31,157,265 |
| vegas | 30 | 0.3811 | 12,674,788 |
| tokyo | 42 | 0.3810 | 44,038,735 |
| lima | 30 | 0.3782 | 31,058,631 |
| warminsko-mazurskie | 7 | 0.3674 | 7,339,941 |
| santiago | 42 | 0.3640 | 44,040,192 |
| rotterdam | 30 | 0.3637 | 19,829,552 |
| kyoto | 42 | 0.3593 | 44,033,703 |
| zanzibar | 87 | 0.3587 | 91,141,342 |
| rio | 42 | 0.3569 | 7,462,407 |
| chincha | 30 | 0.3461 | 31,183,742 |
| ... (52 more regions) | | | |

