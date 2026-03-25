# Noise Characterization Report: OEM Pseudo-label Dataset

**Dataset:** OEM_v2_aDanh (2189 images, 8 classes)
**Classes:** Bareland, Rangeland, Developed, Road, Tree, Water, Agriculture, Building
**Image size:** 1024×1024

---
## 1. Boundary Error Ratio

- **Edge error rate:** 0.5435 (130,196,352 / 239,544,055 edge pixels)
- **Non-edge error rate:** 0.2678 (503,033,032 / 1,878,045,458 pixels)
- **Edge/non-edge ratio:** 2.03x

**Per-class boundary error rate:**

| Class | Edge Error Rate | Edge Errors | Edge Pixels |
|-------|----------------|-------------|-------------|
| Bareland | 0.9213 | 1,540,634 | 1,672,280 |
| Rangeland | 0.6594 | 38,739,083 | 58,749,028 |
| Developed | 0.4857 | 31,277,144 | 64,401,787 |
| Road | 0.6125 | 13,311,886 | 21,732,695 |
| Tree | 0.4921 | 23,207,897 | 47,161,421 |
| Water | 0.7747 | 2,012,031 | 2,597,277 |
| Agriculture | 0.6311 | 4,107,692 | 6,508,796 |
| Building | 0.4357 | 15,999,985 | 36,720,771 |

> **Insight:** Noise tại biên (boundary) cao gấp ~2x so với vùng interior. Điều này cho thấy phần lớn lỗi pseudo-label xảy ra ở ranh giới giữa các class, không phải ở vùng đồng nhất bên trong. Denoiser nên có cơ chế boundary-aware.

---
## 2. Spatial Noise Clustering

- **Total noise clusters:** 1,479,507
- **Mean clusters per image:** 675.9 ± 454.5
- **Median clusters per image:** 618
- **Mean cluster size:** 428.0 pixels
- **Median cluster size:** 8 pixels
- **Max cluster size:** 1,021,252.0 pixels
- **Min cluster size:** 1.0 pixels

**Noise cluster size distribution:**

| Size Range (px) | Count | % of Clusters |
|-----------------|-------|---------------|
| 1-4 | 627,464 | 42.4% |
| 5-9 | 143,206 | 9.7% |
| 10-49 | 325,609 | 22.0% |
| 50-99 | 121,042 | 8.2% |
| 100-499 | 172,295 | 11.6% |
| 500-999 | 36,174 | 2.4% |
| 1k-5k | 38,833 | 2.6% |
| 5k-50k | 12,935 | 0.9% |
| >50k | 1,949 | 0.1% |

> **Insight:** Nếu phần lớn cluster nhỏ (< 50px) → noise chủ yếu là salt-and-pepper / boundary jitter. Nếu nhiều cluster lớn (> 1000px) → noise mang tính vùng (region-level confusion).

---
## 3. Per-Region Noise

**Top 20 noisiest regions:**

| Rank | Region | Images | Noise Ratio | Noisy Pixels |
|------|--------|--------|-------------|-------------|
| 1 | western | 16 | 0.5712 | 9,305,355 |
| 2 | paris | 42 | 0.4888 | 8,661,390 |
| 3 | shanghai | 42 | 0.4063 | 7,207,764 |
| 4 | ulaanbaatar | 28 | 0.4042 | 11,736,236 |
| 5 | khartoum | 42 | 0.4026 | 7,142,540 |
| 6 | chiclayo | 30 | 0.4010 | 12,397,290 |
| 7 | ica | 30 | 0.3980 | 12,466,237 |
| 8 | ngaoundere | 42 | 0.3950 | 17,351,193 |
| 9 | kagera | 42 | 0.3947 | 17,368,285 |
| 10 | pisco | 30 | 0.3900 | 12,152,533 |
| 11 | vegas | 30 | 0.3811 | 4,830,559 |
| 12 | tokyo | 42 | 0.3810 | 16,779,277 |
| 13 | lima | 30 | 0.3782 | 11,745,086 |
| 14 | warminsko-mazurskie | 7 | 0.3674 | 2,696,633 |
| 15 | santiago | 42 | 0.3640 | 16,031,625 |
| 16 | rotterdam | 30 | 0.3637 | 7,211,585 |
| 17 | kyoto | 42 | 0.3593 | 15,819,844 |
| 18 | zanzibar | 87 | 0.3587 | 32,691,437 |
| 19 | rio | 42 | 0.3569 | 2,663,554 |
| 20 | chincha | 30 | 0.3461 | 10,793,931 |

**Top 10 cleanest regions:**

| Rank | Region | Images | Noise Ratio |
|------|--------|--------|-------------|
| 1 | zachodniopomorskie | 7 | 0.1406 |
| 2 | swietokrzyskie | 24 | 0.1765 |
| 3 | wielkopolskie | 24 | 0.1849 |
| 4 | baybay | 40 | 0.1920 |
| 5 | malopolskie | 24 | 0.2053 |
| 6 | mazowieckie | 24 | 0.2099 |
| 7 | lodzkie | 14 | 0.2264 |
| 8 | kitsap | 30 | 0.2270 |
| 9 | muenster | 42 | 0.2273 |
| 10 | dortmund | 42 | 0.2278 |

> **Insight:** Có 72 regions. Noisiest: western (57.12%), Cleanest: zachodniopomorskie (14.06%). Geographic diversity gây ra noise patterns khác nhau đáng kể.

---
## 4. Class-conditional Transition Matrix

Ma trận P(pseudo=j | GT=i) — xác suất chuyển đổi chuẩn hóa theo hàng:

| GT \\ Pseudo | Bareland | Rangeland | Developed | Road | Tree | Water | Agriculture | Building |
|---|---|---|---|---|---|---|---|---|
| Bareland | **0.082** | _0.110_ | _0.683_ | 0.010 | 0.015 | 0.048 | 0.048 | 0.003 |
| Rangeland | 0.005 | **0.578** | _0.146_ | 0.009 | _0.165_ | 0.013 | _0.065_ | 0.019 |
| Developed | 0.006 | _0.056_ | **0.716** | _0.068_ | 0.023 | 0.004 | 0.012 | _0.115_ |
| Road | 0.001 | 0.028 | _0.313_ | **0.614** | 0.017 | 0.003 | 0.009 | 0.016 |
| Tree | 0.001 | _0.132_ | 0.043 | 0.004 | **0.790** | 0.010 | 0.013 | 0.007 |
| Water | 0.003 | _0.119_ | _0.146_ | 0.017 | 0.027 | **0.625** | _0.051_ | 0.012 |
| Agriculture | 0.004 | _0.142_ | 0.049 | 0.004 | 0.031 | 0.010 | **0.756** | 0.004 |
| Building | 0.000 | 0.009 | _0.158_ | 0.003 | 0.007 | 0.001 | 0.004 | **0.818** |

**Class retention rate (diagonal):**

- Bareland    : 0.082 ███
- Rangeland   : 0.578 ███████████████████████
- Developed   : 0.716 ████████████████████████████
- Road        : 0.614 ████████████████████████
- Tree        : 0.790 ███████████████████████████████
- Water       : 0.625 █████████████████████████
- Agriculture : 0.756 ██████████████████████████████
- Building    : 0.818 ████████████████████████████████

> **Insight:** Ma trận này có thể dùng làm informed prior cho bất kỳ denoising model nào. Bareland chỉ giữ lại 8.2% — gần như bị xóa hoàn toàn bởi CISC-R. Building giữ lại 81.8% — class ít bị ảnh hưởng nhất.

---
## 5. Noise Spatial Heatmap (4×4 Grid)

Error rate per 4×4 grid zone (rows = top→bottom, cols = left→right):

| Zone | Col 0 | Col 1 | Col 2 | Col 3 |
|------|------|------|------|------|
| Row 0 | 0.2980 | 0.3051 | 0.3028 | 0.3077 |
| Row 1 | 0.2981 | 0.2957 | 0.2956 | 0.3006 |
| Row 2 | 0.2946 | 0.2940 | 0.2902 | 0.2991 |
| Row 3 | 0.2979 | 0.2986 | 0.3014 | 0.3048 |

- **Corner zones mean error rate:** 0.3021
- **Center zones mean error rate:** 0.2939
- **Border zones error rate:** 0.3007 (476,317,524 / 1,583,944,690)
- **Inner zones error rate:** 0.2939 (156,680,204 / 533,163,115)

> **Insight:** Noise phân bố đều hay tập trung ở vùng nào? Nếu corner/border cao hơn → CISC-R có context window bias. Nếu đều → noise là class-dependent, không phải spatial-dependent.

---
## 6. Small Object Error Rate (Size-binned)

| Size Bin | Num Regions | Total Pixels | Correct | Accuracy |
|----------|-------------|-------------|---------|----------|
| micro(<50) | 392,779 | 5,811,660 | 1,230,470 | 0.2117 |
| tiny(50-100) | 140,078 | 10,104,547 | 3,115,060 | 0.3083 |
| small(100-500) | 323,822 | 78,574,845 | 38,148,070 | 0.4855 |
| medium(500-2k) | 189,449 | 190,985,363 | 122,225,214 | 0.6400 |
| large(2k-10k) | 88,022 | 368,916,932 | 252,689,555 | 0.6849 |
| huge(>10k) | 31,154 | 1,463,196,166 | 1,066,951,760 | 0.7292 |

> **Insight:** Micro objects (<50px) accuracy = 21.17%, Huge objects (>10k px) accuracy = 72.92%. Gap = 51.75%. Pseudo-label reliability scales strongly with object size.

---
## 7. Label Smoothness / Entropy Analysis

- **Mean entropy (pseudo):** 0.9504 ± 0.3683
- **Mean entropy (GT):** 1.0031 ± 0.4008
- **Mean entropy difference (pseudo - GT):** -0.0527 ± 0.1461
- **Pseudo > GT entropy in 701/2189 images (32.0%)**

> **Insight:** Nếu entropy(pseudo) > entropy(GT) → pseudo-label "noisier" hơn GT (nhiều chuyển đổi class hơn trong cùng 1 patch). Entropy cao ↔ boundary phức tạp hoặc noisy pixels rải rác. Có thể dùng entropy difference làm quality score cho mỗi ảnh.

---
## 8. Confidence Map Analysis

**SKIPPED** — CISC-R model output trong dataset này chỉ có hard pseudo-label (argmax), không có softmax probabilities hoặc confidence scores. Nếu có thể re-run CISC-R inference với `--save-probabilities`, thì có thể tính correlation(confidence, error) và dùng confidence làm sample weight.

---
## 9. Geographic/Temporal Bias Analysis

**Per-region per-class noise rate (top 15 noisiest regions):**

| Region | Overall | Bareland | Rangeland | Developed | Road | Tree | Water | Agriculture | Building |
|---|---|---|---|---|---|---|---|---|---|
| western | 0.571 | 0.864 | 0.630 | 0.158 | 0.686 | 0.158 | 0.877 | 0.662 | 0.169 |
| paris | 0.489 | 1.000 | 0.576 | 0.580 | 0.297 | 0.652 | 0.801 | — | 0.198 |
| shanghai | 0.406 | 1.000 | 0.514 | 0.410 | 0.524 | 0.540 | 0.697 | 0.461 | 0.278 |
| ulaanbaatar | 0.404 | 0.956 | 0.678 | 0.100 | 0.392 | 0.619 | 0.292 | 0.810 | 0.363 |
| khartoum | 0.403 | 0.976 | 0.685 | 0.440 | 0.584 | 0.340 | 1.000 | — | 0.218 |
| chiclayo | 0.401 | 0.705 | 0.688 | 0.391 | 0.460 | 0.356 | 0.392 | 0.185 | 0.184 |
| ica | 0.398 | 0.987 | 0.698 | 0.178 | 0.319 | 0.325 | 0.828 | 0.625 | 0.235 |
| ngaoundere | 0.395 | 0.949 | 0.585 | 0.204 | 0.382 | 0.214 | 0.632 | 0.504 | 0.169 |
| kagera | 0.395 | 1.000 | 0.450 | 0.607 | 0.446 | 0.215 | 0.896 | 0.576 | 0.111 |
| pisco | 0.390 | 0.907 | 0.561 | 0.145 | 0.349 | 0.465 | 0.796 | 0.262 | 0.255 |
| vegas | 0.381 | 1.000 | 0.804 | 0.319 | 0.254 | 0.314 | 0.906 | 1.000 | 0.100 |
| tokyo | 0.381 | 0.877 | 0.423 | 0.330 | 0.622 | 0.380 | 0.678 | 0.439 | 0.235 |
| lima | 0.378 | 0.987 | 0.418 | 0.106 | 0.403 | 0.375 | 0.561 | 0.814 | 0.306 |
| warminsko-mazurskie | 0.367 | — | 0.476 | 0.835 | 0.697 | 0.202 | 0.102 | 0.682 | 0.391 |
| santiago | 0.364 | 1.000 | 0.549 | 0.390 | 0.336 | 0.249 | 0.639 | 1.000 | 0.215 |

- **Noise rate range across regions:** 0.1406 — 0.5712
- **Std of noise rate:** 0.0742
- **Coefficient of variation:** 0.2493

> **Insight:** Variation lớn giữa các regions cho thấy CISC-R performance phụ thuộc mạnh vào geographic context (kiểu đô thị, thảm thực vật, khí hậu). Denoiser có thể cần region-aware conditioning hoặc ít nhất domain-specific augmentation.

---
## 10. Class Imbalance Impact

| Class | GT Pixel Ratio | Noise Rate | GT Pixels |
|-------|---------------|------------|-----------|
| Bareland | 0.0192 | 0.9181 | 40,679,210 |
| Rangeland | 0.2118 | 0.4223 | 448,404,305 |
| Developed | 0.1829 | 0.2837 | 387,276,302 |
| Road | 0.0664 | 0.3863 | 140,711,683 |
| Tree | 0.1875 | 0.2104 | 397,090,787 |
| Water | 0.0326 | 0.3745 | 69,034,855 |
| Agriculture | 0.1328 | 0.2441 | 281,300,307 |
| Building | 0.1667 | 0.1819 | 353,092,064 |

- **Pearson correlation (freq vs noise rate):** r = -0.6339, p = 0.0914
- **Spearman correlation (freq vs noise rate):** ρ = -0.3571, p = 0.3851

> **Insight:** Negative correlation → minority classes bị noise nhiều hơn. Điều này phù hợp với intuition: class hiếm có ít training data → CISC-R predict kém hơn. Denoiser nên có class-balanced loss hoặc class-specific noise modeling.

---
## Summary: Key Data Facts for Denoiser Research

1. **~30% pixels are noisy** (pixel accuracy 70.1%, mIoU 0.49)
2. **Boundary noise is 2.0x worse** than interior noise (54.35% vs 26.78%)
3. **Median noise cluster = 8 pixels**, mean = 428 → noise is mostly fine-grained boundary artifacts
4. **Small objects are unreliable**: micro (<50px) accuracy = 21.17%
5. **Bareland is nearly destroyed** by CISC-R: 8.2% retention, 68% → Developed
6. **Developed is a noise sink**: absorbs errors from 5+ other classes
7. **Entropy(pseudo) < Entropy(GT)** in 32% of images → pseudo-labels are smoother
8. **Geographic noise variance is high**: σ=0.0742, range 14.06%—57.12%
9. **Spatial noise is uniform** across image position
10. **Class frequency vs noise correlation:** Pearson r=-0.634 → minority classes noisier

**Transition matrix saved to:** `analysis_output/advanced/transition_matrix.npy` (usable as noise prior for any model)
