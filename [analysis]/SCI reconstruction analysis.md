# SCI reconstruction analysis

1. **E2E_CNN **training info (Cr=10) (psnr)

   | train/valid  | binary mask | combine binary mask | gray mask | combine gray mask |
   | ------------ | :---------: | :-----------------: | :-------: | :---------------: |
   | train        |    28.7     |        26.3         |   27.3    |       25.7        |
   | valid        |    24.4     |        22.7         |   23.5    |       22.1        |
   | test_256_10f |    24.7     |        22.8         |   23.7    |       22.3        |

   

2. **RNN-SCI** test info (Cr=10) (psnr/ssim)

   - test set: bm_256_10f (aerial	crash	drop	kobe	runner	traffic)
   - inadequate training!

   | model                       | aerial     | crash      | drop       | kobe       | runner     | traffic    | [average]   |
   | --------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ----------- |
   | combine_binary_mask_256_10f | 26.5/0.843 | 26.9/0.890 | 35.6/0.974 | 26.6/0.845 | 31.4/0.931 | 22.4/0.773 | 28.23/0.876 |
   | binary_mask_256_10          | 27.5/0.874 | 27.9/0.910 | 38.5/0.983 | 29.8/0.910 | 34.3/0.952 | 25.0/0.858 | 30.5/0.915  |

   

3. **reconstruction performance comparison** between E2E_CNN, RNN-SCI and PnP (psnr/ssim)

- test set: test_256_10f (traffic3  train3  tuk-tuk7  upside-down4  walking12)

| method  | binary mask    | combined binary mask |   gray mask    | combined gray mask |
| :-----: | -------------- | :------------------: | :------------: | :----------------: |
| E2E_CNN | 24.6777/0.7657 |    22.8083/0.6933    | 23.7212/0.7217 |   22.3240/0.6698   |
- test set: bm_256_10f (aerial	crash	drop	kobe	runner	traffic)

|   method   | binary mask    | combined binary mask |   gray mask   | combined gray mask |
| :--------: | -------------- | :------------------: | :-----------: | :----------------: |
|  E2E_CNN   | 29.003/0.9037  |    27.628/0.8730     | 28.791/0.8855 |   27.7606/0.872    |
|  RNN-SCI*  | 31.682/        |       28.964/        |               |                    |
| PnP-FFDNET | 28.3517/0.8641 |     20.84/0.6838     |               |                    |
|   GAP_TV   | 26.45/0.8374   |     23.09/0.7998     |               |                    |

注：

RNN-SCI*： inadequate training!



---
Appendix

1. **E2E_CNN reproduction performance test** (psnr/ssim)
- binary mask
   - (Cr=10, retrained model from Zhihong Zhang) & (Cr=8, original model from Ziyi Meng)
   - test set: test_256_10f (traffic3  train3  tuk-tuk7  upside-down4  walking12)
|       data        |    traffic3    |     train3     |    tuk-tuk7    |  upside-down4  |   walking12    |   [average]    |
| :---------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| **Cr8, original** | 26.5100/0.8721 | 24.9001/0.7140 | 25.3374/0.7892 | 24.2673/0.7690 | 23.6823/0.7664 | 24.9394/0.7821 |
| **Cr10, retrain** | 26.5834/0.8709 | 24.8993/0.7100 | 24.9191/0.7683 | 23.3844/0.7317 | 23.6022/0.7478 | 24.6777/0.7657 |
