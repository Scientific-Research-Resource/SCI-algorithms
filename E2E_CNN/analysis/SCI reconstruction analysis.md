# SCI reconstruction analysis

1. E2E_CNN **training info** (Cr=10)

   | train/valid | binary mask | combine binary mask | gray mask | combine gray mask |
   | ----------- | :---------: | :-----------------: | :-------: | :---------------: |
   | train       |    28.7     |        26.3         |   27.3    |       25.7        |
   | valid       |    24.4     |        22.7         |   23.5    |       22.1        |

   

2. reconstruction performance comparison between E2E_CNN and PnP (psnr/ssim)

   test set: traffic3  train3  tuk-tuk7  upside-down4  walking12

| method  | binary mask    | combined binary mask |   gray mask    | combined gray mask |
| :-----: | -------------- | :------------------: | :------------: | :----------------: |
| E2E_CNN | 24.6777/0.7657 |    22.8083/0.6933    | 23.7212/0.7217 |   22.3240/0.6698   |
| PnP-SCI |                |                      |                |                    |





---

Appendix

1. E2E_CNN reproduction performance test (psnr/ssim)

   binary mask

    (Cr=10, retrained model from Zhihong Zhang) & (Cr=8, original model from Ziyi Meng)

   test set: traffic3  train3  tuk-tuk7  upside-down4  walking12

|       data        |    traffic3    |     train3     |    tuk-tuk7    |  upside-down4  |   walking12    |   [average]    |
| :---------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| **Cr8, original** | 26.5100/0.8721 | 24.9001/0.7140 | 25.3374/0.7892 | 24.2673/0.7690 | 23.6823/0.7664 | 24.9394/0.7821 |
| **Cr10, retrain** | 26.5834/0.8709 | 24.8993/0.7100 | 24.9191/0.7683 | 23.3844/0.7317 | 23.6022/0.7478 | 24.6777/0.7657 |

