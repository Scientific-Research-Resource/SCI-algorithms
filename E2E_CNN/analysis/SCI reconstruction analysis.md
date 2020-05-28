# SCI reconstruction analysis

1. E2E_CNN training info (Cr=10)

   | train/valid | binary mask | combine binary mask | gray mask | combine gray mask |
   | ----------- | :---------: | :-----------------: | :-------: | :---------------: |
   | train       |    28.3     |        26.3         |   27.3    |       25.7        |
   | valid       |    23.9     |        22.7         |   23.5    |        22         |

   

2. reconstruction performance comparison between E2E_CNN and PnP (test set; psnr/ssim)

| method  | binary mask | combined binary mask |   gray mask    | combined gray mask |
| :-----: | ----------- | :------------------: | :------------: | :----------------: |
| E2E_CNN |             |    22.8083/0.6933    | 23.7212/0.7217 |   22.3240/0.6698   |
| PnP-SCI |             |                      |                |                    |





---

Appendix

1. E2E_CNN test info (Cr=8, original model from ziyi meng)

|   data   | traffic3 | train3  | tuk-tuk7 | upside-down4 | walking12 | [average] |
| :------: | :------: | :-----: | :------: | :----------: | :-------: | :-------: |
| **psnr** | 26.5100  | 24.9001 | 25.3374  |   24.2673    |  23.6823  |  24.9394  |
| **ssim** |  0.8721  | 0.7140  |  0.7892  |    0.7690    |  0.7664   |  0.7821   |

