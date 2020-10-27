import os

# CLI script


def cli_run(orig_name, scale, Cr, mask_name, test_algo_flag,
            result_path='/results/tmp', iframe=0, nframe=1, MAXB=255,
            show_res_flag=0, save_res_flag=0,
            tv_weight=None, iter_max1=None, sigma1=None, iter_max2=None, sigma2=None):
    print(('python pnp_sci_video_orig_simuexp1.py \
            --orig_name {} \
            --scale {} \
            --Cr {} \
            --mask_name {} \
            --test_algo_flag {} \
            --result_path {} \
            --iframe {} \
            --nframe {} \
            --MAXB {} \
            --show_res_flag {} \
            --save_res_flag {} \
            --tv_weight {:.2f} \
            --iter_max1 {} \
            --sigma1 {} ' +
           '--iter_max2' + (' {} '*len(iter_max2)) +
           '--sigma2' + (' {:.4f} '*len(sigma2)))
          .format(orig_name, scale, Cr, mask_name, test_algo_flag,  result_path, iframe, nframe, MAXB,
                  show_res_flag, save_res_flag, tv_weight, iter_max1, sigma1, *iter_max2, *sigma2)
          )

    os.system(('python pnp_sci_video_orig_simuexp1.py \
            --orig_name {} \
            --scale {} \
            --Cr {} \
            --mask_name {} \
            --test_algo_flag {} \
            --result_path {} \
            --iframe {} \
            --nframe {} \
            --MAXB {} \
            --show_res_flag {} \
            --save_res_flag {} \
            --tv_weight {:.2f} \
            --iter_max1 {} \
            --sigma1 {} ' +
               '--iter_max2' + (' {} '*len(iter_max2)) +
               '--sigma2' + (' {:.4f} '*len(sigma2)))
              .format(orig_name, scale, Cr, mask_name, test_algo_flag,  result_path, iframe, nframe, MAXB,
                      show_res_flag, save_res_flag, tv_weight, iter_max1, sigma1, *iter_max2, *sigma2)
              )
