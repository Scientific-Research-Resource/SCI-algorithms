import os

# CLI script


def cli_run(script_name, orig_name, scale, Cr, mask_name, test_algo_flag,
            root_dir = '.', result_path='/results/tmp', iframe=0, nframe=1, MAXB=255,
            show_res_flag=0, save_res_flag=0, log_result_flag=0,
            gaussian_noise_level=0, poisson_noise=0, gamma=0,
            tv_weight=None, iter_max1=0, sigma1=0, iter_max2=[0], sigma2=[0]):

    command_str = ('python {} \
        --orig_name {} \
        --scale {} \
        --Cr {} \
        --mask_name {} \
        --test_algo_flag {} \
        --root_dir {} \
        --result_path {} \
        --iframe {} \
        --nframe {} \
        --MAXB {} \
        --show_res_flag {} \
        --save_res_flag {} \
        --log_result_flag {} \
        --gaussian_noise_level {} \
        --poisson_noise {} \
        --gamma {} \
        --tv_weight {:.2f} \
        --iter_max1 {} \
        --sigma1 {} ' +
        '--iter_max2 ' + (' {} '*len(iter_max2)) +
        '--sigma2 ' + (' {:.4f} '*len(sigma2))).format(script_name, orig_name, scale, Cr, mask_name, test_algo_flag, root_dir, result_path, iframe, nframe, MAXB,
                show_res_flag, save_res_flag, log_result_flag, gaussian_noise_level, poisson_noise, gamma, tv_weight, iter_max1, sigma1, *iter_max2, *sigma2)
        
    print(command_str)
    os.system(command_str)