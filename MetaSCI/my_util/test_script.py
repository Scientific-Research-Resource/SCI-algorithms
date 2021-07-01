import plot_util
import numpy as np
imgs = np.random.randint(0,255,size=(256,256,3,8))
imgs[...,[1,4,7]]=0
titles=['1','2','3','4','5','6','7','8']
save_dir = './'
save_name='img'
plot_util.plot_multi(imgs, 'test', titles=titles, col_num=4 savename=save_name, savedir=save_dir)