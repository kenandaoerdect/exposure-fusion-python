from image_io import load_img, save_img
import os
from exposure_fusion import exposure_fusion

input_path = 'data/case5'   # 输入的不同曝光的图片
save_img_name = ''.join(['hdr_output_', os.path.split(input_path)[-1], '.png'])   # output图片文件名
nlev = 9    # 金字塔层数，不能设置过小，否则引起分层

input_path_list = []
for i in os.listdir(input_path):
    input_path_list.append(os.path.join(input_path, i))

I = load_img(input_path_list)
R = exposure_fusion(I, [1, 1, 1, nlev])

save_img(R * 255, save_img_name)
