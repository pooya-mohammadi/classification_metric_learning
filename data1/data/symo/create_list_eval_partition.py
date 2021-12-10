import os
import shutil
from os.path import join
import random
from PIL import Image

train_prob = 0.8
query_prob = 0.5

main_dir = 'img'
txt_path = 'list_eval_partition.txt'
txt_file = open(txt_path, mode='w')
for main_category in os.listdir(main_dir):
    print(f'[INFO] processing main_category:{main_category}')
    category_path = join(main_dir, main_category)
    for sub_category in os.listdir(category_path):
        print(f'[INFO] processing sub_category"{sub_category}')
        sub_category_path = join(category_path, sub_category)
        if random.random() < train_prob:
            for img_name in os.listdir(sub_category_path):
                img_path = join(sub_category_path, img_name)
                if " " in img_name:
                    new_img_name = img_name.replace(" ", "")
                    new_img_path = join(sub_category_path, new_img_name)
                    shutil.move(img_path, new_img_path)
                    img_path = new_img_path
                    print(f'[INFO] rename {img_path} to {new_img_path}')
                try:
                    Image.open(img_path).convert('RGB')
                except OSError:
                    print(f'[INFO] truncation error for {img_path}, skipping')
                    continue

                record = f'{img_path}                      {sub_category_path} train\n'
                txt_file.write(record)
        else:
            for img_name in os.listdir(sub_category_path):
                img_path = join(sub_category_path, img_name)
                if " " in img_name:
                    new_img_name = img_name.replace(" ", "")
                    new_img_path = join(sub_category_path, new_img_name)
                    shutil.move(img_path, new_img_path)
                    img_path = new_img_path
                    print(f'[INFO] rename {img_path} to {new_img_path}')
                if random.random() < query_prob:
                    record = f'{img_path}                      {sub_category_path} query\n'
                else:
                    record = f'{img_path}                      {sub_category_path} gallery\n'
                txt_file.write(record)
                try:
                    Image.open(img_path).convert('RGB')
                except OSError:
                    print(f'[INFO] truncation error for image {img_path}, skipping')
                    continue
print('[INFO] It is done!')
