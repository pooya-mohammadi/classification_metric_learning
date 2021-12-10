import glob
import os
import sys
from deep_utils import dump_pickle, load_pickle
import time
from itertools import chain
from argparse import ArgumentParser
import torch
from pretrainedmodels.utils import ToRange255
from pretrainedmodels.utils import ToSpaceBGR
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from data.inshop import InShop
from metric_learning.util import SimpleLogger
from metric_learning.sampler import ClassBalancedBatchSampler
from PIL import Image
import metric_learning.modules.featurizer as featurizer
import metric_learning.modules.losses as losses
import numpy as np
from evaluation.retrieval import evaluate_float_binary_embedding_faiss, _retrieve_knn_faiss_gpu_inner_product

dataset = "InShop"
dataset_root = ""
batch_size = 64
model_name = "resnet50"
lr = 0.01
gamma = 0.1
class_balancing = True
images_per_class = 5
lr_mult = 1
dim = 2048

test_every_n_epochs = 2
epochs_per_step = 4
pretrain_epochs = 1
num_steps = 3
output = "data1/output"


def adjust_learning_rate(optimizer, epoch, epochs_per_step, gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every epochs"""
    # Skip gamma update on first epoch.
    if epoch != 0 and epoch % epochs_per_step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
            print("learning rate adjusted: {}".format(param_group['lr']))


def main():
    torch.cuda.set_device(0)
    gpu_device = torch.device('cuda')

    output_directory = os.path.join(output, dataset, str(dim),
                                    '_'.join([model_name, str(batch_size)]))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    out_log = os.path.join(output_directory, "train.log")
    sys.stdout = SimpleLogger(out_log, sys.stdout)

    # Select model
    model_factory = getattr(featurizer, model_name)
    model = model_factory(dim)
    weights = torch.load(
        '/home/ai/projects/symo/classification_metric_learning/data1/output/InShop/2048/resnet50_75/epoch_30.pth')
    model.load_state_dict(weights)
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(max(model.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(model.input_space == 'BGR'),
        ToRange255(max(model.input_range) == 255),
        transforms.Normalize(mean=model.mean, std=model.std)
    ])

    # Setup dataset

    # train_dataset = InShop('../data1/data/inshop', transform=train_transform)
    query_dataset = InShop('data1/data/inshop', train=False, query=True, transform=eval_transform)
    index_dataset = InShop('data1/data/inshop', train=False, query=False, transform=eval_transform)

    query_loader = DataLoader(query_dataset,
                              batch_size=batch_size,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0)

    model.to(device='cuda')
    model.eval()
    query_image = Image.open(
        "/home/ai/Pictures/im3.png").convert(
        'RGB')
    with torch.no_grad():
        query_image = model(eval_transform(query_image).to('cuda').unsqueeze(0))[0].cpu().numpy()

    index_dataset = InShop('data1/data/inshop', train=False, query=False, transform=eval_transform)
    index_loader = DataLoader(index_dataset,
                              batch_size=75,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0)
    # db_list = extract_feature(model, index_loader, 'cuda')
    db_list = load_pickle('db.pkl')
    # db_dirs = [
    #     "/home/ai/projects/symo/classification_metric_learning/data1/data/inshop/img/WOMEN/Blouses_Shirts/id_00000001",
    #     "/home/ai/projects/symo/classification_metric_learning/data1/data/inshop/img/WOMEN/Blouses_Shirts/id_00000004",
    #     "/home/ai/projects/symo/classification_metric_learning/data1/data/inshop/img/WOMEN/Blouses_Shirts/id_00000038",
    #     "/home/ai/projects/symo/classification_metric_learning/data1/data/inshop/img/WOMEN/Blouses_Shirts/id_00000067",
    # ]
    # db_list = {}
    # with torch.no_grad():
    #
    #     for dir_ in db_dirs:
    #         for n in os.listdir(dir_):
    #             img_path = os.path.join(dir_, n)
    #             img = Image.open(img_path)
    #             db_list[img_path] = model(eval_transform(img).unsqueeze(0)).cpu().numpy()[0]
    v = get_most_similar(query_image, db_list)
    print(v)


def get_most_similar(feature, features_dict, n=10, distance='cosine'):
    features = list(features_dict.values())
    ids = list(features_dict.keys())
    p = cdist(np.array(features),
              np.expand_dims(feature, axis=0),
              metric=distance)[:, 0]
    group = zip(p, ids.copy())
    res = sorted(group, key=lambda x: x[0])
    r = res[:n]
    return r


def extract_feature(model, loader, gpu_device):
    """
    Extract embeddings from given `model` for given `loader` dataset on `gpu_device`.
    """
    model.eval()
    model.to(gpu_device)
    db_dict = {}
    log_every_n_step = 10

    with torch.no_grad():
        for i, (im, class_label, instance_label, index) in enumerate(loader):
            im = im.to(device=gpu_device)
            embedding = model(im)
            for i, em in zip(index, embedding):
                db_dict[loader.dataset.image_paths[int(i)]] = em.detach().cpu().numpy()
            if (i + 1) % log_every_n_step == 0:
                print('Process Iteration {} / {}:'.format(i, len(loader)))
    dump_pickle('db.pkl', db_dict)
    return db_dict


if __name__ == '__main__':
    main()
