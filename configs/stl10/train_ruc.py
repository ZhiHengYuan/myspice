import torchvision.transforms as transforms
from randaugment import RandAugmentMC
model_name = "eval"
# weight = './model_zoo/self_model_stl10.pth.tar'
weight = './results/stl10/spice_self/checkpoint_final.pth.tar'
o_model = './results/stl10/spice_self/checkpoint_final.pth.tar'
e_model = './results/stl10/spice_self/checkpoint_final.pth.tar'
model_type = "clusterresnet"
# model_type = 'resnet18'
num_cluster = 10
batch_size = 100
fea_dim = 512
center_ratio = 0.5
world_size = 1
workers = 4
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = 0
multiprocessing_distributed = False
lr = 0.01
momentum = 0.9
weight_decay = 5e-4
epochs = 200
s_thr = 0.99
n_num = 100
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
data_test = dict(
    type="stl10",
    root_folder="./datasets/stl10",
    embedding=None,
    split="train+test",
    shuffle=False,
    ims_per_batch=1,
    aspect_ratio_grouping=False,
    train=False,
    show=False,
    trans1 = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    trans2 = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    trans3 = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
)

data_train = dict(
    type="stl10",
    root_folder="./datasets/stl10",
    embedding=None,
    split="train+test",
    shuffle=False,
    ims_per_batch=1,
    aspect_ratio_grouping=False,
    train=False,
    show=False,
    trans1 = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    trans2 = transforms.Compose([
        transforms.RandomResizedCrop(size=96, scale=(0.2,1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    trans3 = transforms.Compose([
        transforms.RandomResizedCrop(size=96, scale=(0.2,1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    trans4 = transforms.Compose([
            transforms.RandomResizedCrop(size=96, scale=(0.2,1.)),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

)

model = dict(
    feature=dict(
        type=model_type,
        num_classes=num_cluster,
        in_channels=3,
        in_size=96,
        batchnorm_track=True,
        test=False,
        feature_only=True
    ),

    head=dict(type="sem_multi",
              multi_heads=[dict(classifier=dict(type="mlp", num_neurons=[fea_dim, fea_dim, num_cluster], last_activation="softmax"),
                                feature_conv=None,
                                num_cluster=num_cluster,
                                ratio_start=1,
                                ratio_end=1,
                                center_ratio=center_ratio,
                                )]*1,
              ratio_confident=0.90,
              num_neighbor=100,
              ),
    model_type="moco_select",
    o_model = o_model,
    e_model = e_model,
    pretrained=weight,
    head_id=3,
    freeze_conv=True,
)

model_sim = dict(
    type=model_type,
    num_classes=128,
    in_channels=3,
    in_size=96,
    batchnorm_track=True,
    test=False,
    feature_only=True,
    pretrained=e_model,
    model_type="moco_embedding",
)

results = dict(
    output_dir="./results/stl10/{}".format(model_name),
)