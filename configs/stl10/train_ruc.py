import torchvision.transforms as transforms
model_name = "eval"
# weight = './model_zoo/self_model_stl10.pth.tar'
weight = '../results/stl10/spice_self/checkpoint_best.pth.tar'
o_model = '../results/stl10/spice_self/checkpoint_best.pth.tar'
e_model = '../results/stl10/moco/checkpoint_final.pth.tar'
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
s_thr = 0.10
n_num = 100
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
data_test = dict(
    type="RUC_stl10_test",
    root_folder="../datasets/stl10",
    embedding=None,
    split="train+test",
    shuffle=False,
    ims_per_batch=1,
    aspect_ratio_grouping=False,
    train=False,
    show=False,
    trans1 = dict(
        aug_type="RUC_test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans2 = dict(
        aug_type="RUC_test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans3 = dict(
        aug_type="RUC_test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
)

data_train = dict(
    type="RUC_stl10_train",
    root_folder="../datasets/stl10",
    embedding=None,
    split="train+test",
    shuffle=False,
    ims_per_batch=1,
    aspect_ratio_grouping=False,
    train=False,
    show=False,
    trans1 = dict(
        aug_type="RUC_test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans2 = dict(
        aug_type="RUC_train",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans3 = dict(
        aug_type="RUC_train",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans4 = dict(
        aug_type="RUC_strong",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),

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
    output_dir="../results/stl10/{}".format(model_name),
)