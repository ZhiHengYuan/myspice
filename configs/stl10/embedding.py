model_name = "embedding"
weight = "/kaggle/input/d/zhihengyuan/modelfile/results/stl10/moco/checkpoint_0105.pth.tar"
model_type = "clusterresnet"
num_cluster = 10
batch_size = 1000
world_size = 1
workers = 4
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = 0
multiprocessing_distributed = True

data_test = dict(
    type="stl10",
    root_folder="/kaggle/input/stl10-binary-files",
    split="train+test",
    shuffle=False,
    ims_per_batch=50,
    aspect_ratio_grouping=False,
    train=False,
    show=False,
    trans1=dict(
        aug_type="test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans2=dict(
        aug_type="test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),

)


model_sim = dict(
    type=model_type,
    num_classes=128,
    in_channels=3,
    in_size=96,
    batchnorm_track=True,
    test=False,
    feature_only=True,
    pretrained=weight,
    model_type="moco_embedding",
)


results = dict(
    output_dir="/kaggle/working/results/stl10/{}".format(model_name),
)