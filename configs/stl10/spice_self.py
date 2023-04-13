model_name = "spice_self"
pre_model = "/kaggle/input/modelfile/results/stl10/moco/checkpoint_180.pth.tar"
embedding = "/kaggle/input/modelfile/results/stl10/embedding/feas_moco_512_l2.npy"
resume = "/kaggle/output/working/results/stl10/{}/checkpoint_last.pth.tar".format(model_name)
model_type = "clusterresnet"
num_head = 10
num_workers = 1
device_id = 0
num_train = 5
num_cluster = 10
batch_size = 1000
target_sub_batch_size = 100
train_sub_batch_size = 128
batch_size_test = 100
num_trans_aug = 1
num_repeat = 8
fea_dim = 512
att_conv_dim = num_cluster
att_size = 7
center_ratio = 0.5
sim_center_ratio = 0.9
epochs = 100
world_size = 1
workers = 1
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = 0
multiprocessing_distributed = False

start_epoch = 0
print_freq = 1
test_freq = 1
eval_ent = False
eval_ent_weight = 0

data_train = dict(
    type="stl10_emb",
    root_folder="/kaggle/input/stl10-binary/stl10",
    embedding=embedding,
    split="train+test",
    ims_per_batch=batch_size,
    shuffle=True,
    aspect_ratio_grouping=False,
    train=True,
    show=False,
    trans1=dict(
        aug_type="weak",
        crop_size=96,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),

    trans2=dict(
        aug_type="scan",
        crop_size=96,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        num_strong_augs=4,
        cutout_kwargs=dict(n_holes=1,
                           length=32,
                           random=True)
    ),
)

data_test = dict(
    type="stl10_emb",
    root_folder="/kaggle/input/stl10-binary/stl10",
    embedding=embedding,
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
                                loss_weight=dict(loss_cls=1),
                                iter_start=epochs,
                                iter_up=epochs,
                                iter_down=epochs,
                                iter_end=epochs,
                                ratio_start=1.0,
                                ratio_end=1.0,
                                center_ratio=center_ratio,
                                )]*num_head,
              ),
    model_type="moco",
    pretrained=pre_model,
    freeze_conv=True,
)


solver = dict(
    type="adam",
    base_lr=0.001,
    bias_lr_factor=1,
    weight_decay=0,
    weight_decay_bias=0,
    target_sub_batch_size=target_sub_batch_size,
    batch_size=batch_size,
    train_sub_batch_size=train_sub_batch_size,
    num_repeat=num_repeat,
)

results = dict(
    output_dir="/kaggle/output/working/results/stl10/{}".format(model_name),
)