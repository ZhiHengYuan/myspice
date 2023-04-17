import argparse
import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from spice.model.build_model_sim import build_model_sim
from fixmatch.train_utils import AverageMeter
from spice.solver import make_lr_scheduler, make_optimizer
from spice.config import Config
from spice.data.build_dataset import build_dataset
from spice.model.sim2sem import Sim2Sem
from spice.utils.miscellaneous import mkdir, save_config
import numpy as np
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
import copy
import math
import torch.nn.functional as F
import torch.nn as nn


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/stl10/train_ruc.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "--embedding",
    default="./results/stl10/embedding/feas_moco_512_l2.npy",
    type=str,
)

def extract_metric(net, p_label, evalloader, n_num):
    net.eval()
    feature_bank = []
    pool = nn.AdaptiveAvgPool2d(1)
    with torch.no_grad():
        for batch_idx, (inputs1 , _, _, _, indexes) in enumerate(evalloader):
            inputs1 = inputs1.cuda()
            if len(out.shape) == 4:
                out = pool(out)
                out = torch.flatten(out, start_dim=1)
            out = nn.functional.normalize(out, dim=1)
            feature_bank.append(out)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        sim_indices_list = []
        for batch_idx, (inputs1 , _, _, _, indexes) in enumerate(evalloader):
            inputs1 = inputs1.cuda()
            out = net(inputs1)
            if len(out.shape) == 4:
                out = pool(out)
                out = torch.flatten(out, start_dim=1)
            out = nn.functional.normalize(out, dim=1)
            sim_matrix = torch.mm(out, feature_bank)
            _, sim_indices = sim_matrix.topk(k=n_num, dim=-1)
            sim_indices_list.append(sim_indices)
        feature_labels = p_label
        first = True
        count = 0
        clean_num = 0
        correct_num = 0
        for batch_idx, (inputs1 , _, _, targets, indexes) in enumerate(evalloader):
            targets = targets.cuda()
            labels = p_label[indexes].long()
            sim_indices = sim_indices_list[count]
            sim_labels = torch.gather(feature_labels.expand(inputs1.size(0), -1), dim=-1, index=sim_indices)
            # counts for each class
            one_hot_label = torch.zeros(inputs1.size(0) * sim_indices.size(1), 10).cuda()
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(inputs1.size(0), -1, 10), dim=1)
            count += 1
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            prob, _ = torch.max(F.softmax(pred_scores, dim=-1), 1)     
            # Check whether prediction and current label are same
            noisy_label = labels
            s_idx1 = (pred_labels[:, :1].float() == labels.unsqueeze(dim=-1).float()).any(dim=-1).float()
            s_idx = (s_idx1 == 1.0)
            clean_num += labels[s_idx].shape[0]
            correct_num += torch.sum((labels[s_idx].float() == targets[s_idx].float())).item()
            
            if first:
                prob_set = prob
                pred_same_label_set = s_idx
                first = False
            else:
                prob_set = torch.cat((prob_set, prob), dim = 0)
                pred_same_label_set = torch.cat((pred_same_label_set, s_idx), dim = 0)
        print(correct_num, clean_num)
        return pred_same_label_set
     
def extract_confidence(net, p_label, evalloader, threshold):
    print(p_label)
    net.eval()
    devide = torch.tensor([]).cuda()
    clean_num = 0
    correct_num = 0
    for batch_idx, (inputs1, _, _, targets, indexes) in enumerate(evalloader):
        labels = p_label[indexes]
        inputs1, targets = inputs1.cuda(), targets.cuda()
        logits = net(inputs1)
        prob = torch.softmax(logits[0].detach_(), dim=-1)
        max_probs, _ = torch.max(prob, dim=-1)
        mask = max_probs.ge(threshold).float()
        devide = torch.cat([devide, mask])
        s_idx = (mask == 1)
        clean_num += labels[s_idx].shape[0]
        correct_num += torch.sum((labels[s_idx] == targets[s_idx])).item()
    print("confidence devide:", devide)
    print(correct_num, clean_num)
    return devide

def extract_hybrid(devide1, devide2, p_label, evalloader):
    devide = (devide1.float() + devide2.float() == 2)
    clean_num = 0
    correct_num = 0
    for batch_idx, (inputs1, _, _, targets, indexes) in enumerate(evalloader):
        targets = targets.cuda()
        labels = p_label[indexes].float()
        mask = devide[indexes]
        s_idx = (mask == 1)
        clean_num += labels[s_idx].shape[0]
        correct_num += torch.sum((labels[s_idx] == targets[s_idx])).item()
    print("extract_hybrid devide: ", devide)
    print(correct_num, clean_num)
    return devide

def adjust_learning_rate(args, optimizer, epoch):
    # cosine learning rate schedule
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_threshold(current):
    return 0.9 + 0.02*int(current / 40)


def linear_rampup(current, rampup_length=200):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip((current) / rampup_length, 0.1, 1.0)
        return float(current)

class criterion_rb(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        # Clean sample Loss
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = 50*torch.mean((probs_u - targets_u)**2)
        Lu = linear_rampup(epoch) * Lu
        return Lx, Lu
    

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.long().unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


LSloss = LabelSmoothLoss(0.5)

def train(epoch, net, net2, trainloader, optimizer, criterion_rb, devide, p_label, conf, cfg):
    train_loss = AverageMeter()
    net.train()
    net2.train()
    
    num_iter = (len(trainloader.dataset)//cfg.batch_size)+1
    # adjust learning rate
    adjust_learning_rate(cfg, optimizer, epoch)  
    optimizer.zero_grad()
    correct_u = 0
    unsupervised = 0
    conf_self = torch.zeros(len(trainloader.dataset))
    for batch_idx, (inputs1 , inputs2, inputs3, inputs4, targets, indexes) in enumerate(trainloader):
        inputs1, inputs2, inputs3, inputs4, targets = inputs1.float().cuda(), inputs2.float().cuda(), inputs3.float().cuda(), inputs4.float().cuda(), targets.cuda().long()
        s_idx = (devide[indexes] == 1)
        u_idx = (devide[indexes] == 0)
       
        labels = p_label[indexes].cuda().long()
        labels_x = torch.tensor(p_label[indexes][s_idx]).squeeze().long().cpu()
        target_x = torch.zeros(labels_x.shape[0], 10).scatter_(1, labels_x.view(-1,1), 1).float().cuda()
        
        logit_o, logit_w1, logit_w2, logit_s = net(inputs1)[0], net(inputs2)[0], net(inputs3)[0], net(inputs4)[0]
        logit_s = logit_s[s_idx]
        max_probs, _ = torch.max(torch.softmax(logit_o, dim=1), dim=-1)
        conf_self[indexes] = max_probs.detach().cpu()
        optimizer.zero_grad()
        
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u11 = logit_w1[u_idx]
            outputs_u21  = logit_w2[u_idx]
            logit_o2 = net2(inputs1)[0]
            logit_w12 = net2(inputs2)[0]
            logit_w22 = net2(inputs3)[0]
            outputs_u12 = logit_w12[u_idx]
            outputs_u22  = logit_w22[u_idx]
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu**(1/0.5) # temparature sharpening
            target_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            target_u = target_u.detach().float() 
            
            px = torch.softmax(logit_o2[s_idx], dim=1)
            w_x = conf[indexes][s_idx.cpu()]
            w_x = w_x.view(-1,1).float().cuda() 
            px = (1-w_x)*target_x + w_x*px              
            ptx = px**(1/0.5) # temparature sharpening           
            target_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            target_x = target_x.detach().float()      
            
            if logit_o[u_idx].shape[0] > 0: 
                max_probs, targets_u1 = torch.max(torch.softmax(logit_o[u_idx], dim=1), dim=-1)
                thr = get_threshold(epoch)    
                mask_u = max_probs.ge(thr).float()
                u_idx2 = (mask_u == 1)
                unsupervised += torch.sum(mask_u).item()
                correct_u += torch.sum((targets_u1[u_idx2] == targets[u_idx][u_idx2])).item()
                update = indexes[u_idx.cpu()][u_idx2.cpu()]
                devide[update] = True
                p_label[update] = targets_u1[u_idx2].float()
        
        
        l = np.random.beta(4.0, 4.0)        
        l = max(l, 1-l)
        
        all_inputs = torch.cat([inputs2[s_idx], inputs3[s_idx], inputs2[u_idx], inputs3[u_idx]],dim=0)
        all_targets = torch.cat([target_x, target_x, target_u, target_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)[0]
        batch_size = target_x.shape[0]
        
        Lx, Lu = criterion_rb(logits[:batch_size*2], mixed_target[:batch_size*2], logits[batch_size*2:], mixed_target[batch_size*2:], epoch+batch_idx/num_iter)
        total_loss = Lx + Lu + LSloss(logit_s, labels_x.cuda())
        
        total_loss.backward()
        train_loss.update(total_loss.item(), inputs2.size(0))
        optimizer.step()
        if batch_idx % 10 == 0:
          print('Epoch: [{epoch}][{elps_iters}/{tot_iters}] '
                'Train loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '.format(
                    epoch=epoch, elps_iters=batch_idx,tot_iters=len(trainloader), 
                    train_loss=train_loss))
    conf_self = (conf_self - conf_self.min()) / (conf_self.max() - conf_self.min())
    return train_loss.avg, devide, p_label, conf_self


def main():
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)

    cfg.embedding = args.embedding

    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)

    output_config_path = os.path.join(output_dir, 'config.py')
    save_config(cfg, output_config_path)

    if cfg.gpu is not None:
        print("Use GPU: {}".format(cfg.gpu))

    # create model
    net = Sim2Sem(**cfg.model)
    net2 = copy.deepcopy(net)
    net_uc = copy.deepcopy(net)
    net_embd = build_model_sim(cfg.model_sim)

    try:
        state_dict = torch.load(cfg.model.o_model)
        state_dict2 = torch.load(cfg.model.e_model)
        net_uc.load_state_dict(state_dict, strict = False)
        net_embd.load_state_dict(state_dict2, strict = False)
        net.load_state_dict(state_dict, strict = False)
        net2.load_state_dict(state_dict, strict = False)
    except RuntimeError as e:
        print(e)
        exit(0)

    torch.cuda.set_device(cfg.gpu)
    net = net.cuda(cfg.gpu)
    net2 = net2.cuda(cfg.gpu)
    net_uc = net_uc.cuda(cfg.gpu)
    net_embd = net_embd.cuda(cfg.gpu) 

    cudnn.benchmark = True

    optimizer1 = torch.optim.SGD(net.parameters(), cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)
    optimizer2 = torch.optim.SGD(net2.parameters(), cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)


    # Data loading code
    dataset_val = build_dataset(cfg.data_test)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=1)
    print("len(dataset_val): ", len(dataset_val))
     # Data loading code
    dataset_train = build_dataset(cfg.data_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=False, num_workers=1)
    print("len(dataset_train):　　", len(dataset_train))
    # Extract Pseudo Label
    net.eval()
    num_heads = len(cfg.model.head.multi_heads)
    assert num_heads == 1
    gt_labels = []
    pred_labels = []
    scores_all = []
    for _, (images, _,_, labels, idx) in enumerate(val_loader):
        images = images.to(cfg.gpu, non_blocking=True)
        with torch.no_grad():
            scores = net(images, forward_type="sem")

        assert len(scores) == num_heads

        pred_idx = scores[0].argmax(dim=1)
        pred_labels.append(pred_idx)
        scores_all.append(scores[0])

        gt_labels.append(labels)

    # Divide Clean and Noisy set
    pred_labels = torch.cat(pred_labels).float()
    devide2 = extract_metric(net_embd, pred_labels, val_loader, cfg.n_num)
    devide1 = extract_confidence(net_uc, pred_labels, val_loader, cfg.s_thr)
    devide = extract_hybrid(devide1, devide2, pred_labels, val_loader)
    # print("devide: ", devide)

    # gt_labels = torch.cat(gt_labels).long().cpu().numpy()
    # feas_sim = torch.from_numpy(np.load(cfg.embedding))

    # pred_labels = pred_labels.cpu().numpy()
    # scores = torch.cat(scores_all).cpu()

    # try:
    #     acc = calculate_acc(pred_labels, gt_labels)
    # except:
    #     acc = -1

    # nmi = calculate_nmi(pred_labels, gt_labels)
    # ari = calculate_ari(pred_labels, gt_labels)

    # print("ACC: {}, NMI: {}, ARI: {}".format(acc, nmi, ari))

    criterion = criterion_rb()
    print(len(train_loader.dataset))
    conf1 =  torch.zeros(len(train_loader.dataset))
    conf2 =  torch.zeros(len(train_loader.dataset))

    for epoch in range(cfg.epochs):
        print("== Train RUC ==")
        loss, devide, p_label, conf1 = train(epoch, net, net2, train_loader, optimizer1, criterion, devide, pred_labels, conf2, cfg)
        loss, devide, p_label, conf2 = train(epoch, net2, net, train_loader, optimizer2, criterion, devide, pred_labels, conf1, cfg)
        # acc, p_list = test_ruc(net, net2, val_loader, cfg.device, class_num)
        # print("accuracy: {}\n".format(acc))

        # state = {'net1': net.state_dict(),
        #             'net2': net2.state_dict() }
        # torch.save(state, './checkpoint/ruc_stl10.t7')

    # idx_select, labels_select = model(feas_sim=feas_sim, scores=scores, forward_type="local_consistency")

    # gt_labels_select = gt_labels[idx_select]

    # acc = calculate_acc(labels_select, gt_labels_select)
    # print('ACC of local consistency: {}, number of samples: {}'.format(acc, len(gt_labels_select)))

    # labels_correct = np.zeros([feas_sim.shape[0]]) - 100
    # labels_correct[idx_select] = labels_select

    # np.save("{}/labels_reliable.npy".format(cfg.results.output_dir), labels_correct)


if __name__ == '__main__':
    main()
