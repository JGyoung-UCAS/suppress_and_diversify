from ast import Num, arg
import torch
import os
import datetime
import pandas as pd
import argparse
import yaml
import time
import torch.nn as nn
import torchvision
import self_transforms
import torchvision.models as models
import utils
import warnings
from collections import defaultdict,deque
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import math
import timm
import random
from torch.utils.data import Dataset
import numpy as np
import random
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(description='Multi-GPU Training with PyTorch and Timm')
parser.add_argument('--yaml', type=str, default='./', help='.')
args = parser.parse_args()
yaml_path = args.yaml

global cfg
with open(yaml_path,'r') as f:
    cfg = yaml.safe_load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_ids']


import torch.nn.functional as F

class SNDRefiner:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.best_fitness = -1.0
        self.best_conv1_weight = None
        self.best_channel_scores = None
        self.lambda_val = cfg.get('snd_lambda', 1.0)
        self.spts_epoch = cfg.get('spts_epoch', 5) 

    def calculate_cka(self, feat1, feat2):
        feat1 = feat1.view(feat1.size(0), -1)
        feat2 = feat2.view(feat2.size(0), -1)
        

        feat1 = feat1 - feat1.mean(dim=0)
        feat2 = feat2 - feat2.mean(dim=0)
        
        dot_product = torch.norm(torch.matmul(feat1, feat2.t()))**2
        norm1 = torch.norm(torch.matmul(feat1, feat1.t()))
        norm2 = torch.norm(torch.matmul(feat2, feat2.t()))
        
        if norm1 == 0 or norm2 == 0: return 0.0
        return (dot_product / (norm1 * norm2)).item()

    def contrast_distortion(self, x):

        contrast_factor = random.uniform(0.5, 1.5) 
        return TF.adjust_contrast(x, contrast_factor)
    
    def get_corrupted_input(self, x_norm):

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_norm.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_norm.device)
        

        x = x_norm * std + mean
        x = torch.clamp(x, 0, 1)
        

        noise = torch.randn_like(x) * 0.05  
        x = x + noise
        

        import random
        factor = random.uniform(0.6, 1.4)
        x = TF.adjust_contrast(x, factor)
        

        x = torch.clamp(x, 0, 1)
        return (x - mean) / std

    def get_fitness_score(self, model, val_loader, epoch, total_epochs):
        model.eval()
        samples = []
        target_num = 1000
        
        max_batches = (target_num // self.cfg.get('batch_size', 128)) + 1
        for i, (imgs, _) in enumerate(val_loader):
            if i >= max_batches: break 
            samples.append(imgs)

        x_all = torch.cat(samples, dim=0)
        actual_num = min(x_all.size(0), target_num)
        x_clean = x_all[:actual_num].to(self.device)
        x_corrupt = self.get_corrupted_input(x_clean)

        real_model = model.module if hasattr(model, 'module') else model
        
        with torch.no_grad():
            def get_stem_output(x):
                x = real_model.conv1(x)
                x = real_model.bn1(x)
                return real_model.relu(x)
            
            f_clean = get_stem_output(x_clean)
            f_corrupt = get_stem_output(x_corrupt)
            

            channel_scores_list = []
            valid_cka_list = []
            
            for c in range(f_clean.size(1)):
                if torch.max(f_clean[:, c]) == 0:
                    c_score = 0.0
                else:
                    c_score = self.calculate_cka(f_clean[:, c:c+1], f_corrupt[:, c:c+1])
                    valid_cka_list.append(c_score)
                channel_scores_list.append(c_score)
            

            R_total = np.mean(valid_cka_list) if len(valid_cka_list) > 0 else 0.0
            

            res_tensor = torch.tensor([R_total], device=self.device, dtype=torch.float32)

            ch_tensor = torch.tensor(channel_scores_list, device=self.device, dtype=torch.float32)
            
            if dist.is_initialized():
                dist.barrier() 
                dist.all_reduce(res_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(ch_tensor, op=dist.ReduceOp.SUM)
                
                R_total = res_tensor.item() / dist.get_world_size()

                ch_tensor = ch_tensor / dist.get_world_size()


        sigmoid_term = 1 / (1 + math.exp(- (epoch / total_epochs)))
        fitness = R_total + self.lambda_val * sigmoid_term
        

        return fitness, ch_tensor.cpu(), real_model.conv1.weight.data.clone()

    def apply_spts(self, model):

        import torch.distributed as dist
        import numpy as np
        import random

        real_model = model.module if hasattr(model, 'module') else model
        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        device = next(real_model.parameters()).device

        updated_weights = real_model.conv1.weight.data.clone()


        if rank == 0:
            if self.best_conv1_weight is None:
                print("Warning: No best weights found in Memory Bank, skipping SPTS.")
            else:

                weights = self.best_conv1_weight.cpu().numpy()
                scores = self.best_channel_scores.numpy()
                
                num_c = weights.shape[0]
                K = num_c // 2
                
                sorted_indices = np.argsort(scores)[::-1]
                top_k_indices = list(sorted_indices[:K]) 
                discard_indices = list(sorted_indices[K:]) 
                
                new_weights_np = np.zeros_like(weights)

                for idx in top_k_indices:
                    new_weights_np[idx] = weights[idx]
                
                source_pool = top_k_indices.copy()
                random.shuffle(source_pool)
                pool_idx = 0
                
                for target_idx in discard_indices:
                    source_idx = source_pool[pool_idx]
                    w = weights[source_idx].copy()
                    
                    pool_idx += 1
                    if pool_idx >= len(source_pool):
                        random.shuffle(source_pool)
                        pool_idx = 0
                    
                    in_c_idx = np.arange(w.shape[0])
                    np.random.shuffle(in_c_idx)
                    w = w[in_c_idx]
                    
                    op = np.random.choice(['hflip', 'vflip', 'transpose', 'none'])
                    for in_c in range(w.shape[0]):
                        if op == 'hflip': w[in_c] = np.fliplr(w[in_c])
                        elif op == 'vflip': w[in_c] = np.flipud(w[in_c])
                        elif op == 'transpose': w[in_c] = w[in_c].T
                    
                    new_weights_np[target_idx] = w
                
                updated_weights = torch.from_numpy(new_weights_np).to(device).to(real_model.conv1.weight.dtype)


        if is_dist:
            dist.broadcast(updated_weights, src=0)

        with torch.no_grad():
            real_model.conv1.weight.copy_(updated_weights)

            real_model.conv1.weight.requires_grad = False
            
        if is_dist:
            dist.barrier()
            if rank == 0:
                print(f"Successfully synchronized and applied SPTS to conv1 on all {dist.get_world_size()} GPUs.")

        
def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  


        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"Random seed set to {seed}")
    else:
        print("No random seed provided; training will be non-deterministic.")
        
def main():

    device = cfg['device']

    train_crop_size = cfg['train_crop_size']
    test_resize_size = cfg['test_resize_size']
    test_crop_size = cfg['test_crop_size']
    mean = cfg['mean']
    std = cfg['std']
    model_name = cfg['arch']
    data_name = cfg['dataset']
    method = cfg['method']
    num_classes = cfg['num_classes']
    pretrained = cfg['pretrained']

    lr = cfg['lr']
    momentum = cfg['momentum']
    weight_decay = cfg['weight_decay']
    num_epochs = cfg['epochs']
    sync_bn = cfg['sync_bn']

    imagenet_path = cfg['imagenet_path']
    imagenet100_path = cfg['imagenet100_path']
    store_root = cfg['store_root']


    
    start_epoch = cfg['start_epoch']
    best_acc1 = 0.0

    
    init_distributed_mode(cfg)
    device = torch.device(cfg['device'])
    if cfg['use_deterministic_algorithms']:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        
    cfg['nprocs'] = cfg['world_size']   
    world_size = cfg['world_size']
    batch_size = int(cfg['batch_size']/world_size)
    
    if cfg['distributed']:
        if dist.get_rank() == 0:
            auto_seed = random.randint(1, 1000)
            auto_seed_tensor = torch.tensor(auto_seed, dtype=torch.long, device=device)
        else:
            auto_seed_tensor = torch.tensor(0, dtype=torch.long, device=device)
        dist.broadcast(auto_seed_tensor, src=0)
        auto_seed = auto_seed_tensor.item()
    else:
        auto_seed = random.randint(1, 1000)

    set_random_seed(auto_seed)
    print(f"Random seed set to {auto_seed}")


    if data_name == 'imagenet':
        cfg['data_root'] = imagenet_path
        
    elif data_name == 'imagenet100':
        cfg['data_root'] = imagenet100_path
        
    if 'hf_hub' in model_name:
        _,store_model_name = model_name.split('/')
    else:
        store_model_name = model_name
        
    if method in ['sd','sd+augmix']:
        cfg['store_dir'] = os.path.join(store_root,store_model_name+'_'+data_name+'_'+method+'_spts'+str(cfg['spts_epoch'])+'_s'+str(auto_seed)) 
    else:
        cfg['store_dir'] = os.path.join(store_root,store_model_name+'_'+data_name+'_'+method+'_s'+str(auto_seed))
    
    print('Loading data.')
    traindir = os.path.join(cfg['data_root'],'train')
    interpolation = InterpolationMode(cfg['interpolation'])
    
    
    print('Loading training data.')
    
    train_dataset = datasets.ImageFolder(
                    traindir,
                    ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    mean = mean,
                    std = std,
                    interpolation=interpolation,
                    auto_augment_policy=cfg['auto_augment'],
                    random_erase_prob=cfg['random_erase_prob'],
                    ra_magnitude=cfg['ra_magnitude'],
                    augmix_severity=cfg['augmix_severity']
                    ))
    

    print("Loading validation data")
    valdir = os.path.join(cfg['data_root'],'test')
    preprocessing = ClassificationPresetEval(
                crop_size=test_crop_size, resize_size=test_resize_size,
                mean = mean,
                std = std,
                interpolation=interpolation
            )
    val_dataset = datasets.ImageFolder(
        valdir,
        preprocessing)
    
    
    print('Creating data loaders.')
    
    
    if cfg['distributed']:
        if cfg['ra_sampler']:
            train_sampler = utils.RASampler(train_dataset, shuffle=True, repetitions=cfg['ra_reps'])
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    collate_fn = None
    mixup_transforms = []
    if cfg['mixup_alpha'] > 0.0:
        mixup_transforms.append(self_transforms.RandomMixup(num_classes, p=1.0, alpha=cfg['mixup_alpha']))
    if cfg['cutmix_alpha'] > 0.0:
        mixup_transforms.append(self_transforms.RandomCutmix(num_classes, p=1.0, alpha=cfg['cutmix_alpha']))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=cfg['num_workers'],
                                        pin_memory=True,
                                        sampler=train_sampler,
                                        collate_fn=collate_fn)
    
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                num_workers=cfg['num_workers'],
                                                pin_memory=True,
                                                sampler=val_sampler)
    
    
    print('Creating model')
    if pretrained:
        model = getattr(models,model_name)(weights=cfg['weights'],num_classes=num_classes)
    else:
        model = getattr(models,model_name)(weights=None,num_classes=num_classes)
        
    if cfg['model_path']:
        checkpoint = torch.load(cfg['model_path'], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        print('ckpt file loaded.')
        
    else:
        print('no ckpt file.')
             
    
    model.to(device)
    
    
    if cfg['distributed'] and cfg['sync_bn']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    
    custom_keys_weight_decay = []
    if cfg['bias_weight_decay'] is not None:
        custom_keys_weight_decay.append(("bias",cfg['bias_weight_decay']))
    if cfg['transformer_embedding_decay'] is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, cfg['transformer_embedding_decay']))
            
    parameters = utils.set_weight_decay(
        model,
        cfg['weight_decay'],
        norm_weight_decay=cfg['norm_weight_decay'],
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = cfg['opt']
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {opt_name}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if cfg['amp'] else None
    
    lr_scheduler = cfg['lr_scheduler'].lower()
    if lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=cfg['lr_step_size'], gamma=cfg['lr_gamma'])
    elif lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'] - cfg['lr_warmup_epochs'], eta_min=cfg['lr_min']
        )
    elif lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg['lr_gamma'])
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )
        
    if cfg['lr_warmup_epochs'] > 0:
        lr_warmup_method = cfg['lr_warmup_method']
        if cfg['lr_warmup_method'] == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=cfg['lr_warmup_decay'], total_iters=cfg['lr_warmup_epochs']
            )
        elif cfg['lr_warmup_method'] == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=cfg['lr_warmup_decay'], total_iters=cfg['lr_warmup_epochs']
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[cfg['lr_warmup_epochs']]
        )
    else:
        lr_scheduler = main_lr_scheduler
    

    
    model_without_ddp = model
    if cfg['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg['gpu']],find_unused_parameters=True)
        model_without_ddp = model.module
        
    total_steps = len(train_loader)

    if total_steps < 5:
        print_freq = 1
    else:
        print_freq = max(1, total_steps // 5) 


    cfg['print_freq'] = print_freq
    
    if cfg['only_test_ood']:
        if cfg['model_path']:
            cfg['store_dir'] = os.path.dirname(cfg['model_path'])
        else:
            cfg['store_dir'] = os.path.join(store_root,store_model_name+'_'+data_name+'_'+method+'_spts'+str(cfg['spts_epoch'])+'_s'+str(auto_seed)) 


        if data_name == 'imagenet100':
            test_ood_100(model, device)
        elif data_name == 'imagenet':
            test_ood(model, device)
             
        return
    else:

        if cfg['model_path']:
            cfg['store_dir'] = os.path.dirname(cfg['model_path'])
        else:
            if method in ['sd','sd+augmix']:
                cfg['store_dir'] = os.path.join(store_root,store_model_name+'_'+data_name+'_'+method+'_spts'+str(cfg['spts_epoch'])+'_s'+str(auto_seed))
            else:
                cfg['store_dir'] = os.path.join(store_root,store_model_name+'_'+data_name+'_'+method+'_s'+str(auto_seed)) 
 


    if cfg['distributed']:
        if torch.distributed.get_rank()==0:
            if not os.path.exists(cfg['store_dir']):
                os.makedirs(cfg['store_dir'])
    else:
        if not os.path.exists(cfg['store_dir']):
            os.makedirs(cfg['store_dir'])
            
    model_ema = None
    if cfg['model_ema']:
        adjust = world_size * batch_size * cfg['model_ema_steps'] / cfg['epochs']
        alpha = 1.0 - cfg['model_ema_decay']
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
        
    if cfg['resume_path']:
        checkpoint = torch.load(cfg['resume_path'], map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
            
    log_dir = os.path.join(cfg['store_dir'],'logs/')
    
    if cfg['distributed']:
        if torch.distributed.get_rank()==0:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
            
        else:
            writer = None
    else:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        writer = SummaryWriter(log_dir=log_dir)


    snd_refiner = SNDRefiner(cfg, device)
            
    print("Start training")
    for epoch in range(start_epoch, num_epochs):
        if cfg['distributed']:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        train_one_epoch(train_loader,model,criterion, optimizer, device,
                        epoch,writer,model_ema,scaler)
        
        
        if method in ['sd','sd+augmix']:
            spts_limit = cfg.get('spts_epoch', 5)
            if 0 <= epoch < spts_limit:
                fitness, ch_scores, cur_w = snd_refiner.get_fitness_score(model, val_loader, epoch, spts_limit)
                if fitness > snd_refiner.best_fitness:
                    snd_refiner.best_fitness = fitness
                    snd_refiner.best_conv1_weight = cur_w
                    snd_refiner.best_channel_scores = ch_scores
                    if not cfg['distributed'] or dist.get_rank() == 0:
                        print(f"--> Epoch {epoch} Memory Bank Updated.")

                if epoch == spts_limit - 1:
                    snd_refiner.apply_spts(model) 


        lr_scheduler.step()
        
        acc1 = validate(val_loader,model, criterion, 
                        epoch,writer,device)
        

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if model_ema:
            validate(val_loader,model_ema, criterion, 
                     epoch,writer,device,log_suffix="EMA")
        
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }
        if model_ema:
            checkpoint["model_ema"] = model_ema.state_dict()
        if scaler:
            checkpoint["scaler"] = scaler.state_dict()

        if cfg['distributed']:
            if torch.distributed.get_rank() == 0:
                save_checkpoint(checkpoint,is_best)
                save_resume_ckpt(checkpoint)
                if epoch%cfg['ckpt_freq']==0:
                    save_mid_ckpt(checkpoint,epoch)
                    
        else:
            save_checkpoint(checkpoint,is_best)
            save_resume_ckpt(checkpoint)
            if epoch%cfg['ckpt_freq']==0:
                save_mid_ckpt(checkpoint,epoch)
        
    if cfg['distributed']:
        if torch.distributed.get_rank() == 0:
            writer.close()
    else:
        writer.close()
        

    best_model_path = os.path.join(cfg['store_dir'],'model_best.pth.tar')
    best_ckpt = torch.load(best_model_path, map_location="cpu")
    model_without_ddp.load_state_dict(best_ckpt["model"])
    if data_name == 'imagenet':
        test_ood(model,device)
    elif data_name == 'imagenet100':
        test_ood_100(model,device)
    print('Done!')
            
        
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def train_one_epoch(train_loader,model,criterion, optimizer,device,
    epoch,writer,model_ema=None,scaler=None):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))


    model.train()
    cons_loss_val = None
    end = time.time()
    header = f"Epoch: [{epoch}]"
    for i, (images, target) in enumerate(metric_logger.log_every(train_loader, cfg['print_freq'], header)):
        start_time = time.time()

        images = images.to(device)
        target = target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):

            output = model(images)
            loss = criterion(output, target)
            output_for_acc = output
            
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg['clip_grad_norm'] is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg['clip_grad_norm'] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
            optimizer.step()
            
        if model_ema and i % cfg['model_ema_steps'] == 0:
            model_ema.update_parameters(model)
            if epoch < cfg['lr_warmup_epochs']:
                model_ema.n_averaged.fill_(0)
            

        acc1, acc5 = accuracy(output_for_acc, target, topk=(1,5))
        batch_size = target.size(0)
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        
        if cfg['distributed']:
            if torch.distributed.get_rank() == 0:
                writer.add_scalar('train_loss',loss.item(),epoch*len(train_loader)+i)
                if cons_loss_val is not None:
                    writer.add_scalar('cons_loss', cons_loss_val, epoch*len(train_loader)+i)
        else:
            writer.add_scalar('train_loss',loss.item(),epoch*len(train_loader)+i)
            if cons_loss_val is not None:
                writer.add_scalar('cons_loss', cons_loss_val, epoch*len(train_loader)+i)
        
    metric_logger.synchronize_between_processes()
    
    if cfg['distributed']:
        if torch.distributed.get_rank() == 0:
            writer.add_scalar('train_acc1',metric_logger.acc1.global_avg,epoch+1)
    else:
        writer.add_scalar('train_acc1',metric_logger.acc1.global_avg,epoch+1)
        
        

def test_ood_100(model,device):
    

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    batch_size = int(cfg['batch_size']/cfg['nprocs'])
    cudnn.benchmark =True
    model.eval()

    test_ood_dir = os.path.join(cfg['store_dir'],'test_ood')
    
    if cfg['distributed']:
        if torch.distributed.get_rank()==0:
            if not os.path.exists(test_ood_dir):
                os.makedirs(test_ood_dir)
    else:
        if not os.path.exists(test_ood_dir):
            os.makedirs(test_ood_dir)
    

    
    t_df_name = 'ood_results.csv'
    store_path = os.path.join(test_ood_dir,t_df_name)
    if os.path.exists(store_path):
        total_df = pd.read_csv(store_path,index_col=0)
    else:

        total_index = ['Clean']
        total_columns = ['CLS acc']
        total_df = pd.DataFrame(data=0, columns=total_columns, index=total_index)

        
    index_names = total_df.index
    

    if not total_df.loc['Clean','CLS acc'] == 0:

        print('ImageNet has been tested.')
        normalize = transforms.Normalize(mean=cfg['mean'],
                                            std=cfg['std'])
        clean_transform = transforms.Compose([
                transforms.Resize(cfg['test_resize_size']),
                transforms.CenterCrop(cfg['test_crop_size']),
                transforms.ToTensor(),
                normalize,
            ])
        
        c_acc = total_df.loc['Clean','CLS acc']
    else:
    

        print('Evaluating on ImageNet...')
        clean_valdir = os.path.join(cfg['data_root'],'val')
        normalize = transforms.Normalize(mean=cfg['mean'],
                                            std=cfg['std'])
        clean_transform = transforms.Compose([
                transforms.Resize(cfg['test_resize_size']),
                transforms.CenterCrop(cfg['test_crop_size']),
                transforms.ToTensor(),
                normalize,
            ])

        val_dataset = datasets.ImageFolder(
            clean_valdir,clean_transform)
        
        if cfg['distributed']:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,shuffle=False)
        else:
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=8,
                                                    pin_memory=True,
                                                    sampler=val_sampler)
        
        if cfg['distributed']:
            val_sampler.set_epoch(0)
        batch_time = AverageMeter('Batch_Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(images)
                acc1, _ = accuracy(output, target, topk=(1,5))
                if cfg['distributed']:
                    torch.distributed.barrier()
                reduced_acc1 = reduce_mean(acc1, cfg['nprocs'])
                top1.update(reduced_acc1.item(), images.size(0))
                
                
                batch_time.update(time.time() - end)
                end = time.time()
                

        c_acc = top1.avg
        total_df.loc['Clean','CLS acc'] = round(c_acc,2)
        if cfg['distributed']:
            if torch.distributed.get_rank() == 0:
                total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
                print('{} on ImageNet: {}%.'.format(cfg['arch'],round(c_acc,2)))
        else:
            total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
            print('{} on ImageNet: {}%.'.format(cfg['arch'],round(c_acc,2))) 

    if not 'ImageNet-C' in index_names:
        
        total_df.loc['ImageNet-C','CLS acc'] = 0
        
    if not total_df.loc['ImageNet-C','CLS acc'] == 0:

        print('ImageNet-C has been tested.')
    
    else:
        corruption_lists = ['defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression','gaussian_noise', 'shot_noise', 'impulse_noise']
        

        
        imc_index = ['clean']+corruption_lists
        imc_columns = ['1','2','3','4','5']
        imc_df = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
        imc_df.loc['clean'] = round(c_acc,2)
        imc_df_name = 'imagenet-c_results.csv'
        if cfg['distributed']:
            if torch.distributed.get_rank() ==0:
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
        else:
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
        


        print('Evaluating on ImageNet-C...')
        imc_dir = os.path.join(cfg['data_root'],'corruptions')
        co_transform = clean_transform
        for co in corruption_lists:
            
            for level in range(1,6):
                

                print('{} level_{}...'.format(co,level))
                co_dir = os.path.join(imc_dir,co,str(level))
                co_dataset = datasets.ImageFolder(co_dir,co_transform)
                if cfg['distributed']:
                    co_sampler = torch.utils.data.distributed.DistributedSampler(co_dataset,shuffle=False)
                else:
                    co_sampler = torch.utils.data.SequentialSampler(co_dataset)
                co_loader = torch.utils.data.DataLoader(co_dataset,
                                                            batch_size=batch_size,
                                                            num_workers=8,
                                                            pin_memory=True,
                                                            sampler=co_sampler)
                
                co_sampler.set_epoch(0)
                top1 = AverageMeter('Acc@1', ':6.2f')
                with torch.no_grad():
                    end = time.time()
                    for i, (images, target) in enumerate(co_loader):
                        images = images.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        output = model(images)
                        pred = output.argmax(dim=1, keepdim=True)
                        acc1, _ = accuracy(output, target, topk=(1,5))
                        if cfg['distributed']:
                            torch.distributed.barrier()
                        reduced_acc1 = reduce_mean(acc1, cfg['nprocs'])
                        top1.update(reduced_acc1.item(), images.size(0))
                        
                co_acc = top1.avg
                imc_df.loc[co,str(level)] = round(co_acc,2)
                if cfg['distributed']:
                    if torch.distributed.get_rank() == 0:
                        imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                        print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
                else:
                    imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                    print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
        
        if cfg['distributed']:     
            if torch.distributed.get_rank() == 0:    
                imc_df['avg'] = imc_df.mean(axis=1).round(2)
                co_cols = list(imc_df.columns)
                co_cols = [co_cols[-1]] + co_cols[:-1]
                imc_df = imc_df[co_cols]
                
                rows_exclude = ['clean']
                cols_exclude = ['avg']
                mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
                total_df.loc['ImageNet-C','CLS acc'] = round(mean_coerror,2)
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
                print('{} on ImageNet-C: {}%.'.format(cfg['arch'],round(mean_coerror,2)))
        else:
            imc_df['avg'] = imc_df.mean(axis=1).round(2)
            co_cols = list(imc_df.columns)
            co_cols = [co_cols[-1]] + co_cols[:-1]
            imc_df = imc_df[co_cols]
            
            rows_exclude = ['clean']
            cols_exclude = ['avg']
            mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
            total_df.loc['ImageNet-C','CLS acc'] = round(mean_coerror,2)
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
            print('{} on ImageNet-C: {}%.'.format(cfg['arch'],round(mean_coerror,2)))
            
    if not 'ImageNet-C_bar' in index_names:
        
        total_df.loc['ImageNet-C_bar','CLS acc'] = 0
        
    if not total_df.loc['ImageNet-C_bar','CLS acc'] == 0:
        print('ImageNet-C_bar has been tested.')
    
    else:
        corruption_lists = ['blue_noise_sample','checkerboard_cutout','perlin_noise',
        'sparkles','brownish_noise','cocentric_sine_waves','plasma_noise','caustic_refraction',
        'inverse_sparkles','single_frequency_greyscale']
        

        
        
        imc_index = ['clean']+corruption_lists
        imc_columns = ['1','2','3','4','5']
        imc_df = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
        imc_df.loc['clean'] = round(c_acc,2)
        imc_df_name = 'imagenet-c_bar_results.csv'
        if cfg['distributed']:
            if torch.distributed.get_rank() ==0:
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
        else:
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            

        print('Evaluating on ImageNet-C_bar...')
        imc_dir = os.path.join(cfg['data_root'],'corruptions_cbar')
        co_transform = clean_transform
        for co in corruption_lists:
            
            for level in range(1,6):
                print('{} level_{}...'.format(co,level))
                co_dir = os.path.join(imc_dir,co,str(level))
                co_dataset = datasets.ImageFolder(co_dir,co_transform)
                if cfg['distributed']:
                    co_sampler = torch.utils.data.distributed.DistributedSampler(co_dataset,shuffle=False)
                else:
                    co_sampler = torch.utils.data.SequentialSampler(co_dataset)
                co_loader = torch.utils.data.DataLoader(co_dataset,
                                                            batch_size=batch_size,
                                                            num_workers=8,
                                                            pin_memory=True,
                                                            sampler=co_sampler)
                
                co_sampler.set_epoch(0)
                top1 = AverageMeter('Acc@1', ':6.2f')
                with torch.no_grad():
                    end = time.time()
                    for i, (images, target) in enumerate(co_loader):
                        images = images.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        output = model(images)
                        pred = output.argmax(dim=1, keepdim=True)
                        acc1, _ = accuracy(output, target, topk=(1,5))
                        if cfg['distributed']:
                            torch.distributed.barrier()
                        reduced_acc1 = reduce_mean(acc1, cfg['nprocs'])
                        top1.update(reduced_acc1.item(), images.size(0))
                        
                co_acc = top1.avg
                imc_df.loc[co,str(level)] = round(co_acc,2)
                if cfg['distributed']:
                    if torch.distributed.get_rank() == 0:
                        imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                        print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
                else:
                    imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                    print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
    
        if cfg['distributed']:
            if torch.distributed.get_rank() == 0:    
                imc_df['avg'] = imc_df.mean(axis=1).round(2)
                co_cols = list(imc_df.columns)
                co_cols = [co_cols[-1]] + co_cols[:-1]
                imc_df = imc_df[co_cols]
                
                rows_exclude = ['clean']
                cols_exclude = ['avg']
                mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
                total_df.loc['ImageNet-C_bar','CLS acc'] = round(mean_coerror,2)
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
                print('{} on ImageNet-C_bar: {}%.'.format(cfg['arch'],mean_coerror))
        else:
   
            imc_df['avg'] = imc_df.mean(axis=1).round(2)
            co_cols = list(imc_df.columns)
            co_cols = [co_cols[-1]] + co_cols[:-1]
            imc_df = imc_df[co_cols]
            
            rows_exclude = ['clean']
            cols_exclude = ['avg']
            mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
            total_df.loc['ImageNet-C_bar','CLS acc'] = round(mean_coerror,2)
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
            print('{} on ImageNet-C_bar: {}%.'.format(cfg['arch'],mean_coerror))
            
    if not 'ImageNet-3DCC' in index_names:
        
        total_df.loc['ImageNet-3DCC','CLS acc'] = 0
        
    if not total_df.loc['ImageNet-3DCC','CLS acc'] == 0:
        print('ImageNet-3DCC has been tested.')
    
    else:
        corruption_lists = ['bit_error','far_focus','fog_3d','h265_crf','low_light','xy_motion_blur',
                            'color_quant','flash','h265_abr','iso_noise','near_focus','z_motion_blur']
        

        
        
        imc_index = ['clean']+corruption_lists
        imc_columns = ['1','2','3','4','5']
        imc_df = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
        imc_df.loc['clean'] = round(c_acc,2)
        imc_df_name = 'imagenet-3dcc_results.csv'
        if cfg['distributed']:
            if torch.distributed.get_rank() ==0:
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
        else:
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            

        print('Evaluating on ImageNet-3DCC...')
        imc_dir = os.path.join(cfg['data_root'],'imagenet_3dcc')
        co_transform = clean_transform
        for co in corruption_lists:
            
            for level in range(1,6):
                print('{} level_{}...'.format(co,level))
                co_dir = os.path.join(imc_dir,co,str(level))
                co_dataset = datasets.ImageFolder(co_dir,co_transform)
                if cfg['distributed']:
                    co_sampler = torch.utils.data.distributed.DistributedSampler(co_dataset,shuffle=False)
                else:
                    co_sampler = torch.utils.data.SequentialSampler(co_dataset)
                co_loader = torch.utils.data.DataLoader(co_dataset,
                                                            batch_size=batch_size,
                                                            num_workers=8,
                                                            pin_memory=True,
                                                            sampler=co_sampler)
                
                co_sampler.set_epoch(0)
                top1 = AverageMeter('Acc@1', ':6.2f')
                with torch.no_grad():
                    end = time.time()
                    for i, (images, target) in enumerate(co_loader):
                        images = images.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        output = model(images)
                        pred = output.argmax(dim=1, keepdim=True)
                        acc1, _ = accuracy(output, target, topk=(1,5))
                        if cfg['distributed']:
                            torch.distributed.barrier()
                        reduced_acc1 = reduce_mean(acc1, cfg['nprocs'])
                        top1.update(reduced_acc1.item(), images.size(0))
                        
                co_acc = top1.avg
                imc_df.loc[co,str(level)] = round(co_acc,2)
                if cfg['distributed']:
                    if torch.distributed.get_rank() == 0:
                        imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                        print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
                else:
                    imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                    print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
    
        if cfg['distributed']:
            if torch.distributed.get_rank() == 0:    
                imc_df['avg'] = imc_df.mean(axis=1).round(2)
                co_cols = list(imc_df.columns)
                co_cols = [co_cols[-1]] + co_cols[:-1]
                imc_df = imc_df[co_cols]
                
                rows_exclude = ['clean']
                cols_exclude = ['avg']
                mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
                total_df.loc['ImageNet-3DCC','CLS acc'] = round(mean_coerror,2)
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
                print('{} on ImageNet-3DCC: {}%.'.format(cfg['arch'],mean_coerror))
        else:
   
            imc_df['avg'] = imc_df.mean(axis=1).round(2)
            co_cols = list(imc_df.columns)
            co_cols = [co_cols[-1]] + co_cols[:-1]
            imc_df = imc_df[co_cols]
            
            rows_exclude = ['clean']
            cols_exclude = ['avg']
            mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
            total_df.loc['ImageNet-3DCC','CLS acc'] = round(mean_coerror,2)
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
            print('{} on ImageNet-3DCC: {}%.'.format(cfg['arch'],mean_coerror))

def test_ood(model,device):
    

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    batch_size = int(cfg['batch_size']/cfg['nprocs'])

    cudnn.benchmark =True
    model.eval()

    test_ood_dir = os.path.join(cfg['store_dir'],'test_ood')
    
    if cfg['distributed']:
        if torch.distributed.get_rank()==0:
            if not os.path.exists(test_ood_dir):
                os.makedirs(test_ood_dir)
    else:
        if not os.path.exists(test_ood_dir):
            os.makedirs(test_ood_dir)
    
    t_df_name = 'ood_results.csv'
    store_path = os.path.join(test_ood_dir,t_df_name)
    if os.path.exists(store_path):
        total_df = pd.read_csv(store_path,index_col=0)
    else:

        total_index = ['Clean']
        total_columns = ['CLS acc']
        total_df = pd.DataFrame(data=0, columns=total_columns, index=total_index)
        
    index_names = total_df.index
    

    if not total_df.loc['Clean','CLS acc'] == 0:

        print('ImageNet has been tested.')
        normalize = transforms.Normalize(mean=cfg['mean'],
                                            std=cfg['std'])
        clean_transform = transforms.Compose([
                transforms.Resize(cfg['test_resize_size']),
                transforms.CenterCrop(cfg['test_crop_size']),
                transforms.ToTensor(),
                normalize,
            ])
        
        c_acc = total_df.loc['Clean','CLS acc']
    else:
    

        print('Evaluating on ImageNet...')
        clean_valdir = os.path.join(cfg['data_root'],'val')
        normalize = transforms.Normalize(mean=cfg['mean'],
                                            std=cfg['std'])
        clean_transform = transforms.Compose([
                transforms.Resize(cfg['test_resize_size']),
                transforms.CenterCrop(cfg['test_crop_size']),
                transforms.ToTensor(),
                normalize,
            ])
        
        val_dataset = datasets.ImageFolder(
            clean_valdir,clean_transform)
        
        if cfg['distributed']:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,shuffle=False)
        else:
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=8,
                                                    pin_memory=True,
                                                    sampler=val_sampler)
        
        if cfg['distributed']:
            val_sampler.set_epoch(0)
        batch_time = AverageMeter('Batch_Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(len(val_loader), [batch_time, top1],
                                prefix='ImageNet test: ')
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(images)
                acc1, _ = accuracy(output, target, topk=(1,5))
                if cfg['distributed']:
                    torch.distributed.barrier()
                reduced_acc1 = reduce_mean(acc1, cfg['nprocs'])
                top1.update(reduced_acc1.item(), images.size(0))

                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % cfg['print_freq'] == 0:
                    if cfg['distributed']:
                        if torch.distributed.get_rank() == 0:
                            progress.display(i)
                    else:
                        progress.display(i)
        c_acc = top1.avg
        total_df.loc['Clean','CLS acc'] = round(c_acc,2)
        if cfg['distributed']:
            if torch.distributed.get_rank() == 0:
                total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
                print('{} on ImageNet: {}%.'.format(cfg['arch'],round(c_acc,2)))
        else:
            total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
            print('{} on ImageNet: {}%.'.format(cfg['arch'],round(c_acc,2))) 


    if not 'ImageNet-C' in index_names:
        
        total_df.loc['ImageNet-C','CLS acc'] = 0
        
    if not total_df.loc['ImageNet-C','CLS acc'] == 0:

        print('ImageNet-C has been tested.')
    
    else:
        corruption_lists = ['defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression','gaussian_noise', 'shot_noise', 'impulse_noise']

        
        
        imc_index = ['clean']+corruption_lists
        imc_columns = ['1','2','3','4','5']
        imc_df = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
        imc_df.loc['clean'] = round(c_acc,2)
        imc_df_name = 'imagenet-c_results.csv'
        if cfg['distributed']:
            if torch.distributed.get_rank() ==0:
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
        else:
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
        


        print('Evaluating on ImageNet-C...')
        imc_dir = os.path.join(cfg['data_root'],'corruptions')
        co_transform = clean_transform
        for co in corruption_lists:
            
            for level in range(1,6):
                

                print('{} level_{}...'.format(co,level))
                co_dir = os.path.join(imc_dir,co,str(level))
                co_dataset = datasets.ImageFolder(co_dir,co_transform)
                if cfg['distributed']:
                    co_sampler = torch.utils.data.distributed.DistributedSampler(co_dataset,shuffle=False)
                else:
                    co_sampler = torch.utils.data.SequentialSampler(co_dataset)
                co_loader = torch.utils.data.DataLoader(co_dataset,
                                                            batch_size=batch_size,
                                                            num_workers=8,
                                                            pin_memory=True,
                                                            sampler=co_sampler)
                
                co_sampler.set_epoch(0)
                top1 = AverageMeter('Acc@1', ':6.2f')
                with torch.no_grad():
                    end = time.time()
                    for i, (images, target) in enumerate(co_loader):
                        images = images.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        output = model(images)
                        pred = output.argmax(dim=1, keepdim=True)
                        acc1, _ = accuracy(output, target, topk=(1,5))
                        if cfg['distributed']:
                            torch.distributed.barrier()
                        reduced_acc1 = reduce_mean(acc1, cfg['nprocs'])
                        top1.update(reduced_acc1.item(), images.size(0))
                        
                co_acc = top1.avg
                imc_df.loc[co,str(level)] = round(co_acc,2)
                if cfg['distributed']:
                    if torch.distributed.get_rank() == 0:
                        imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                        print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
                else:
                    imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                    print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
        
        if cfg['distributed']:     
            if torch.distributed.get_rank() == 0:    
                imc_df['avg'] = imc_df.mean(axis=1).round(2)
                co_cols = list(imc_df.columns)
                co_cols = [co_cols[-1]] + co_cols[:-1]
                imc_df = imc_df[co_cols]
                
                rows_exclude = ['clean']
                cols_exclude = ['avg']
                mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
                total_df.loc['ImageNet-C','CLS acc'] = round(mean_coerror,2)
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
                print('{} on ImageNet-C: {}%.'.format(cfg['arch'],round(mean_coerror,2)))
        else:
            imc_df['avg'] = imc_df.mean(axis=1).round(2)
            co_cols = list(imc_df.columns)
            co_cols = [co_cols[-1]] + co_cols[:-1]
            imc_df = imc_df[co_cols]
            
            rows_exclude = ['clean']
            cols_exclude = ['avg']
            mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
            total_df.loc['ImageNet-C','CLS acc'] = round(mean_coerror,2)
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
            print('{} on ImageNet-C: {}%.'.format(cfg['arch'],round(mean_coerror,2)))

    if not 'ImageNet-C_bar' in index_names:
        
        total_df.loc['ImageNet-C_bar','CLS acc'] = 0
        
    if not total_df.loc['ImageNet-C_bar','CLS acc'] == 0:
        print('ImageNet-C_bar has been tested.')
    
    else:
        corruption_lists = ['blue_noise_sample','checkerboard_cutout','perlin_noise',
        'sparkles','brownish_noise','cocentric_sine_waves','plasma_noise','caustic_refraction',
        'inverse_sparkles','single_frequency_greyscale']
        
        
        imc_index = ['clean']+corruption_lists
        imc_columns = ['1','2','3','4','5']
        imc_df = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
        imc_df.loc['clean'] = round(c_acc,2)
        imc_df_name = 'imagenet-c_bar_results.csv'
        if cfg['distributed']:
            if torch.distributed.get_rank() ==0:
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
        else:
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            

        print('Evaluating on ImageNet-C_bar...')
        imc_dir = os.path.join(cfg['data_root'],'corruptions_cbar')
        co_transform = clean_transform
        for co in corruption_lists:
            
            for level in range(1,6):
                print('{} level_{}...'.format(co,level))
                co_dir = os.path.join(imc_dir,co,str(level))
                co_dataset = datasets.ImageFolder(co_dir,co_transform)
                if cfg['distributed']:
                    co_sampler = torch.utils.data.distributed.DistributedSampler(co_dataset,shuffle=False)
                else:
                    co_sampler = torch.utils.data.SequentialSampler(co_dataset)
                co_loader = torch.utils.data.DataLoader(co_dataset,
                                                            batch_size=batch_size,
                                                            num_workers=8,
                                                            pin_memory=True,
                                                            sampler=co_sampler)
                
                co_sampler.set_epoch(0)
                top1 = AverageMeter('Acc@1', ':6.2f')
                with torch.no_grad():
                    end = time.time()
                    for i, (images, target) in enumerate(co_loader):
                        images = images.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        output = model(images)
                        pred = output.argmax(dim=1, keepdim=True)
                        acc1, _ = accuracy(output, target, topk=(1,5))
                        if cfg['distributed']:
                            torch.distributed.barrier()
                        reduced_acc1 = reduce_mean(acc1, cfg['nprocs'])
                        top1.update(reduced_acc1.item(), images.size(0))
                        
                co_acc = top1.avg
                imc_df.loc[co,str(level)] = round(co_acc,2)
                if cfg['distributed']:
                    if torch.distributed.get_rank() == 0:
                        imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                        print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
                else:
                    imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                    print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
    
        if cfg['distributed']:
            if torch.distributed.get_rank() == 0:
                imc_df['avg'] = imc_df.mean(axis=1).round(2)
                co_cols = list(imc_df.columns)
                co_cols = [co_cols[-1]] + co_cols[:-1]
                imc_df = imc_df[co_cols]
                
                rows_exclude = ['clean']
                cols_exclude = ['avg']
                mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
                total_df.loc['ImageNet-C_bar','CLS acc'] = round(mean_coerror,2)
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
                print('{} on ImageNet-C_bar: {}%.'.format(cfg['arch'],mean_coerror))
        else:

            imc_df['avg'] = imc_df.mean(axis=1).round(2)
            co_cols = list(imc_df.columns)
            co_cols = [co_cols[-1]] + co_cols[:-1]
            imc_df = imc_df[co_cols]
            
            rows_exclude = ['clean']
            cols_exclude = ['avg']
            mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
            total_df.loc['ImageNet-C_bar','CLS acc'] = round(mean_coerror,2)
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
            print('{} on ImageNet-C_bar: {}%.'.format(cfg['arch'],mean_coerror))


    if not 'ImageNet-3DCC' in index_names:
        
        total_df.loc['ImageNet-3DCC','CLS acc'] = 0
        
    if not total_df.loc['ImageNet-3DCC','CLS acc'] == 0:
        print('ImageNet-3DCC has been tested.')
    
    else:
        corruption_lists = ['bit_error','far_focus','fog_3d','h265_crf','low_light','xy_motion_blur',
                            'color_quant','flash','h265_abr','iso_noise','near_focus','z_motion_blur']
        
        
        imc_index = ['clean']+corruption_lists
        imc_columns = ['1','2','3','4','5']
        imc_df = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
        imc_df.loc['clean'] = round(c_acc,2)
        imc_df_name = 'imagenet-3dcc_results.csv'
        if cfg['distributed']:
            if torch.distributed.get_rank() ==0:
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
        else:
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            

        print('Evaluating on ImageNet-3DCC...')
        imc_dir = os.path.join(cfg['data_root'],'imagenet_3dcc')
        co_transform = clean_transform
        for co in corruption_lists:
            
            for level in range(1,6):
                print('{} level_{}...'.format(co,level))
                co_dir = os.path.join(imc_dir,co,str(level))
                co_dataset = datasets.ImageFolder(co_dir,co_transform)
                if cfg['distributed']:
                    co_sampler = torch.utils.data.distributed.DistributedSampler(co_dataset,shuffle=False)
                else:
                    co_sampler = torch.utils.data.SequentialSampler(co_dataset)
                co_loader = torch.utils.data.DataLoader(co_dataset,
                                                            batch_size=batch_size,
                                                            num_workers=8,
                                                            pin_memory=True,
                                                            sampler=co_sampler)
                
                if cfg['distributed']:
                    co_sampler.set_epoch(0)
                top1 = AverageMeter('Acc@1', ':6.2f')
                with torch.no_grad():
                    end = time.time()
                    for i, (images, target) in enumerate(co_loader):
                        images = images.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        output = model(images)
                        pred = output.argmax(dim=1, keepdim=True)
                        acc1, _ = accuracy(output, target, topk=(1,5))
                        if cfg['distributed']:
                            torch.distributed.barrier()
                        reduced_acc1 = reduce_mean(acc1, cfg['nprocs'])
                        top1.update(reduced_acc1.item(), images.size(0))
                        
                co_acc = top1.avg
                imc_df.loc[co,str(level)] = round(co_acc,2)
                if cfg['distributed']:
                    if torch.distributed.get_rank() == 0:
                        imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                        print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
                else:
                    imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                    print('{} on {}_{}: {}%.'.format(cfg['arch'],co,level,round(co_acc,2)))
    
        if cfg['distributed']:
            if torch.distributed.get_rank() == 0:    
                imc_df['avg'] = imc_df.mean(axis=1).round(2)
                co_cols = list(imc_df.columns)
                co_cols = [co_cols[-1]] + co_cols[:-1]
                imc_df = imc_df[co_cols]
                
                rows_exclude = ['clean']
                cols_exclude = ['avg']
                mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
                total_df.loc['ImageNet-3DCC','CLS acc'] = round(mean_coerror,2)
                imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
                total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
                print('{} on ImageNet-3DCC: {}%.'.format(cfg['arch'],mean_coerror))
        else:

            imc_df['avg'] = imc_df.mean(axis=1).round(2)
            co_cols = list(imc_df.columns)
            co_cols = [co_cols[-1]] + co_cols[:-1]
            imc_df = imc_df[co_cols]
            
            rows_exclude = ['clean']
            cols_exclude = ['avg']
            mean_coerror = imc_df.drop(rows_exclude).drop(columns=cols_exclude).values.mean()    
            total_df.loc['ImageNet-3DCC','CLS acc'] = round(mean_coerror,2)
            imc_df.to_csv(os.path.join(test_ood_dir,imc_df_name))
            total_df.to_csv(os.path.join(test_ood_dir,t_df_name))
            print('{} on ImageNet-3DCC: {}%.'.format(cfg['arch'],mean_coerror)) 

               
def validate(val_loader, model, criterion,
             epoch,writer,device,log_suffix=''):
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    model.eval()

    num_processed_samples = 0
    with torch.inference_mode():
        end = time.time()
        for i, (images, target) in enumerate(metric_logger.log_every(val_loader, cfg['print_freq'], header)):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)


            output = model(images)

            loss = criterion(output, target)


            acc1, acc5 = accuracy(output, target, topk=(1,5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
            
    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(val_loader.dataset, "__len__")
        and len(val_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the dataset has {len(val_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    if cfg['distributed']:
        if torch.distributed.get_rank() == 0:
            writer.add_scalar('val_acc1',metric_logger.acc1.global_avg,epoch+1)
    else:
        writer.add_scalar('val_acc1',metric_logger.acc1.global_avg,epoch+1)
        
    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg

def save_checkpoint(state,is_best, filename='model_best.pth.tar'):
    file_path = os.path.join(cfg['store_dir'],filename)
    if is_best:
        if os.path.exists(file_path):
            os.remove(file_path)
        torch.save(state, file_path)
        
def save_resume_ckpt(state,filename='ckpt.pth.tar'):
    file_path = os.path.join(cfg['store_dir'],filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    torch.save(state, file_path)
    
def save_mid_ckpt(state,epoch):
    file_path = os.path.join(cfg['store_dir'],'epoch_{}.ckpt.pth.tar'.format(epoch))
    torch.save(state, file_path)

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")
        
class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )
        
def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = cfg['lr'] * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:

            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)


            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)
    
class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        
    def __call__(self, img):
        return self.transforms(img)
    
def init_distributed_mode(cfg):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg['rank'] = int(os.environ["RANK"])
        cfg['world_size'] = int(os.environ["WORLD_SIZE"])
        cfg['gpu'] = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        cfg['rank'] = int(os.environ["SLURM_PROCID"])
        cfg['gpu'] = cfg['rank'] % torch.cuda.device_count()
    elif 'rank' in cfg:
        pass
    else:
        print("Not using distributed mode")
        cfg['distributed'] = False
        return

    cfg['distributed'] = True

    torch.cuda.set_device(cfg['gpu'])
    cfg['dist_backend'] = "nccl"
    print("| distributed init (rank {}): {}".format(cfg['rank'],cfg['dist_url']), flush=True)
    torch.distributed.init_process_group(
        backend=cfg['dist_backend'], init_method=cfg['dist_url'], world_size=cfg['world_size'], rank=cfg['rank']
    )
    torch.distributed.barrier()
    setup_for_distributed(cfg['rank'] == 0)
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

   
    
if __name__ == "__main__":
    
    main()