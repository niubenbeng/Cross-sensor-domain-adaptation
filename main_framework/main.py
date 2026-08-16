# main.py — JVM (Joint MMSD-VDR Squared) cross-sensor transfer training
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import argparse
from torch.utils import data as da
from torchmetrics import MeanMetric, Accuracy, Precision, Recall, F1Score
import time
import logging
import json
import os

from losses import MMSD, JVM
from swin_transformer_1d import SwinTransformerLayer


def setup_logging(save_path, run_name='train'):
    """Configure logger that writes to both console and log file."""
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, run_name + '.log')

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.info('Logging to %s', log_file)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='JVM cross-sensor transfer training')

    parser.add_argument('--src_data', type=str, default="channel_0.npy", help='')
    parser.add_argument('--src_label', type=str, default="label_0.npy", help='')
    parser.add_argument('--dst_data', type=str, default="channel_1.npy", help='')
    parser.add_argument('--dst_label', type=str, default="label_1.npy", help='')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--nepoch', type=int, default=200, help='max number of epochs')
    parser.add_argument('--num_classes', type=int, default=5, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--u', type=float, default=10.0, help='domain loss weight during warmup')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--save_path', type=str, default='./', help='')
    parser.add_argument('--src_name', type=str, default='source', help='source domain label (for logging)')
    parser.add_argument('--dst_name', type=str, default='target', help='target domain label (for logging)')
    # Model architecture (Swin-1D)
    parser.add_argument('--dim', type=int, default=128, help='embedding dim')
    parser.add_argument('--depth', type=int, default=4, help='transformer blocks')
    parser.add_argument('--num_heads', type=int, default=4, help='attention heads')
    parser.add_argument('--window_size', type=int, default=12, help='attention window size')
    parser.add_argument('--lambd', type=float, default=1.0, help='domain loss weight (combined with --u)')
    parser.add_argument('--lr_min', type=float, default=1e-4, help='cosine schedule min lr')
    parser.add_argument('--lr_schedule', type=str, default='none', choices=['none', 'cosine'],
                        help='lr scheduler: "none" or "cosine"')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clip max norm (0=disabled)')
    parser.add_argument('--accum_steps', type=int, default=1, help='gradient accumulation steps (1=no accumulation)')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='BatchNorm momentum')
    parser.add_argument('--jmmsd_warmup', type=int, default=20,
                        help='joint alignment warmup epochs: first N epochs use plain MMSD, then switch to JVM')
    parser.add_argument('--jmmsd_u', type=float, default=1.0,
                        help='domain loss weight after JVM activation; warmup uses --u')
    parser.add_argument('--w_c', type=float, default=1.0,
                        help='classification loss weight: L_total = w_c*L_c + w_d*lambd*u_eff*L_d')
    parser.add_argument('--w_d', type=float, default=1.0,
                        help='domain loss weight: L_total = w_c*L_c + w_d*lambd*u_eff*L_d')
    parser.add_argument('--w_v', type=float, default=1.0,
                        help='JVDR weight in JVM: L_d = w_v*L_VDR + w_m*L_MMSD')
    parser.add_argument('--w_m', type=float, default=1.0,
                        help='JMMSD weight in JVM: L_d = w_v*L_VDR + w_m*L_MMSD')
    args = parser.parse_args()
    return args


class SignalDataset(da.Dataset):
    """1D signal dataset."""
    def __init__(self, X, y):
        self.Data = X
        self.Label = y

    def __getitem__(self, index):
        return self.Data[index], self.Label[index]

    def __len__(self):
        return len(self.Data)


def load_data():
    """Load and normalize source/target domain data, return Dataset."""
    source_data = np.load(args.src_data)
    source_label = np.load(args.src_label).argmax(axis=-1)
    target_data = np.load(args.dst_data)
    target_label = np.load(args.dst_label).argmax(axis=-1)
    # per-sample min-max normalization
    source_data = (source_data - source_data.min(axis=1).reshape((-1, 1))) / (
                source_data.max(axis=1).reshape((-1, 1)) - source_data.min(axis=1).reshape((-1, 1)))
    target_data = (target_data - target_data.min(axis=1).reshape((-1, 1))) / (
                target_data.max(axis=1).reshape((-1, 1)) - target_data.min(axis=1).reshape((-1, 1)))
    source_data, source_label = shuffle(source_data, source_label, random_state=0)
    target_data, target_label = shuffle(target_data, target_label, random_state=0)
    source_data = np.expand_dims(source_data, axis=1)
    target_data = np.expand_dims(target_data, axis=1)
    return SignalDataset(source_data, source_label), SignalDataset(target_data, target_label)


def train(model, source_loader, target_loader, optimizer, cls_criterion, domain_criterion,
          epoch, warmup_criterion):
    """Single epoch training. Returns (train_acc, train_loss, cls_loss, domain_loss)."""
    lambd = args.lambd
    w_c = args.w_c
    w_d = args.w_d
    accum_steps = max(1, args.accum_steps)
    # JVM warmup: first N epochs use plain MMSD on 128D features, then switch to JVM joint alignment
    jmvs_active = (epoch >= args.jmmsd_warmup)
    u_eff = args.jmmsd_u if jmvs_active else args.u
    model.train()
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)
    optimizer.zero_grad()
    for i in range(num_iter):
        source_data, source_label = next(iter_source)
        target_data, _ = next(iter_target)
        source_data, source_label = source_data.cuda(), source_label.cuda()
        target_data = target_data.cuda()
        # align 128D pre-classifier features (DAN-style)
        output1, feat1 = model(source_data.float(), return_feat=True)
        output2, feat2 = model(target_data.float(), return_feat=True)
        cls_loss = cls_criterion(output1, source_label)
        if jmvs_active:
            # JVM two-layer joint alignment (feat 128D x softmax 5D)
            domain_loss = domain_criterion(feat1, feat2, output1, output2)
        else:
            # warmup: plain MMSD marginal alignment
            domain_loss = warmup_criterion(feat1, feat2)
        loss = w_c * cls_loss + w_d * lambd * u_eff * domain_loss
        metric_train_acc.update(output1.max(1)[1], source_label)
        metric_train_loss.update(loss)
        metric_cls_loss.update(cls_loss)
        metric_domain_loss.update(domain_loss)
        # gradient accumulation
        (loss / accum_steps).backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == num_iter:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
    train_acc = metric_train_acc.compute()
    train_loss = metric_train_loss.compute()
    cls_loss = metric_cls_loss.compute()
    domain_loss = metric_domain_loss.compute()
    metric_train_acc.reset()
    metric_train_loss.reset()
    metric_cls_loss.reset()
    metric_domain_loss.reset()
    return train_acc, train_loss, cls_loss, domain_loss


def test(model, target_loader, cls_criterion):
    """Evaluate on target domain. Returns (acc, loss, precision, recall, f1)."""
    model.eval()
    iter_target = iter(target_loader)
    num_iter = len(target_loader)
    with torch.no_grad():
        for i in range(num_iter):
            target_data, target_label = next(iter_target)
            target_data, target_label = target_data.cuda(), target_label.cuda()
            output = model(target_data.float())
            preds = output.max(1)[1]
            metric_test_acc.update(preds, target_label)
            metric_test_precision.update(preds, target_label)
            metric_test_recall.update(preds, target_label)
            metric_test_f1.update(preds, target_label)
            metric_test_loss.update(cls_criterion(output, target_label))
        test_acc = metric_test_acc.compute()
        test_precision = metric_test_precision.compute()
        test_recall = metric_test_recall.compute()
        test_f1 = metric_test_f1.compute()
        test_loss = metric_test_loss.compute()
        metric_test_acc.reset()
        metric_test_precision.reset()
        metric_test_recall.reset()
        metric_test_f1.reset()
        metric_test_loss.reset()
    return test_acc, test_loss, test_precision, test_recall, test_f1


if __name__ == '__main__':
    args = parse_args()
    logger = setup_logging(args.save_path, run_name='train')
    logger.info('Transfer task: %s -> %s | kernel=JVM', args.src_name, args.dst_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        _props = torch.cuda.get_device_properties(0)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total_mem_gb = _props.total_memory / 1024 ** 3
        gpu_capability = '{}.{}'.format(_props.major, _props.minor)
        cuda_version = torch.version.cuda
        logger.info('GPU: %s | %.1f GiB | compute capability %s | CUDA %s | torch %s',
                    gpu_name, gpu_total_mem_gb, gpu_capability, cuda_version, torch.__version__)
    else:
        gpu_name, gpu_total_mem_gb, gpu_capability, cuda_version = 'CPU', 0.0, 'n/a', 'n/a'
        logger.info('No CUDA GPU available - running on CPU (torch %s)', torch.__version__)

    # metrics
    metric_train_acc = Accuracy(task='multiclass', num_classes=args.num_classes).cuda()
    metric_test_acc = Accuracy(task='multiclass', num_classes=args.num_classes).cuda()
    metric_test_precision = Precision(task='multiclass', num_classes=args.num_classes).cuda()
    metric_test_recall = Recall(task='multiclass', num_classes=args.num_classes).cuda()
    metric_test_f1 = F1Score(task='multiclass', num_classes=args.num_classes).cuda()
    metric_train_loss = MeanMetric().cuda()
    metric_test_loss = MeanMetric().cuda()
    metric_cls_loss = MeanMetric().cuda()
    metric_domain_loss = MeanMetric().cuda()

    # data
    source_dataset, target_dataset = load_data()
    g = torch.Generator()
    source_loader = da.DataLoader(dataset=source_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    g = torch.Generator()
    target_loader = da.DataLoader(dataset=target_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    target_loader_test = da.DataLoader(dataset=target_dataset, batch_size=16, shuffle=False)

    # model
    model = SwinTransformerLayer(dim=args.dim, depth=args.depth, num_heads=args.num_heads,
                                 window_size=args.window_size, n_classes=args.num_classes).to(device)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = args.bn_momentum

    # loss functions: LabelSmoothing for classification, JVM for domain alignment, MMSD for warmup
    cls_criterion = LabelSmoothingCrossEntropy()
    domain_criterion = JVM(w_v=args.w_v, w_m=args.w_m)
    warmup_criterion = MMSD()

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch, eta_min=args.lr_min)
    else:
        scheduler = None

    # hyperparameters record
    n_params = sum(p.numel() for p in model.parameters())
    hparams = {
        'src_name': args.src_name, 'dst_name': args.dst_name,
        'src_data': args.src_data, 'src_label': args.src_label,
        'dst_data': args.dst_data, 'dst_label': args.dst_label,
        'num_classes': args.num_classes,
        'n_source_samples': len(source_dataset), 'n_target_samples': len(target_dataset),
        'batch_size': args.batch_size, 'nepoch': args.nepoch,
        'lr': args.lr, 'lr_min': args.lr_min, 'lr_schedule': args.lr_schedule,
        'weight_decay': args.weight_decay,
        'optimizer': type(optimizer).__name__,
        'grad_clip': args.grad_clip, 'accum_steps': args.accum_steps,
        'bn_momentum': args.bn_momentum,
        'steps_per_epoch': len(source_loader),
        'kernel': 'JVM', 'lambd': args.lambd, 'u': args.u,
        'w_c': args.w_c, 'w_d': args.w_d,
        'domain_loss_weight': args.w_d * args.lambd * args.u,
        'align': 'feat',
        'jmmsd_warmup': args.jmmsd_warmup,
        'jmmsd_u': args.jmmsd_u,
        'w_v': args.w_v,
        'w_m': args.w_m,
        'cls_criterion': type(cls_criterion).__name__,
        'domain_criterion': type(domain_criterion).__name__,
        'model': type(model).__name__,
        'dim': args.dim, 'depth': args.depth, 'num_heads': args.num_heads,
        'window_size': args.window_size, 'n_params': n_params,
        'device': str(device),
        'gpu_name': gpu_name,
        'gpu_total_mem_gb': round(gpu_total_mem_gb, 2),
        'gpu_capability': gpu_capability,
        'cuda_version': cuda_version,
        'torch_version': torch.__version__,
        'save_path': args.save_path,
    }
    logger.info('-' * 70)
    logger.info('Hyper-parameters for this run:')
    for k, v in hparams.items():
        logger.info('    %-20s = %s', k, v)
    logger.info('-' * 70)
    with open(os.path.join(args.save_path, 'hparams.json'), 'w', encoding='utf-8') as f:
        json.dump(hparams, f, indent=2)

    # training loop
    best_test_acc = 0.0
    best_epoch = 0
    total_start = time.time()

    logger.info('Start training for %d epochs (%s -> %s)', args.nepoch, args.src_name, args.dst_name)
    logger.info('JVM: L_total = w_c*L_c + w_d*lambd*u_eff*L_d, L_d = w_v*L_VDR + w_m*L_MMSD. '
                'w_c=%.3f, w_d=%.3f, w_v=%.3f, w_m=%.3f. warmup epochs 1-%d use plain MMSD (u=%.1f), epochs %d+ use JVM (u=%.1f).',
                args.w_c, args.w_d, args.w_v, args.w_m, args.jmmsd_warmup, args.u, args.jmmsd_warmup + 1, args.jmmsd_u)
    for epoch in range(args.nepoch):
        start_time = time.time()
        if epoch == args.jmmsd_warmup:
            logger.info('JVM activated at epoch %d - switching to JVM (u=%.1f)', epoch + 1, args.jmmsd_u)
        train_acc, train_loss, cls_loss, domain_loss = train(
            model, source_loader, target_loader, optimizer,
            cls_criterion, domain_criterion, epoch, warmup_criterion)
        test_acc, test_loss, test_precision, test_recall, test_f1 = test(
            model, target_loader_test, cls_criterion)
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - start_time

        if test_acc.item() > best_test_acc:
            best_test_acc = test_acc.item()
            best_epoch = epoch + 1

        logger.info(
            'Epoch %3d/%d | time %.2fs | train_loss %.5f (cls %.5f, domain %.5f) | '
            'train_acc %.5f | test_loss %.5f | test_acc %.5f | P %.5f | R %.5f | F1 %.5f | '
            'best_test_acc %.5f @ep%d',
            epoch + 1, args.nepoch, elapsed, train_loss.item(), cls_loss.item(),
            domain_loss.item(), train_acc.item(), test_loss.item(), test_acc.item(),
            test_precision.item(), test_recall.item(), test_f1.item(), best_test_acc, best_epoch
        )

    total_elapsed = time.time() - total_start
    final_test_acc = test_acc.item()

    # final summary
    logger.info('Final evaluation uses last-epoch model (epoch %d, best_test_acc=%.5f @ep%d)',
                args.nepoch, best_test_acc, best_epoch)
    summary = {
        'hparams': hparams,
        'src_name': args.src_name,
        'dst_name': args.dst_name,
        'kernel': 'JVM',
        'nepoch': args.nepoch,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'u': args.u,
        'w_c': args.w_c,
        'w_d': args.w_d,
        'w_v': args.w_v,
        'w_m': args.w_m,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'final_test_acc': final_test_acc,
        'final_test_precision': test_precision.item(),
        'final_test_recall': test_recall.item(),
        'final_test_f1': test_f1.item(),
        'final_train_acc': train_acc.item(),
        'total_train_sec': total_elapsed,
    }
    with open(os.path.join(args.save_path, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info('=' * 70)
    logger.info('DONE %s -> %s | best_test_acc %.5f @ep%d | final_test_acc %.5f (last-epoch) | '
                'final P/R/F1 %.5f/%.5f/%.5f | total %.1fs',
                args.src_name, args.dst_name, best_test_acc, best_epoch, final_test_acc,
                test_precision.item(), test_recall.item(), test_f1.item(), total_elapsed)
    logger.info('=' * 70)
