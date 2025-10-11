import os
import torch
import argparse
import yaml
import copy
import numpy as np
import time
import json
import csv
from datetime import datetime
from itertools import islice

from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.nn.functional as F
from sklearn import metrics

import utils
import emb_distance as EDIS
from net import AEDC, Discriminator,DiffusionRefiner  
from datasets import seg_set, clip_set
from losses import mmd_rbf, pyramid_mmd, make_beta_schedule, DiffusionHelper, sample_timesteps, q_sample, ddim_refine
from tqdm import tqdm
from utils import save_checkpoint, try_load_checkpoint, append_metrics, dump_run_state, build_train_loader


with open('config.yaml') as fp:
    param = yaml.safe_load(fp)

BS_LADDER = param.get('train', {}).get('bs_ladder', [32, 40, 48])


class D2GLoss(torch.nn.Module):
    '''
        Feature matching loss described in the paper.
    '''
    def __init__(self, cfg):
        super(D2GLoss, self).__init__()
        self.cfg = cfg

    def forward(self, feat_fake, feat_real):
        loss = 0
        norm_loss = {'l2': lambda x, y: F.mse_loss(x, y), 'l1': lambda x, y: F.l1_loss(x, y)}
        stat = {'mu': lambda x: x.mean(dim=0),
                'sigma': lambda x: (x - x.mean(dim=0, keepdim=True)).pow(2).mean(dim=0).sqrt()}

        if 'mu' in self.cfg.keys():
            mu_eff = self.cfg['mu']
            mu_fake, mu_real = stat['mu'](feat_fake), stat['mu'](feat_real)
            norm = norm_loss['l2'](mu_fake, mu_real)
            loss += mu_eff * norm
        if 'sigma' in self.cfg.keys():
            sigma_eff = self.cfg['sigma']
            sigma_fake, sigma_real = stat['sigma'](feat_fake), stat['sigma'](feat_real)
            norm = norm_loss['l2'](sigma_fake, sigma_real)
            loss += sigma_eff * norm
        return loss


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1, 1, 1), dtype=torch.float32, device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[0].view(-1, 1)
    fake = torch.ones((real_samples.shape[0], 1), dtype=torch.float32, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # gradients.shape为B*1*128*128
    gradients = gradients.view(gradients.size(0), -1)  # reshape为B*(128**2)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_one_epoch(netD, netG, tr_ld, optimD, optimG, device, d2g_eff, netU, optimU, helper, param,
                    epoch_idx=0, start_step=0, global_step=0,
                    metrics_csv_path=None, save_every_steps=5000,
                    checkpoint_last_path=None, best_aver=None,
                    scaler=None, use_amp=False, current_bs=None, logger=None):
    netD.train(); netG.train(); netU.train()
    aver_loss, gloss_num = {'recon': 0, 'd2g': 0, 'gloss': 0}, 0
    lambda_gp = param['train']['wgan']['lambda_gp']
    MSE = torch.nn.MSELoss()

    # CSV 字段
    csv_fields = ["time", "epoch", "step_in_epoch", "global_step", "bs",
                  "loss_recon", "loss_mmd", "loss_g_total"]

    # 准备进度条
    total_steps = len(tr_ld)
    if start_step > 0:
        pbar = tqdm(
            enumerate(islice(tr_ld, start_step, None), start=start_step),
            total=total_steps - start_step,            # ✅ 调整 total
            desc=f"Train Epoch {epoch_idx} (resume)"
        )
    else:
        pbar = tqdm(enumerate(tr_ld), total=total_steps, desc=f"Train Epoch {epoch_idx}")


    # 跳过 start_step 之前的 batch（续跑）
    if start_step > 0:
        pbar = tqdm(enumerate(islice(tr_ld, start_step, None), start=start_step),
                    total=total_steps, desc=f"Train Epoch {epoch_idx} (resume)")

    for i, (mel, _, _) in pbar:
        if i % 50 == 0:
            pbar.set_postfix({
                "step": f"{i}/{total_steps}",
                "GPU_mem": f"{torch.cuda.memory_allocated()//1024//1024}MB",
                "bs": current_bs
            })

        mel = mel.to(device)

        # ====== 1) 判别器 D ======
        # 说明：这里未加 AMP；如需 AMP，可把前向/反向包进 autocast 并用 scaler
        recon = netG(mel).detach()
        pred_real, _ = netD(mel)
        pred_fake, _ = netD(recon)
        gp = compute_gradient_penalty(netD, mel.data, recon.data, device)
        d_loss = - torch.mean(pred_real) + torch.mean(pred_fake) + lambda_gp * gp
        optimD.zero_grad()
        d_loss.backward()
        optimD.step()

        # ====== 2) 每 n_critic 次更新 G + U ======
        if i % param['train']['wgan']['ncritic'] == 0:
            # 2.1 G: 重构 + MMD
            recon = netG(mel)
            logit_real, D_feats_real = netD(mel, return_pyramid=True)
            logit_fake, D_feats_fake = netD(recon, return_pyramid=True)

            recon_l = MSE(recon, mel)
            mmd_l = pyramid_mmd(D_feats_fake, D_feats_real, w=[1.0, 0.7, 0.5])
            g_loss = recon_l + param['train']['lambda_mmd'] * mmd_l

            optimG.zero_grad()
            g_loss.backward(retain_graph=True)
            optimG.step()

            # 2.2 U: 扩散噪声回归
            t = sample_timesteps(recon.size(0), helper.T, device)
            x_t, eps = q_sample(recon.detach(), t, helper)
            eps_pred = netU(x_t, t, D_feats_real)
            diff_l = F.mse_loss(eps_pred, eps)

            optimU.zero_grad()
            (param['train']['lambda_diff'] * diff_l).backward()
            optimU.step()

            # 统计
            aver_loss['recon'] += recon_l.item()
            aver_loss['d2g']   += mmd_l.item()
            aver_loss['gloss'] += (g_loss + diff_l).item()
            gloss_num += 1

        # ====== 写入 CSV（每步/每N步均可，这里每100步写一次） ======
        if metrics_csv_path and (global_step % 100 == 0) and gloss_num > 0:
            row = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch_idx,
                "step_in_epoch": i,
                "global_step": global_step,
                "bs": current_bs,
                "loss_recon": aver_loss['recon']/gloss_num,
                "loss_mmd":   aver_loss['d2g']/gloss_num,
                "loss_g_total": aver_loss['gloss']/gloss_num
            }
            append_metrics(metrics_csv_path, csv_fields, row)

        # ====== 按步保存 last（强烈建议开启，默认每 5000 步） ======
        if checkpoint_last_path and (global_step > 0) and (global_step % save_every_steps == 0):
            scaler_state = None
            save_checkpoint(checkpoint_last_path, netD, netG, netU,
                            optimD, optimG, optimU,
                            epoch=epoch_idx, step_in_epoch=i+1, global_step=global_step,
                            best_aver=best_aver, current_bs=current_bs, scaler_state=scaler_state)
            if logger: logger.info(f"[checkpoint] step={global_step} saved to {checkpoint_last_path}")

            # 同步写 run_state.json，外部随时可读
            root, ext = os.path.splitext(checkpoint_last_path)
            run_state_path = f"{root}_run_state.json"
            dump_run_state(run_state_path, {
                "epoch": epoch_idx, "step_in_epoch": i+1, "global_step": global_step,
                "bs": current_bs, "last_ckpt": checkpoint_last_path
            })

        global_step += 1  # 全局步数自增

    # epoch 末尾均值
    aver_loss['recon'] /= max(gloss_num, 1)
    aver_loss['d2g']   /= max(gloss_num, 1)
    aver_loss['gloss']  = f"{(aver_loss['gloss']/max(gloss_num, 1)):.4e}"

    return netD, netG, aver_loss, global_step



def get_d_aver_emb(netD, train_set, device):
    netD.eval()
    train_embs = {mid: [] for mid in param['all_mid']}
    with torch.no_grad():
        for idx in range(train_set.get_clip_num()):
            mel, mid, _ = train_set.get_clip_data(idx)
            mel = torch.from_numpy(mel).to(device)
            _, feat_real = netD(mel)                     # [B, C, H, W]
            feat_real = feat_real.mean(dim=(0, 2, 3))    # [C]
            train_embs[mid].append(feat_real.cpu().numpy().astype(np.float32))
    for mid in train_embs.keys():
        train_embs[mid] = np.array(train_embs[mid], dtype=np.float32)  # [N_clips, C]
    return train_embs



def train(netD, netG, netU, tr_dataset, te_ld, optimD, optimG, optimU, logger, device, helper, param, best_aver=None,
          start_epoch=0, start_step_in_epoch=0, global_step=0, resume_bs=None, bs_ladder=None, run_dir=None):
    d2g_eff = param['train']['wgan']['feat_match_eff']
    logger.info("============== MODEL TRAINING ==============")

    # 路径
    root, ext = os.path.splitext(param['model_pth'])
    last_path    = f"{root}_last{ext}"
    metrics_path = os.path.join(param['log_dir'], "train_metrics.csv")
    run_state_js = f"{root}_run_state.json"

    # 训练控制
    patience   = int(param['train'].get('patience', 0))
    save_every = int(param['train'].get('save_every', 0))     # 每 N 个 epoch 额外存一次 last（按 epoch）
    save_every_steps = int(param['train'].get('save_every_steps', 5000))  # ★新增：按步保存 last
    wait = 0

    # 批量阶梯
    bs_list = bs_ladder if bs_ladder else [param['train']['bs']]
    if resume_bs is not None and resume_bs in bs_list:
        current_bs = resume_bs
        bs_idx = bs_list.index(resume_bs)
    else:
        current_bs = bs_list[0]
        bs_idx = 0

    # 构造 DataLoader（可续步）
    tr_ld = build_train_loader(tr_dataset, current_bs)

    for epoch_i in range(start_epoch, param['train']['epoch']):
        # 如果是续跑同一 epoch，需要从某个 step 接续
        if epoch_i == start_epoch and start_step_in_epoch > 0:
            step0 = start_step_in_epoch
        else:
            step0 = 0

        start_time = time.time()

        netD, netG, aver_loss, global_step = train_one_epoch(
            netD, netG, tr_ld, optimD, optimG, device, d2g_eff, netU, optimU, helper, param,
            epoch_idx=epoch_i, start_step=step0, global_step=global_step,
            metrics_csv_path=metrics_path,
            save_every_steps=save_every_steps,
            checkpoint_last_path=last_path,
            best_aver=best_aver,
            scaler=None, use_amp=False,       # 如要 AMP，这里替换为你的 scaler / use_amp
            current_bs=current_bs,
            logger=logger
        )

        # epoch 结束，清零“续步”
        start_step_in_epoch = 0

        # 评估
        train_embs = get_d_aver_emb(netD, tr_ld.dataset, device)
        mt_aver, metric = np.zeros((len(param['mt']['test']), 3)), {}
        for j, mt in enumerate(te_ld.keys()):
            mt_aver[j], metric[mt] = test(netD, netG, netU, te_ld[mt], train_embs, logger, device, helper, param)
        aver_all_mt = np.mean(mt_aver[:, 2])

        # 更新 best / 早停计数
        improved = (best_aver is None) or (aver_all_mt > best_aver + 1e-6)
        if improved:
            best_aver = aver_all_mt
            bestD = copy.deepcopy(netD.state_dict())
            bestG = copy.deepcopy(netG.state_dict())
            wait = 0
        else:
            wait += 1

        logger.info('epoch {}: [recon: {:.4e}] [mmd: {:.4e}] [gloss: {}] [best: {:.4f}] [time: {:.0f}s] [bs: {}]'.format(
                    epoch_i, float(aver_loss['recon']), float(aver_loss['d2g']), aver_loss['gloss'],
                    best_aver, time.time() - start_time, current_bs))
        for j, mt in enumerate(param['mt']['test']):
            logger.info('{}: [AUC: {:.4f}] [pAUC: {:.4f}] [aver: {:.4f}] [metric: {}] '
                        .format(mt, mt_aver[j, 0], mt_aver[j, 1], mt_aver[j, 2], metric[mt]))

        # 保存 best
        torch.save({'netD': bestD, 'netG': bestG, 'netU': netU.state_dict(), 'best_aver': best_aver}, param['model_pth'])

        # 保存 last（按 epoch）
        if (save_every == 0) or ((epoch_i + 1) % save_every == 0):
            save_checkpoint(last_path, netD, netG, netU, optimD, optimG, optimU,
                            epoch=epoch_i+1, step_in_epoch=0, global_step=global_step,
                            best_aver=best_aver, current_bs=current_bs, scaler_state=None)
            logger.info(f"[checkpoint-epoch] saved to {last_path}")

        # 同步 run_state.json
        dump_run_state(run_state_js, {
            "epoch": epoch_i+1, "step_in_epoch": 0, "global_step": global_step,
            "bs": current_bs, "best_aver": best_aver, "last_ckpt": last_path
        })

        # ====== 逐档提升 bs（下一轮 epoch 再生效）======
        if (bs_idx + 1) < len(bs_list):
            next_bs = bs_list[bs_idx + 1]
            logger.info(f"[bs ladder] try to increase bs: {current_bs} -> {next_bs}")
        
            new_loader = build_train_loader(tr_dataset, next_bs, logger=logger)
            if new_loader is not None:
                tr_ld = new_loader
                current_bs = next_bs
                bs_idx += 1
                logger.info(f"[bs ladder] switch to bs={current_bs} for next epoch")
            else:
                logger.warning(f"[bs ladder] failed to switch to bs={next_bs}, keep bs={current_bs}")


        # 早停
        if patience > 0 and wait >= patience:
            logger.info(f"[early stop] no improvement for {patience} evals; stop at epoch {epoch_i}")
            break



# @profile
def test(netD, netG, netU, te_ld, train_embs, logger, device, helper,param):
    # detect_location, score_type, score_comb= ('x', 'z'), ('2', '1'), ('sum', 'min', 'max')
    D_metric = ['D_maha', 'D_knn', 'D_lof', 'D_cos']
    G_metric = ['G_x_2_sum', 'G_x_2_min', 'G_x_2_max', 'G_x_1_sum', 'G_x_1_min', 'G_x_1_max',
                'G_z_2_sum', 'G_z_2_min', 'G_z_2_max', 'G_z_1_sum', 'G_z_1_min', 'G_z_1_max',
                'G_z_cos_sum', 'G_z_cos_min', 'G_z_cos_max']
    New_metric = ['U_refine_delta', 'U_diff_residual'] 
    all_metric = D_metric + G_metric + New_metric
    edetect = EDIS.EmbeddingDetector(train_embs)
    edfunc = {'maha': edetect.maha_score, 'knn': edetect.knn_score,
              'lof': edetect.lof_score, 'cos': edetect.cos_score}
    metric2id = {m: meid for m, meid in zip(all_metric, range(len(all_metric)))}
    id2metric = {v: k for k, v in metric2id.items()}

    def specfunc(x):
        return x.sum(axis=tuple(list(range(1, x.ndim))))
    stfunc = {'2': lambda x, y: (x - y).pow(2),
              '1': lambda x, y: (x - y).abs(),
              'cos': lambda x, y: 1 - F.cosine_similarity(x, y)}
    scfunc = {'sum': lambda x: x.sum().item(),
              'min': lambda x: x.min().item(),
              'max': lambda x: x.max().item()}

    netD.eval()
    netG.eval()
    netU.eval()
    # {mid: []}
    y_true_all, y_score_all = [{} for _ in metric2id.keys()], [{} for _ in metric2id.keys()]
    with torch.no_grad():
        for mel, mid, status in te_ld:  # mel: 1*186*1*128*128
            mel = mel.squeeze(0).to(device)
            recon = netG(mel)
            melz  = netG(mel, outz=True)
            reconz = netG(recon, outz=True)

            # ---- 一次性拿到 D 的 pyramid 特征（供 MMD/U条件） ----
            _, cond_feats = netD(mel, return_pyramid=True)
            feat_last = cond_feats[-1]                               # [B, C, H, W]
            feat_vec  = feat_last.mean(dim=(0, 2, 3))                # [C]
            feat_t_np = feat_vec.unsqueeze(0).cpu().numpy().astype(np.float32)  # [1, C]

            assert feat_t_np.ndim == 2, f"feat_t_np shape should be [1,C], got {feat_t_np.shape}"


            mid_i, status_i = mid.item(), status.item()

            # ---- 扩散精修（按 losses.py 的签名）----
            recon_ref, diff_residual_sum = ddim_refine(
                x0=recon,
                netU=netU,
                cond_feats=cond_feats,
                helper=helper,
                steps=param['train']['ddim_steps'],
                eta=0.0,
                guidance_fn=None if not param['train'].get('use_guidance', False)
                            else (lambda x: -netD(x)[0].mean()),
                guidance_scale=param['train'].get('guidance_scale', 0.0)
            )
            S_ref  = (recon_ref - recon).abs().sum().item()
            S_diff = float(diff_residual_sum)

            # ---- 原 D/G 各指标 ----
            for idx, metric in id2metric.items():
                wn = metric.split('_')[0]
                if wn == 'D':
                    dname = metric.split('_')[1]
                    score = edfunc[dname](feat_t_np)
                elif wn == 'G':
                    dd, st, sc = tuple(metric.split('_')[1:])
                    ori = mel if dd == 'x' else melz
                    hat = recon if dd == 'x' else reconz
                    score = scfunc[sc](specfunc(stfunc[st](hat, ori)))
                else:
                    # 跳过新指标，这里单独写入
                    continue

                y_true_all[idx].setdefault(mid_i, []).append(status_i)
                y_score_all[idx].setdefault(mid_i, []).append(score)

            # ---- 写入两条新增指标 ----
            idx_ref  = metric2id['U_refine_delta']
            idx_diff = metric2id['U_diff_residual']
            y_true_all[idx_ref].setdefault(mid_i, []).append(status_i)
            y_score_all[idx_ref].setdefault(mid_i, []).append(S_ref)
            y_true_all[idx_diff].setdefault(mid_i, []).append(status_i)
            y_score_all[idx_diff].setdefault(mid_i, []).append(S_diff)


    aver_of_all_me = []
    for idx in range(len(y_true_all)):
        result = []
        y_true = dict(sorted(y_true_all[idx].items(), key=lambda t: t[0]))  # sort by machine id
        y_score = dict(sorted(y_score_all[idx].items(), key=lambda t: t[0]))
        for mid in y_true.keys():
            AUC_mid = metrics.roc_auc_score(y_true[mid], y_score[mid])
            pAUC_mid = metrics.roc_auc_score(y_true[mid], y_score[mid], max_fpr=param['detect']['p'])
            result.append([AUC_mid, pAUC_mid])
        aver_over_mid = np.mean(result, axis=0)
        aver_of_m = np.mean(aver_over_mid)
        aver_of_all_me.append([aver_over_mid[0], aver_over_mid[1], aver_of_m])
    aver_of_all_me = np.array(aver_of_all_me)
    best_aver = np.max(aver_of_all_me[:, 2])
    best_idx = np.where(aver_of_all_me[:, 2] == best_aver)[0][0]
    best_metric = id2metric[best_idx]

    logger.info('-' * 110)
    return aver_of_all_me[best_idx, :], best_metric


def main(logger):
    train_data = seg_set(param, param['train_set'], 'train')
    param['all_mid'] = train_data.get_mid()

    device = torch.device('cuda:{}'.format(param['card_id']))
    netD = Discriminator(param).to(device)
    netG = AEDC(param).to(device)
    netU = DiffusionRefiner(param).to(device)

    # Diffusion helper
    betas = make_beta_schedule(
        T=param['model'].get('T_diffusion', 1000),
        schedule=param['model'].get('beta_schedule', 'linear'),
        device=device
    )
    helper = DiffusionHelper(betas)

    logger.info(f"[flags] resume={param['resume']} eval_only={param.get('eval_only', False)} "
                f"epochs={param['train']['epoch']} ft_lr_scale={param['train'].get('ft_lr_scale',1.0)} "
                f"patience={param['train'].get('patience',0)} save_every={param['train'].get('save_every',0)}")

    if param.get('eval_only', False):
        tr_ld = build_train_loader(train_data, param['train']['bs'])  # 评估也要 loader 提供 dataset
        train_embs = get_d_aver_emb(netD, tr_ld.dataset, device)
        te_ld = {}
        for mt in param['mt']['test']:
            mt_test_set = clip_set(param, mt, 'dev', 'test')
            te_ld[mt] = DataLoader(mt_test_set, batch_size=1, shuffle=False, num_workers=0)
        mt_aver, metric = np.zeros((len(param['mt']['test']), 3)), {}
        for j, mt in enumerate(te_ld.keys()):
            mt_aver[j], metric[mt] = test(netD, netG, netU, te_ld[mt], train_embs, logger, device, helper, param)
        logger.info('EVAL-ONLY aver_all_mt={:.4f}'.format(np.mean(mt_aver[:, 2])))
        return

    # 优化器
    lr_scale = param['train'].get('ft_lr_scale', 1.0) if param['resume'] else 1.0
    optimD = torch.optim.Adam(netD.parameters(), lr=param['train']['lrD']*lr_scale,
                              betas=(param['train']['beta1'], 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=param['train']['lrG']*lr_scale,
                              betas=(param['train']['beta1'], 0.999))
    optimU = torch.optim.Adam(netU.parameters(), lr=param['train']['lrU'], betas=(0.9, 0.999))

    # ===== 真正的 resume 加载 =====
    ckpt, resume_path = try_load_checkpoint(param['resume'], param['model_pth'])
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_aver = None
    resume_bs = None

    if ckpt is not None:
        logger.info(f"[resume] load from {resume_path}")
        netD.load_state_dict(ckpt['netD'], strict=False)
        netG.load_state_dict(ckpt['netG'], strict=False)
        if 'netU' in ckpt: netU.load_state_dict(ckpt['netU'], strict=False)

        if 'optimD' in ckpt: optimD.load_state_dict(ckpt['optimD'])
        if 'optimG' in ckpt: optimG.load_state_dict(ckpt['optimG'])
        if 'optimU' in ckpt: optimU.load_state_dict(ckpt['optimU'])

        start_epoch       = int(ckpt.get('epoch', 0))
        start_step_in_epoch = int(ckpt.get('step_in_epoch', 0))
        global_step       = int(ckpt.get('global_step', 0))
        best_aver         = ckpt.get('best_aver', None)
        resume_bs         = ckpt.get('current_bs', None)

        # 恢复随机态（可选）
        if 'rng_state' in ckpt:
            torch.set_rng_state(ckpt['rng_state'])
        if 'cuda_rng_state' in ckpt:
            try:
                torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
            except Exception:
                pass

        logger.info(f"[resume] start_epoch={start_epoch}, start_step_in_epoch={start_step_in_epoch}, "
                    f"global_step={global_step}, resume_bs={resume_bs}, best_aver={best_aver}")

    # === 构造测试集 loader ===
    te_ld = {}
    for mt in param['mt']['test']:
        mt_test_set = clip_set(param, mt, 'dev', 'test')
        te_ld[mt] = DataLoader(mt_test_set, batch_size=1, shuffle=False, num_workers=0)

    # === 进入训练（含 bs 阶梯、续步、CSV/JSON 持久化）===
    train(netD, netG, netU,
          tr_dataset=train_data, te_ld=te_ld,
          optimD=optimD, optimG=optimG, optimU=optimU,
          logger=logger, device=device, helper=helper, param=param,
          best_aver=best_aver,
          start_epoch=start_epoch, start_step_in_epoch=start_step_in_epoch, global_step=global_step,
          resume_bs=resume_bs, bs_ladder=BS_LADDER, run_dir=param['log_dir'])



if __name__ == '__main__':
    mt_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    card_num = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mt', choices=mt_list, default='fan')
    parser.add_argument('-c', '--card_id', type=int, choices=list(range(card_num)), default=6)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=783)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=None, help='finetune epochs; override config')
    parser.add_argument('--ft_lr_scale', type=float, default=1.0, help='scale LR when resume (e.g., 0.5)')
    parser.add_argument('--patience', type=int, default=0, help='early stop patience (0=disable)')
    parser.add_argument('--save_every', type=int, default=0, help='save a last checkpoint every N epochs (0=disable)')
    parser.add_argument('--tag', type=str, default='', help='suffix for model filename to avoid overwrite, e.g., ft1')

    opt = parser.parse_args()

    utils.set_seed(opt.seed)
    param['card_id'] = opt.card_id
    param['resume'] = opt.resume
    param['mt'] = {'train': [opt.mt], 'test': [opt.mt]}
    param['eval_only'] = opt.eval_only  

    # ============ 覆盖 epoch（可选：用于微调只跑少量轮次） ============
    if opt.epochs is not None:
        # 覆盖 config.yaml 里的训练轮数
        param.setdefault('train', {})
        param['train']['epoch'] = opt.epochs
    
    # ============ 微调相关（学习率缩放 / 早停 / 定期保存） ============
    param['train']['ft_lr_scale'] = opt.ft_lr_scale   # 仅 resume 时在 main() 里生效
    param['train']['patience']    = opt.patience      # 早停，0 表示关闭
    param['train']['save_every']  = opt.save_every    # 每 N 个 epoch 额外存一次 last，0 表示关闭
    
    param['model_pth'] = utils.get_model_pth(param)
    if opt.tag:
        # 给文件名加后缀，例如 *_ft1.pth
        root, ext = os.path.splitext(param['model_pth'])
        param['model_pth'] = f"{root}_{opt.tag}{ext}"

    
    for dir in [param['model_dir'], param['spec_dir'], param['log_dir']]:
        os.makedirs(dir, exist_ok=True)

    logger = utils.get_logger(param)
    logger.info(f'Seed: {opt.seed}')
    logger.info(f"Train Machine: {param['mt']['train']}")
    logger.info(f"Test Machine: {param['mt']['test']}")
    logger.info('============== TRAIN CONFIG SUMMARY ==============')
    summary = utils.config_summary(param)
    for key in summary.keys():
        message = '{}: '.format(key)
        for k, v in summary[key].items():
            message += '[{}: {}] '.format(k, summary[key][k])
        logger.info(message)

    main(logger)
