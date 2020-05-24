import os, sys
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
from loss import LogManager, calc_gaussprob, calc_kl_vae
import pickle

import model
from speech_tools import feat_loader
from itertools import combinations


def load_sp(feat_dir, num_mcep=36):
    feat_path = os.path.join(feat_dir, 'cache{}.p'.format(num_mcep))
    with open(feat_path, 'rb') as f:
        sp, sp_m, sp_s, logf0_m, logf0_s = pickle.load(f)
    return sp


def make_one_hot_vector(spk_idx, spk_num):
    vec = np.zeros(spk_num)
    vec[spk_idx] = 1.0
    return vec


def split_train_dev(datadict, train_percent=0.8):
    train_dict = dict()
    dev_dict = dict()
    for spk_id, cur_data in datadict.items():
        datanum = len(cur_data)
        train_num = int(datanum * train_percent)
        train_dict[spk_id] = cur_data[:train_num]
        dev_dict[spk_id] = cur_data[train_num:]
    return train_dict, dev_dict


def calc_parm_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def update_parm(opt_list, loss):
    for opt in opt_list:
        opt.zero_grad()
    loss.backward()
    for opt in opt_list:
        opt.step()


"""
VAE 1: Vanila
VAE 2: Decoder Speaker vector
VAE 3: All Speaker vector
MD: Multi Decoder

SI: Minimize speaker info (cross entropy) of latent
I: Minimize speaker entropy of latent

LI: Maximize ppg info of latent => ALC: ppg loss in converted x
AC: speaker loss in converted x

SC: l1(latent - cycle latent)
CC: cycle loss

GAN : discriminator
"""

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str)
parser.add_argument('--CC', type=int, default=0)


parser.add_argument('--test_mode', type=int, default=0)
parser.add_argument('--model_dir', default='')
parser.add_argument('--lr', type=float, default=0)

args = parser.parse_args()
assert args.model_type in ["VAE3", "MD"]

# Data load
SPK_LIST = ['VCC2SF1', 'VCC2SF2', 'VCC2SM1', 'VCC2SM2']
TOTAL_SPK_NUM = len(SPK_LIST)
SPK_DICT = {
    spk_idx: spk_id
    for spk_idx, spk_id in enumerate(SPK_LIST)
}
VEC_DICT = {
    spk_id: make_one_hot_vector(spk_idx, TOTAL_SPK_NUM)  # one-hot
    for spk_idx, spk_id in SPK_DICT.items()
}

SP_DICT = {
    spk_id: load_sp(os.path.join("data", "train", spk_id))
    for spk_id in SPK_LIST
}
SP_DICT_TRAIN, SP_DICT_DEV = split_train_dev(SP_DICT, train_percent=0.8)

# Model initilaization
# model_dir = args.model_dir
model_dir = "model/" + args.model_type

if args.CC:
    model_dir += "_CC"

lr = 0.0001
coef_dict = {
    "VAE": {"rec": 3.0, "cyc": 1.0, "adv": 1.0, "kl": 1.0},
    "CC": {"rec": 10.0, "cyc": 5.0, "adv": 1.0, "kl": 1.0},
    "MD_CC": {"rec": 3.0, "cyc": 10.0, "adv": 1.0, "kl": 1.0}
}


if args.CC:
    if args.model_type == "MD":
        # lr = 0.001
        coef = coef_dict["MD_CC"]
    else:
        coef = coef_dict["CC"]
else:
    coef = coef_dict["VAE"]

print(model_dir)
os.makedirs(model_dir + "/parm", exist_ok=True)

latent_dim = 8

is_MD = True if args.model_type == "MD" else False

## Encoder
Enc = model.Encoder(style_dim=4, latent_dim=latent_dim, vae_type=args.model_type)
Enc.cuda()
Enc_opt = optim.Adam(Enc.parameters(), lr=lr)
print(calc_parm_num(Enc))
## Decoder
if is_MD:
    # Enc.load_state_dict(torch.load("model/VAE3_base/final_enc.pt"))

    Dec_group = dict()
    Dec_opt_group = dict()
    for spk_id in SPK_DICT.values():
        Dec_group[spk_id] = model.Decoder(style_dim=4, latent_dim=latent_dim, vae_type=args.model_type)
        Dec_group[spk_id].cuda()
        # Dec_group[spk_id].load_state_dict(torch.load("model/MD_base/final_"+spk_id+"_dec.pt"))

        Dec_opt_group[spk_id] = optim.Adam(Dec_group[spk_id].parameters(), lr=lr)

else:
    Dec = model.Decoder(style_dim=4, latent_dim=latent_dim, vae_type=args.model_type)
    Dec.cuda()
    Dec_opt = optim.Adam(Dec.parameters(), lr=lr)

## Discriminator


# 8 16
# (0-499) (500-999)
init_batch_size = 2
epochs = 2000
print("Training Settings")
print("LR", lr)
print("Batch Size", init_batch_size)
print("Number of epochs", epochs)
print(".....................")
lm = LogManager()
lm.alloc_stat_type_list(["rec_loss", "kl_loss", "cyc_loss", "D_adv_loss", "G_adv_loss"])

total_time = 0

clip_val = 0.05

min_dev_loss = 9999999999999999
min_epoch = 0
d_epoch = 1

batch_size = init_batch_size
coef["adv"] = 0.0
coef["cyc"] = 0.0
for epoch in range(epochs):
    print("EPOCH:", epoch)
    spk_pair = list(combinations(SPK_LIST, 2))
    np.random.shuffle(spk_pair)

    batch_size = init_batch_size * pow(2, epoch // 500)

    if epoch == 1000:
        coef["cyc"] = 1.0

    is_cyc = args.CC and coef["cyc"] > 0.0
    if epoch % 1000 == 0:
        print("Batch Size", batch_size)
        print("Cyc training:", is_cyc)

    lm.init_stat()

    start_time = time.time()
    # Discriminator Training

    # VAE Training
    Enc.train()
    if is_MD:
        for m in Dec_group.values():
            m.train()
    else:
        Dec.train()

    for A_spk, B_spk in spk_pair:
        A_loader = feat_loader(SP_DICT_TRAIN[A_spk], VEC_DICT[A_spk], batch_size, shuffle=True)
        B_loader = feat_loader(SP_DICT_TRAIN[B_spk], VEC_DICT[B_spk], batch_size, shuffle=True)

        if is_MD:
            Dec_A = Dec_group[A_spk]
            Dec_B = Dec_group[B_spk]
            Dec_opt_A = Dec_opt_group[A_spk]
            Dec_opt_B = Dec_opt_group[B_spk]
        else:
            Dec_A = Dec
            Dec_B = Dec
            Dec_opt_A = Dec_opt;
            Dec_opt_B = Dec_opt


        for (A_x, A_y), (B_x, B_y) in zip(A_loader, B_loader):
            A_mu, A_logvar, A_z = Enc(A_x, A_y)
            B_mu, B_logvar, B_z = Enc(B_x, B_y)

            # VAE
            A2A_mu, A2A_logvar, A2A = Dec_A(A_z, A_y)
            B2B_mu, B2B_logvar, B2B = Dec_B(B_z, B_y)

            rec_loss = -calc_gaussprob(A_x, A2A_mu, A2A_logvar) - calc_gaussprob(B_x, B2B_mu, B2B_logvar)
            kl_loss = calc_kl_vae(A_mu, A_logvar) + calc_kl_vae(B_mu, B_logvar)
            total_loss = coef["rec"] * rec_loss + coef["kl"] * kl_loss

            if is_cyc:
                A2B_mu, A2B_logvar, A2B = Dec_B(A_z, B_y)
                B2A_mu, B2A_logvar, B2A = Dec_A(B_z, A_y)

            # CYC
            if is_cyc:
                A2B_z_mu, A2B_z_logvar, A2B_z = Enc(A2B, B_y)
                B2A_z_mu, B2A_z_logvar, B2A_z = Enc(B2A, A_y)

                A2B2A_mu, A2B2A_logvar, _ = Dec_A(A2B_z, A_y)
                B2A2B_mu, B2A2B_logvar, _ = Dec_B(B2A_z, B_y)

                cyc_loss = -calc_gaussprob(A_x, A2B2A_mu, A2B2A_logvar) - calc_gaussprob(B_x, B2A2B_mu, B2A2B_logvar)
                cyc_kl_loss = calc_kl_vae(A2B_z_mu, A2B_z_logvar) + calc_kl_vae(B2A_z_mu, B2A_z_logvar)
                kl_loss += cyc_kl_loss
                total_loss = coef["cyc"] * cyc_loss + coef["kl"] * cyc_kl_loss

            # Update
            if is_MD:
                update_parm([Enc_opt, Dec_opt_A, Dec_opt_B], total_loss)
            else:
                update_parm([Enc_opt, Dec_opt], total_loss)

            # write to log
            lm.add_torch_stat("rec_loss", rec_loss)
            lm.add_torch_stat("kl_loss", kl_loss)

            if is_cyc:
                lm.add_torch_stat("cyc_loss", cyc_loss)

    print("Train:", end=' ')
    lm.print_stat()
    # VAE Evaluation
    lm.init_stat()
    Enc.eval()
    if is_MD:
        for m in Dec_group.values():
            m.eval()
    else:
        Dec.eval()
    for A_spk, B_spk in spk_pair:
        A_loader = feat_loader(SP_DICT_DEV[A_spk], VEC_DICT[A_spk], 81 - int(81 * 0.8), shuffle=False)
        B_loader = feat_loader(SP_DICT_DEV[B_spk], VEC_DICT[B_spk], 81 - int(81 * 0.8), shuffle=False)

        if is_MD:
            Dec_A = Dec_group[A_spk]
            Dec_B = Dec_group[B_spk]

        else:
            Dec_A = Dec;
            Dec_B = Dec


        with torch.no_grad():
            for (A_x, A_y), (B_x, B_y) in zip(A_loader, B_loader):
                A_mu, A_logvar, A_z = Enc(A_x, A_y)
                B_mu, B_logvar, B_z = Enc(B_x, B_y)

                # VAE
                A2A_mu, A2A_logvar, A2A = Dec_A(A_z, A_y)
                B2B_mu, B2B_logvar, B2B = Dec_B(B_z, B_y)

                rec_loss = -calc_gaussprob(A_x, A2A_mu, A2A_logvar) - calc_gaussprob(B_x, B2B_mu, B2B_logvar)
                kl_loss = calc_kl_vae(A_mu, A_logvar) + calc_kl_vae(B_mu, B_logvar)

                if is_cyc:
                    A2B_mu, A2B_logvar, A2B = Dec_B(A_z, B_y)
                    B2A_mu, B2A_logvar, B2A = Dec_A(B_z, A_y)



                # CYC
                if is_cyc:
                    A2B_z_mu, A2B_z_logvar, A2B_z = Enc(A2B, B_y)
                    B2A_z_mu, B2A_z_logvar, B2A_z = Enc(B2A, A_y)

                    A2B2A_mu, A2B2A_logvar, _ = Dec_A(A2B_z, A_y)
                    B2A2B_mu, B2A2B_logvar, _ = Dec_B(B2A_z, B_y)

                    cyc_loss = -calc_gaussprob(A_x, A2B2A_mu, A2B2A_logvar) - calc_gaussprob(B_x, B2A2B_mu,
                                                                                             B2A2B_logvar)
                    cyc_kl_loss = calc_kl_vae(A2B_z_mu, A2B_z_logvar) + calc_kl_vae(B2A_z_mu, B2A_z_logvar)
                    kl_loss += cyc_kl_loss

                # write to log
                lm.add_torch_stat("rec_loss", rec_loss)
                lm.add_torch_stat("kl_loss", kl_loss)

                if is_cyc:
                    lm.add_torch_stat("cyc_loss", cyc_loss)

    print("DEV:", end=' ')
    lm.print_stat()
    end_time = time.time()

    total_time += (end_time - start_time)

    print(".....................")

    if epoch % 10 == 0:
        ### check min loss
        if is_cyc:
            cur_loss = lm.get_stat("rec_loss") + lm.get_stat("cyc_loss")
        else:
            cur_loss = lm.get_stat("rec_loss")
        if np.isnan(cur_loss):
            print("Nan at", epoch)
            break

        if min_dev_loss > cur_loss:
            min_dev_loss = cur_loss
            min_epoch = epoch

        ### Parmaeter save
        torch.save(Enc.state_dict(), os.path.join(model_dir, "parm", str(epoch) + "_enc.pt"))

        if args.model_type == "MD":
            for spk_id, Dec in Dec_group.items():
                torch.save(Dec.state_dict(), os.path.join(model_dir, "parm", str(epoch) + "_" + spk_id + "_dec.pt"))
        else:
            torch.save(Dec.state_dict(), os.path.join(model_dir, "parm", str(epoch) + "_dec.pt"))

print("***********************************")
print("Model name:", model_dir.split("/")[-1])
print("TIME PER EPOCH:", total_time / epochs)
print("Final Epoch:", min_epoch, min_dev_loss)
print("***********************************")

# torch.save(Enc.state_dict(), os.path.join(model_dir,"final_enc.pt"))
os.system(
    "cp " + os.path.join(model_dir, "parm", str(min_epoch) + "_enc.pt") + " " + os.path.join(model_dir, "final_enc.pt"))
if args.model_type == "MD":
    for spk_id, Dec in Dec_group.items():
        # torch.save(Dec.state_dict(), os.path.join(model_dir,"final_"+spk_id+"_dec.pt"))
        os.system(
            "cp " + os.path.join(model_dir, "parm", str(min_epoch) + "_" + spk_id + "_dec.pt") + " " + os.path.join(
                model_dir, "final_" + spk_id + "_dec.pt"))
else:
    # torch.save(Dec.state_dict(), os.path.join(model_dir,"final_dec.pt"))
    os.system("cp " + os.path.join(model_dir, "parm", str(min_epoch) + "_dec.pt") + " " + os.path.join(model_dir,
                                                                                                       "final_dec.pt"))
