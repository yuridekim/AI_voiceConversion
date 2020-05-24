#!/usr/bin/env python3
#coding=utf8
import torch
import torch.nn.functional as F
import numpy as np

class LogManager:
    def __init__(self):
        self.log_book=dict()
    def alloc_stat_type(self, stat_type):
        self.log_book[stat_type] = []
    def alloc_stat_type_list(self, stat_type_list):
        for stat_type in stat_type_list:
            self.alloc_stat_type(stat_type)
    def init_stat(self):
        for stat_type in self.log_book.keys():
            self.log_book[stat_type] = []
    def add_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat)
    def add_torch_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat.detach().cpu().item())
    def get_stat(self, stat_type):
        result_stat = 0
        stat_list = self.log_book[stat_type]
        if len(stat_list) != 0:
            result_stat = np.mean(stat_list)
            result_stat = np.round(result_stat, 4)
        return result_stat

    def print_stat(self):
        for stat_type in self.log_book.keys():
           stat = self.get_stat(stat_type)
           if stat != 0:
            print(stat_type,":",stat, end=' / ')
        print(" ")


def calc_gaussprob(x, mu, log_var):
    c = torch.log(2.*torch.from_numpy(np.array(3.141592)))
    
    var = torch.exp(log_var)
    x_mu2 = (x - mu).pow(2)  
    x_mu2_over_var = torch.div(x_mu2, var + 1e-6)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    
    return torch.mean(log_prob)

def calc_kl_vae(mu, logvar):
    mu2 = torch.zeros_like(mu)
    logvar2 = torch.zeros_like(logvar)
    logvar = logvar.exp()
    logvar2 = logvar2.exp()
    
    mu_diff_sq = mu - mu2
    mu_diff_sq = mu_diff_sq.pow(2)
    
    dimwise_kld = .5 * (
        (logvar2 - logvar) + torch.div(logvar + mu_diff_sq, logvar2 + 1e-6) - 1.)
    
    return torch.mean(dimwise_kld)
