import math
import random
from typing import List, Tuple
import numpy as np
import torch
from easydict import EasyDict
import logging

def fZout(omega: torch.Tensor, y: torch.Tensor, V: torch.Tensor):
    return torch.sqrt(torch.pi * V / 2) * (1 + y * torch.erf(omega / torch.sqrt(2 * V)))

def fgout(omega: torch.Tensor, y: torch.Tensor, V: torch.Tensor):
    denominator = y * V * torch.exp(-(omega**2) / (V * 2)) 
    return denominator / (fZout(omega, y, V) * V)

def fdgout(omega: torch.Tensor, y: torch.Tensor, V: torch.Tensor):
    denominator = (-y) * torch.exp(-(omega**2) / (V * 2)) * omega * V + math.sqrt(torch.pi / 2) * torch.pow(V, 1.5) * (1 + y * torch.erf(omega / torch.sqrt(2 * V)))
    return denominator / (V**2 * fZout(omega, y, V)) - 1 / V - fgout(omega, y, V)**2

def fw(Sigma: torch.Tensor, T: torch.Tensor):
    return T / (1 + Sigma)

def fc(Sigma: torch.Tensor, T: torch.Tensor):
    return Sigma / (1 + Sigma)

def amp(data: Tuple[torch.Tensor, torch.Tensor], device: torch.device, args: EasyDict):
    x, y = data
    x2 = x**2
    P = args.P
    N = args.N
    sqrtn = math.sqrt(N)
    maxIteration = 1000

    What = torch.rand(size=(N, ), device=device)
    Vhat = torch.rand(size=(N, ), device=device)
    gout = torch.rand(size=(P, ), device=device)
    dgout = torch.rand(size=(P, ), device=device)

    flag = False
    for _ in range(maxIteration):
        V = torch.mv(x2, Vhat) / N
        omega = torch.mv(x, What) / sqrtn - V * gout
        gout = fgout(omega, y, V)
        dgout = fdgout(omega, y, V)
        Sigma = - 1 / (torch.mv(x2.t(), dgout) / N)
        T = What + Sigma * torch.mv(x.t(), gout) / sqrtn
        newWhat = fw(Sigma, T)
        newVhat = fc(Sigma, T)

        if torch.max(torch.abs(torch.stack([newWhat - What, newVhat - Vhat], dim=0))) < 1e-3:
            flag = True
            break
        
        What = newWhat.clone()
        Vhat = newVhat.clone()

    return newWhat, newVhat, flag

def se(device: torch.device, args: EasyDict, initialM: List = None):
    numX = 10000
    numY = 10000
    alpha = args.alphas
    x = torch.normal(0, 1, size=(100000000, ), device=device)
    y = torch.normal(0, 1, size=(100000000, ), device=device)
    tx = torch.normal(0, 1, size=(numX, 1), device=device)
    Q = torch.tensor(1.0, device=device)
    if initialM is not None:
        m = torch.tensor(initialM[0], device=device)
        mhat = torch.tensor(initialM[1], device=device)
    else:
        m = torch.rand(size=(1, ), device=device)
        mhat = torch.rand(size=(1, ), device=device)

    maxIteration = 1000
    flag = False
    for _ in range(maxIteration):
        t1 = alpha * mhat
        newm = torch.mean(fw(1 / t1, x + y / torch.sqrt(t1))**2)

        x1 = (tx * torch.sqrt(m)).repeat_interleave(numY, dim=1)
        y1 = tx * torch.sqrt(Q - m) + x1
        newmhat = -torch.mean(fdgout(x1, torch.sign(y1), Q - m))

        if torch.max(torch.abs(torch.tensor([newm - m, newmhat - mhat]))) < 1e-3:
            flag = True
            break

        m = newm.clone()
        mhat = newmhat.clone()

    q = newm.clone()
    x = torch.normal(0, 1, size=(100000000, ), device=device)
    y = torch.normal(0, 1, size=(100000000, ), device=device)
    error = 1 - torch.mean((torch.sign(torch.sqrt(m) * x) == torch.sign(torch.sqrt(m) * x + torch.sqrt(1 - m) * y)).float())
    return newm.item(), newmhat.item(), q.item(), error.item(), flag


def replica(device: torch.device, args: EasyDict, initialQ: List = None):
    numX = 100000000
    xi = torch.normal(0, 1, size=(numX, ), device=device)
    alpha = args.alphas
    Q = torch.tensor(1.0, device=device)
    if initialQ is not None:
        q = torch.tensor(initialQ[0], device=device)
        qhat = torch.tensor(initialQ[1], device=device)
    else:
        q = torch.rand(size=(1, ), device=device)
        qhat = torch.rand(size=(1, ), device=device)

    maxIteration = 1000
    flag = False
    for _ in range(maxIteration):
        t1 = torch.exp(qhat * (xi ** 2) / (2 * (1 + qhat))) / torch.sqrt(1 + qhat)
        newq = 2 * (-torch.mean(t1 * (1 + qhat - xi**2) * (1 + torch.log(t1)) / (2 * (1 + qhat)**2)))

        t2 = q / (Q - q)
        t = torch.erf(torch.sqrt(t2 / 2) * xi)
        t3 = (1 + t) / 2
        t4 = (1 - t) / 2
        newqhat = 2 * alpha * torch.mean(torch.exp(-t2 * xi**2 / 2) * Q * xi * (torch.log(t3 / (t4 + 1e-12) + 1e-12)) / (2 * math.sqrt(2 * math.pi) * ((Q - q)**2) * torch.sqrt(t2)))

        if not torch.isfinite(newq) or not torch.isfinite(newqhat):
            break

        if torch.max(torch.abs(torch.tensor([newq - q, newqhat - qhat]))) < 1e-3:
            flag = True
            break

        q = newq.clone()
        qhat = newqhat.clone()

        logging.debug(f"q: {q.item():.2f}, qhat: {qhat.item():.2f}")

    q = newq.clone()
    x = torch.normal(0, 1, size=(100000000, ), device=device)
    y = torch.normal(0, 1, size=(100000000, ), device=device)
    error = 1 - torch.mean((torch.sign(torch.sqrt(q) * x) == torch.sign(torch.sqrt(q) * x + torch.sqrt(1 - q) * y)).float())
    return newq.item(), newqhat.item(), error.item(), flag