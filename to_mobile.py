import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from model import *

gA2B = Generator(3, 3)
gB2A = Generator(3, 3)
dA = Discriminator(3)
dB = Discriminator(3)

gA2B.load_state_dict(torch.load('weight_G_A2B.pth'))
gB2A.load_state_dict(torch.load('weight_G_B2A.pth'))
dA.load_state_dict(torch.load('weight_D_A.pth'))
dB.load_state_dict(torch.load('weight_D_B.pth'))

gA2B.eval()
gB2A.eval()
dA.eval()
dB.eval()

example = torch.rand(1, 3, 256, 256)

traced = torch.jit.trace(gA2B, example)
traced = optimize_for_mobile(traced)
traced._save_for_lite_interpreter('modelA2B.ptl')
traced = torch.jit.trace(gB2A, example)
traced = optimize_for_mobile(traced)
traced._save_for_lite_interpreter('modelB2A.ptl')
traced = torch.jit.trace(dA, example)
traced = optimize_for_mobile(traced)
traced._save_for_lite_interpreter('modelDA.ptl')
traced = torch.jit.trace(dB, example)
traced = optimize_for_mobile(traced)
traced._save_for_lite_interpreter('modelDB.ptl')
