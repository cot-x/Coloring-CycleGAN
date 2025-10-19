import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

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

example_inputs = (torch.rand(1, 3, 256, 256),)

exported_program = torch.export.export(gA2B, example_inputs)
program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]  # CPU | CoreMLPartitioner() for iOS | QnnPartitioner() for Qualcomm
).to_executorch()
with open("modelA2B.pte", "wb") as f:
    f.write(program.buffer)

exported_program = torch.export.export(gB2A, example_inputs)
program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]  # CPU | CoreMLPartitioner() for iOS | QnnPartitioner() for Qualcomm
).to_executorch()
with open("modelB2A.pte", "wb") as f:
    f.write(program.buffer)

exported_program = torch.export.export(dA, example_inputs)
program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]  # CPU | CoreMLPartitioner() for iOS | QnnPartitioner() for Qualcomm
).to_executorch()
with open("modelDA.pte", "wb") as f:
    f.write(program.buffer)

exported_program = torch.export.export(dB, example_inputs)
program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()]  # CPU | CoreMLPartitioner() for iOS | QnnPartitioner() for Qualcomm
).to_executorch()
with open("modelDB.pte", "wb") as f:
    f.write(program.buffer)
