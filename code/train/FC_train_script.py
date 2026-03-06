# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:19:30 2025

@author: 14026
"""

import torch
import torch.utils.data as data
import torch.nn.functional as f
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import itertools
import time
import numpy as np
import math
import os
import csv
import torch.optim.lr_scheduler as lr_scheduler  # should have a larger patience!!!!!!
#from funs import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

torch.set_printoptions(linewidth=200, precision=3, sci_mode=False)

##########Parameters#########################
dir_train = './training_data/'
Lsize = 40
cutoff = np.sqrt(13)#4#5
neighbor_ramp = 8#9#13
input_size = 45
from_bench_mark = False
size_l = Lsize
size_w = Lsize
num_site = size_l * size_w
num_feature = input_size
epoch_start = 0
duration = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
center_spin_index = 0


#######################################################################
ref_self_a2_content = torch.zeros(3, 3, 2).to(device)
ref_self_b1_content = torch.zeros(3, 3, 2).to(device)
ref_self_b2_content = torch.zeros(3, 3, 2).to(device)

ref_self_a2_content[1, 1, 0] = 1
ref_self_b1_content[1, 1, 0] = 1
ref_self_b2_content[1, 1, 0] = 1

ref_self_a2_content[-1, 1, 1] = 1
ref_self_b1_content[-1, 1, 1] = -1
ref_self_b2_content[-1, 1, 1] = -1

ref_self_a2_content[1, -1, 1] = 1
ref_self_b1_content[1, -1, 1] = -1
ref_self_b2_content[1, -1, 1] = -1

ref_self_a2_content[-1, -1, 0] = 1
ref_self_b1_content[-1, -1, 0] = 1
ref_self_b2_content[-1, -1, 0] = 1

ref_self_a2_content[1, -1, 0] = -1
ref_self_b1_content[1, -1, 0] = 1
ref_self_b2_content[1, -1, 0] = -1

ref_self_a2_content[-1, 1, 0] = -1
ref_self_b1_content[-1, 1, 0] = 1
ref_self_b2_content[-1, 1, 0] = -1

ref_self_a2_content[1, 1, 1] = -1
ref_self_b1_content[1, 1, 1] = -1
ref_self_b2_content[1, 1, 1] = 1

ref_self_a2_content[-1, -1, 1] = -1
ref_self_b1_content[-1, -1, 1] = -1
ref_self_b2_content[-1, -1, 1] = 1

#####################Descriptor####################################

def screenANDscale_tensor2(IR_tensor, tensor_max):
    tensor_min = -1 * tensor_max
    return 2 * (IR_tensor - tensor_min) / (tensor_max - tensor_min) - 1

def read_neighbor_list(file_name, ramp, Lsize):
    num_site = Lsize * Lsize

    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        neighbor_list = list(reader)
    
    neighbor_list_length = torch.zeros([ramp + 1], dtype=torch.int32)
    sum  = 0
    for i in range(0, ramp+1):
        neighbor_list_length[i] = len(neighbor_list[i])
        sum += len(neighbor_list[i])

    #print(neighbor_list_length)
    #print(sum)

    tot_length = sum

 
    neighbor_2d = torch.zeros([num_site, tot_length], dtype=torch.long).to(device)
    count = 0
    for i in range(len(neighbor_list)):
        for j in range(len(neighbor_list[i])):
            neighbor_list[i][j] = int(neighbor_list[i][j])

            m = int(count / tot_length)
            n = count % tot_length 
            neighbor_2d[m][n] = neighbor_list[i][j]

            count += 1
    
    #print(neighbor_2d)

    neighbor_type = torch.zeros(ramp + 1, dtype=torch.int).to(device)

    for i in range(1, ramp + 1):
        if neighbor_list_length[i] == 4:
            if neighbor_list[0][0] // Lsize == neighbor_list[i][0] // Lsize or neighbor_list[0][0] % Lsize == neighbor_list[i][0] % Lsize:
                neighbor_type[i] = 1
            else:
                neighbor_type[i] = 2
        if neighbor_list_length[i] == 8:
            neighbor_type[i] = 3

    return neighbor_2d, tot_length, neighbor_type


def generate_feature_mat(Lsize, num_feature, neighbor_type):
    num_site = Lsize * Lsize
    num_layer = neighbor_type.size(0)

    ir4 = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 0, -1, 0], [0, -1, 0, 1]], dtype=torch.double).to(device)

    ir4_2 = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.double).to(device)

    ir8 = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, -1, 1, -1, 1, -1, 1, -1], [1, 1, -1, -1, 1, 1, -1, -1], [1, -1, -1, 1, 1, -1, -1, 1],
    [1, 1, 0, 0, -1, -1, 0, 0], [0, 0, -1, -1, 0, 0, 1, 1], [0, 0, 1, -1, 0, 0, -1, 1], [1, -1, 0, 0, -1, 1, 0, 0]], dtype=torch.double).to(device)

    mat_ir = torch.tensor([[1.]]).to(device)

    for i in range(1, num_layer):
        if neighbor_type[i] == 1:
            mat_ir = torch.block_diag(mat_ir, ir4)
        if neighbor_type[i] == 2:
            mat_ir = torch.block_diag(mat_ir, ir4_2)
        if neighbor_type[i] == 3:
            mat_ir = torch.block_diag(mat_ir, ir8)


    mat1_1d = torch.tensor([[0.]]).to(device)
    mat1_2d = torch.tensor([[1, 1], [0, 0]], dtype=torch.double).to(device)
    mat1_4 = torch.block_diag(mat1_1d, mat1_1d, mat1_2d)
    mat1_8 = torch.block_diag(mat1_1d, mat1_1d, mat1_1d, mat1_1d, mat1_2d, mat1_2d)

    mat1 = torch.tensor([[0.]]).to(device)
    for i in range(1, num_layer):
        if neighbor_type[i] == 1 or neighbor_type[i] == 2:
            mat1 = torch.block_diag(mat1, mat1_4)
        if neighbor_type[i] == 3:
            mat1 = torch.block_diag(mat1, mat1_8)

    #f1 = open("f1.dat", "w")
    #print(mat1, file=f1)

    mat2_1d = torch.tensor([[1.]]).to(device)
    mat2_2d = torch.tensor([[0, 0], [1, 1]], dtype=torch.double).to(device)
    mat2_4 = torch.block_diag(mat2_1d, mat2_1d, mat2_2d)
    mat2_8 = torch.block_diag(mat2_1d, mat2_1d, mat2_1d, mat2_1d, mat2_2d, mat2_2d)

    mat2 = torch.tensor([[1.]]).to(device)
    for i in range(1, num_layer):
        if neighbor_type[i] == 1 or neighbor_type[i] == 2:
            mat2 = torch.block_diag(mat2, mat2_4)
        if neighbor_type[i] == 3:
            mat2 = torch.block_diag(mat2, mat2_8)

    ref_index = torch.zeros(2, num_site, num_feature, dtype=torch.long).to(device)
    for i in range(num_site):
        for j in range(num_feature):
            ref_index[0][i][j] = i

        ref_index[1][i][0] = 0
        count = 1
        for j in range(1, num_layer):
            if neighbor_type[j] == 1:
                ref_index[1][i][count] = 1
                ref_index[1][i][count + 1] = 3
                ref_index[1][i][count + 2] = 5
                ref_index[1][i][count + 3] = 6
                count += 4
            if neighbor_type[j] == 2:
                ref_index[1][i][count] = 1
                ref_index[1][i][count + 1] = 4
                ref_index[1][i][count + 2] = 5
                ref_index[1][i][count + 3] = 6
                count += 4
            if neighbor_type[j] == 3:
                ref_index[1][i][count] = 1
                ref_index[1][i][count + 1] = 2
                ref_index[1][i][count + 2] = 3
                ref_index[1][i][count + 3] = 4
                ref_index[1][i][count + 4] = 5
                ref_index[1][i][count + 5] = 6
                ref_index[1][i][count + 6] = 5
                ref_index[1][i][count + 7] = 6
                count += 8

    return mat_ir, mat1, mat2, ref_index



def generate_Q_feature(Q, neighbor_2d, mat_ir, mat1, mat2, ref_index):
    num_site = neighbor_2d.size(0)
    num_feature = neighbor_2d.size(1)

    Q_2d = Q[neighbor_2d]
    #print(Q_2d[0])

    mat_ir = mat_ir.expand(num_site, -1, -1)

    Q_2d = Q_2d.unsqueeze(2) 

    bf = torch.matmul(mat_ir, Q_2d)
    bf = bf.squeeze()
    

    mat1 = mat1.expand(num_site, -1, -1)
    mat2 = mat2.expand(num_site, -1, -1)

    #referene layer index is [13:19], total of 6

    ref_cont_Q0 = torch.ones(num_site, 1).to(device)    #for the central Q
    ref_cont_1d = torch.sign(bf[:, 13:17])     #for the reference a1 a2 b1 b2 representation
    ref_cont_1d[:, 0] = 1     #a1 is not changed, for the reference layer we should not change it as it introduces spurious symmetry, for other layers we can do either way
    ref_cont_2d = torch.nn.functional.normalize(bf[:, 17:19], dim=1)  #for the E irreducible representation reference

    ref_cont = torch.cat((ref_cont_Q0, ref_cont_1d, ref_cont_2d), 1)

    ref = ref_cont[ref_index[0], ref_index[1]]

    #=============== for the reference layer, transformation of a2 b1 b2 to be consistent with E
    ref_self_abb_index1 = torch.sign(bf[:, 17]).long()
    ref_self_abb_index2 = torch.sign(bf[:, 18]).long()
    ref_self_abb_index3 = torch.max(torch.abs(bf[:, 17:19]), 1)[1].long()

    ref_self_a2 = ref_self_a2_content[ref_self_abb_index1, ref_self_abb_index2, ref_self_abb_index3]
    ref_self_b1 = ref_self_b1_content[ref_self_abb_index1, ref_self_abb_index2, ref_self_abb_index3]
    ref_self_b2 = ref_self_b2_content[ref_self_abb_index1, ref_self_abb_index2, ref_self_abb_index3]

    ref[:, 14] = ref_self_a2 #something is changed here...why change the referernce scalar?
    ref[:, 15] = ref_self_b1
    ref[:, 16] = ref_self_b2
    #==============
    #print(bf.size(), ref.size())
    #print(mat1.size(), mat2.size())

    feature_ref_E2 = torch.max(torch.abs(bf[:, 17:19]), 1)[0] #this is the 2d reference multiply (1, 0), with 8-fold symmetry removed

    #matmul() cannot do batched matrix times batched vector directly, so have to use unsqueeze() and squeeze()    
    Q_feature = torch.sqrt(torch.matmul( mat1, torch.mul(bf, bf).unsqueeze(2) ).squeeze())

    Q_feature += torch.matmul( mat2, torch.mul(bf, ref).unsqueeze(2) ).squeeze()

    Q_feature[:, 18] = feature_ref_E2
    
    #n = 200
    #print(ref_self_a2[n], ref_self_b1[n], ref_self_b2[n])
    #print(bf[n])
    #print(ref_cont[n])
    #print(ref[n])
    #print(Q_feature[n])

    return Q_feature


         
            

####################End of Descriptor#################################



#####################Train prepare##################################

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(input_size, 4096)
        self.hidden_1 = torch.nn.Linear(4096, 2048)
        self.hidden_2 = torch.nn.Linear(2048, 750)
        self.hidden_3 = torch.nn.Linear(750, 592)
        self.hidden_4 = torch.nn.Linear(592, 256)
        self.hidden_5 = torch.nn.Linear(256, 127)
        self.hidden_6 = torch.nn.Linear(127, 69)
        self.output = torch.nn.Linear(69, 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0, std=0.1)  #0.15
            if module.bias is not None:
                module.bias.data.normal_(mean=0, std=0.05)

    def forward(self, x):
        x = f.relu(self.input(x))
        x = f.relu(self.hidden_1(x))
        x = f.relu(self.hidden_2(x))
        x = f.relu(self.hidden_3(x))
        x = f.relu(self.hidden_4(x))
        x = f.relu(self.hidden_5(x))
        x = f.relu(self.hidden_6(x))
        x = self.output(x)
        return x

#########################################################################
file_name1 = './neighbor/40_ramp_8.csv'
neighbor_2d, num_input, neighbor_type = read_neighbor_list(file_name1, neighbor_ramp, Lsize)
print("number of input: ", num_input)
#print(neighbor_2d[0])
#print(neighbor_type)

mat_ir, mat1, mat2, ref_index = generate_feature_mat(Lsize, num_input, neighbor_type)

net = Net()
net.to(device)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")



optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
######################################################################


Q_tensor_1 = torch.load(dir_train + "Q_random_200.pt").to(device)
force_tensor_1 = torch.load(dir_train + "force_random_200.pt").to(device)
Q_tensor_2 = torch.load(dir_train + "Q_quench_400.pt").to(device)
force_tensor_2 = torch.load(dir_train + "force_quench_400.pt").to(device)
Q_tensor = torch.cat((Q_tensor_1, Q_tensor_2))
force_tensor = torch.cat((force_tensor_1, force_tensor_2))


indices = []

for i in range(0, Q_tensor.shape[0]):
    if ((i+1) % 7 != 0):
        indices.append(i)
    
indices = torch.tensor(indices).to(device)

dim = 0

Q_tensor_train = torch.index_select(Q_tensor,dim,indices).to(device)

force_tensor_train = torch.index_select(force_tensor,dim,indices).to(device)

print(Q_tensor_train.shape)

print(force_tensor_train.shape)




torch_data_set = data.TensorDataset(Q_tensor_train, force_tensor_train)
loader = data.DataLoader(dataset=torch_data_set, batch_size=1, shuffle=True)



if from_bench_mark:
    print("load ./model/model_status.pt")
    prev_model_data = torch.load("./model/" + "more_data.pt")  
    net.load_state_dict(prev_model_data['model'])
    optimizer.load_state_dict(prev_model_data['optimizer'])



saved_loss = 1000
for epoch in range(epoch_start,epoch_start+duration):  #epoch_start,epoch_start+duration
    loss_perstep = []
    for step, (Q, force) in enumerate(loader):
        
        temp_coordinate = Q[0].requires_grad_(False)
        temp_force = force[0].requires_grad_(False)
        
        # print("this is temp_coordiante: ", temp_coordinate.shape) #torch.Size([4096, 49])
        
        # print("this is temp_force: ", temp_force.shape)  #torch.Size([4096])

        # print("this is with the neighbors: ", temp_coordinate[neighbor_2d_1].shape)
        
       
        optimizer.zero_grad()
        
        x_temp = generate_Q_feature(temp_coordinate, neighbor_2d, mat_ir, mat1, mat2, ref_index)
        
        force_prediction = net(x_temp).view(-1)
        
        loss = loss_func(force_prediction, force[0])
        loss_perstep.append(loss.item())
        loss.backward()
        optimizer.step()
        #print("This is loss: ", loss.item())
   
    average_loss = sum(loss_perstep) / len(loss_perstep)
    if (average_loss < saved_loss ):                                   #((epoch+1)%100)==0: or (epoch+1) % 1000 == 0
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict()},"./model/hol_dC"+".pt")
        saved_loss = average_loss
        best_epoch = []
        best_epoch.append(epoch)
        with open('hol_dC.csv','a',newline='') as file:
            writer = csv.writer(file)
            writer.writerows([best_epoch])    
    elif ((epoch+1) % 500 == 0):
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict()},"./model/hol_dC_temp"+str(epoch)+".pt")
    
    scheduler.step(average_loss)
    info_per_epoch=[]
    info_per_epoch.append(epoch)
    info_per_epoch.append(optimizer.param_groups[-1]['lr'])
    info_per_epoch.append(average_loss)
    with open('loss_hol_dC.csv','a',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([info_per_epoch]) 
  
