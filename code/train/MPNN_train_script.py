import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import csv

import numpy as np
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Lsize = 40
dir_train = './training_data/'
cutoff = 4#4#5
ramp = 9#9#13
input_size = 49
edges_per_site = 1 + 4
file_name = "./neighbor/" + str(Lsize) + "_ramp_" + str(ramp) + ".csv"
from_bench_mark = False
epoch_start = 0
duration = 10000

##############Functions###################################################
def read_neighbor_list(file_name, ramp, Lsize):
    num_site = Lsize * Lsize

    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        neighbor_list = list(reader)
        
    sample_neighbor = [] 
    for i in range(0,len(neighbor_list)):
        each_layer = []
        if (len(neighbor_list[i]) == 1 and int(neighbor_list[i][0]) == 1):
            break
        for j in range(0,len(neighbor_list[i])):
            lattice_index = int(neighbor_list[i][j])
            x = lattice_index % Lsize
            y = lattice_index // Lsize
            if (x > cutoff):
                x = x - Lsize
            if (y > cutoff):
                y = y - Lsize
            each_layer.append([lattice_index,x,y])
        sample_neighbor.append(each_layer)
    
    #print("This is sample neighbor: ", sample_neighbor)
    
    
    neighbor_list_length = torch.zeros([ramp + 1], dtype=torch.int32)
    sum  = 0
    for i in range(0, ramp+1):
        neighbor_list_length[i] = len(neighbor_list[i])
        sum += len(neighbor_list[i])
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

    neighbor_type = torch.zeros(ramp + 1, dtype=torch.int).to(device)

    for i in range(1, ramp + 1):
        if neighbor_list_length[i] == 4:
            if neighbor_list[0][0] // Lsize == neighbor_list[i][0] // Lsize or neighbor_list[0][0] % Lsize == neighbor_list[i][0] % Lsize: # it can be used for any lsize!
                
                neighbor_type[i] = 1
            else:
                neighbor_type[i] = 2
                
        if neighbor_list_length[i] == 8:
            neighbor_type[i] = 3

    return neighbor_2d, tot_length, neighbor_type, sample_neighbor

def find_edges(neighbor_2d, edges_per_site):
    edge_index_center = torch.zeros((2,Lsize * Lsize),dtype = int).to(device)
    edge_index_neighbor = torch.zeros((2,Lsize * Lsize * (edges_per_site -1)),dtype = int).to(device)
    
    print("this is edge_index_center: ", edge_index_center.shape)
    print("this is edge_index_neighbor: ", edge_index_neighbor.shape)
    onsite_index = 0
    for i in range (0, Lsize * Lsize):
        edge_index_center[0,i] = i
        edge_index_center[1,i] = i
        
        
    for i in range (0, Lsize * Lsize * (edges_per_site -1)):
        if i % (edges_per_site -1) == 0 and i != 0:
            onsite_index = onsite_index + 1
       
        edge_index_neighbor[1,i] = neighbor_2d[onsite_index,0]
        edge_index_neighbor[0,i] = neighbor_2d[onsite_index,i % (edges_per_site-1) + 1]
        
    return edge_index_center, edge_index_neighbor
    
#############End of Funciton###########################

##################Read Data########################################
neighbor_2d, num_input, neighbor_type, sample_neighbor = read_neighbor_list("./neighbor/" + str(Lsize) + "_ramp_" + str(ramp) + ".csv", ramp, Lsize)

print(neighbor_2d.shape)

edge_index_center, edge_index_neighbor = find_edges(neighbor_2d, edges_per_site) # you need to turn self_loop off!!!!



print(edge_index_neighbor[0, -20:])
print(edge_index_neighbor[1, -20:])

#print(edge_weight[-21:])





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



#######End of Read data###################################################
#There is one shared weight matrix W of shape [in_channels,out_channels]
#Every node, and every neighbor during aggregation, uses the same  W.
#number of nodes: same across all the layers generally
#number of features at each node: change according to int_channel and out_channel
#The graph structure (nodes + edges) is fixed (unless you explicitly change it)

class MPNNLayer_node_neighbor(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None):
        super().__init__(aggr='mean')  # can be 'add' or 'max'
        input_dim = in_channels + (edge_dim or 0)
        self.mlp_msg = Sequential(
            Linear(input_dim, out_channels),
            ReLU(),
            Linear(out_channels, out_channels, bias=False)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
       
        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        if edge_attr is not None:
            msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([x_j], dim=-1)
        return self.mlp_msg(msg_input)
    

    
class MPNNLayer_self_loop_node2(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None):
        super().__init__(aggr='mean')  # can be 'add' or 'max'
        input_dim = in_channels + (edge_dim or 0)
        self.mlp_msg = Sequential(
            Linear(input_dim, out_channels),
            ReLU(),
            Linear(out_channels, out_channels, bias=False)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        if edge_attr is not None:
            msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([x_j], dim=-1)
        return self.mlp_msg(msg_input)
    
    
    
    
class MPNN(torch.nn.Module):
    def __init__(self, edge_dim=None):
        super().__init__()
        
        self.mlp_update1 = Sequential(
            Linear(1, 512),
            ReLU()
        )
        self.norm1  = nn.LayerNorm(512)
        
        self.mlp_update2 = Sequential(
            Linear(512, 1024),
            ReLU()
        )
        self.norm2  = nn.LayerNorm(1024)
        
        self.mlp_update3 = Sequential(
            Linear(1024, 512),
            ReLU()
        )
        self.norm3  = nn.LayerNorm(512)
        
    
        
        self.res_proj1 = nn.Linear(1, 512)
        
        self.res_proj2 = nn.Linear(512, 1024)
        
        self.res_proj3 = nn.Linear(1024, 512)
        
        self.res_proj6 = nn.Linear(512, 1)
        
    
       
        self.layer1b = MPNNLayer_node_neighbor(1, 512, edge_dim=edge_dim)
        self.layer1d = MPNNLayer_self_loop_node2(1, 512, edge_dim=edge_dim)
        
        
        
        self.layer2b = MPNNLayer_node_neighbor(512, 1024, edge_dim=edge_dim)
        self.layer2d = MPNNLayer_self_loop_node2(512, 1024, edge_dim=edge_dim)
        
        
        
        self.layer3b = MPNNLayer_node_neighbor(1024, 512, edge_dim=edge_dim)
        self.layer3d = MPNNLayer_self_loop_node2(1024, 512, edge_dim=edge_dim)
        
        
        
        self.layer6b = MPNNLayer_node_neighbor(512, 1, edge_dim=edge_dim)
        self.layer6d = MPNNLayer_node_neighbor(512, 1, edge_dim=edge_dim)
        
        
        
    def forward(self, x, edge_index,edge_index_self_loop, edge_attr=None):
        
        
        
        x2 = self.layer1b(x, edge_index, edge_attr)
        x4 = self.layer1d(x, edge_index_self_loop, edge_attr)
        u = x2  + x4 + self.mlp_update1(x)
        u = self.norm1(u)
        res = self.res_proj1(x)                  
        x = res + u 
        
        
        x2 = self.layer2b(x, edge_index, edge_attr)
        x4 = self.layer2d(x, edge_index_self_loop, edge_attr)
        u = x2  + x4 + self.mlp_update2(x)
        u = self.norm2(u)
        res = self.res_proj2(x)                  
        x = res + u 
        
        
        
        x2 = self.layer3b(x, edge_index, edge_attr)
        x4 = self.layer3d(x, edge_index_self_loop, edge_attr)
        u = x2  + x4 + self.mlp_update3(x)
        u = self.norm3(u)
        res = self.res_proj3(x)                  
        x = res + u 
        
        
      
        x2 = self.layer6b(x, edge_index, edge_attr)
        x4 = self.layer6d(x, edge_index_self_loop, edge_attr)
        u = x2  + x4
        res = self.res_proj6(x)                  
        x = res + u 
        
        
        return x
###################Train Begins#####################################

net = MPNN()
net.to(device)


#print(list(net.parameters()))


optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
loss_func = torch.nn.MSELoss()
if from_bench_mark:
    print("load ./model/model_status.pt")
    prev_model_data = torch.load("./model/1.pt")  
    net.load_state_dict(prev_model_data['model'])
    optimizer.load_state_dict(prev_model_data['optimizer'])
    
    
saved_loss = 1000
for epoch in range(epoch_start,epoch_start+duration):  #epoch_start,epoch_start+duration
    loss_perstep = []
    for step, (Q, force) in enumerate(loader):
        
        temp_coordinate = Q[0].requires_grad_(True)
        temp_force = force[0].requires_grad_(False)
        
        #print("temp_coordinate: ", temp_coordinate.shape)
        
       
       
        optimizer.zero_grad()
        
        energy_prediction_temp = net(temp_coordinate.reshape(-1,1),edge_index_neighbor, edge_index_center).view(-1)
        
        #print("this is the shape of energy_prediction_temp:", energy_prediction_temp.shape)
        
        energy_prediction = torch.sum(energy_prediction_temp)
        force_prediction = -torch.autograd.grad(energy_prediction, temp_coordinate, create_graph=True)[0]
        
        
        #print("shape of force prediction: ", force_prediction.shape)
        
        
        loss = 1000.0 * loss_func(force_prediction, temp_force)
        
        loss_perstep.append(loss.item())
        loss.backward()
        optimizer.step()
        #print("This is loss: ", loss.item())


    average_loss = sum(loss_perstep) / len(loss_perstep)
    if (average_loss < saved_loss ):                                   #((epoch+1)%100)==0: or (epoch+1) % 1000 == 0
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict()},"./model/mpnn_bp10b"+".pt")
        saved_loss = average_loss
        best_epoch = []
        best_epoch.append(epoch)
        with open('mpnn_bp10b.csv','a',newline='') as file:
            writer = csv.writer(file)
            writer.writerows([best_epoch])    
    elif ((epoch+1) % 50 == 0):
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict()},"./model/mpnn_bp10b_temp"+str(epoch)+".pt")
    
    scheduler.step(average_loss)
    info_per_epoch=[]
    info_per_epoch.append(epoch)
    info_per_epoch.append(optimizer.param_groups[-1]['lr'])
    info_per_epoch.append(average_loss)
    with open('loss_mpnn_bp10b.csv','a',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([info_per_epoch]) 

