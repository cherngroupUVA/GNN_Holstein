import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import csv

import numpy as np
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Lsize = 40
dir_train = './training_data/'
cutoff = 4#4#5
ramp = 9#9#13
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



print(edge_index_neighbor[0, 0:16])
print(edge_index_neighbor[1, 0:16])

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

class GCN(torch.nn.Module):
    def __init__(self, edges_per_site):
        super().__init__()
        self.conv1 = GCNConv(1, 128 , add_self_loops = False, normalize = True, bias =True)
        self.conv1_2 = GCNConv(1, 128 , add_self_loops = False, normalize = True, bias =False)
        
        self.conv2 = GCNConv(128, 512, add_self_loops = False, normalize = True, bias =True)
        self.conv2_2 = GCNConv(128, 512, add_self_loops = False, normalize = True, bias =False)
        
        self.conv3 = GCNConv(512, 1024, add_self_loops = False, normalize = True, bias =True)
        self.conv3_2 = GCNConv(512, 1024, add_self_loops = False, normalize = True, bias =False)
        
        self.conv4 = GCNConv(1024, 2048, add_self_loops = False, normalize = True, bias =True)
        self.conv4_2 = GCNConv(1024, 2048, add_self_loops = False, normalize = True, bias =False)
        
        self.conv5 = GCNConv(2048, 1024, add_self_loops = False, normalize = True, bias =True)
        self.conv5_2 = GCNConv(2048, 1024, add_self_loops = False, normalize = True, bias =False)
        
        self.conv6 = GCNConv(1024, 512, add_self_loops = False, normalize = True, bias =True)
        self.conv6_2 = GCNConv(1024, 512, add_self_loops = False, normalize = True, bias =False)
        
        self.conv7 = GCNConv(512, 128, add_self_loops = False, normalize = True, bias =True)
        self.conv7_2 = GCNConv(512, 128, add_self_loops = False, normalize = True, bias =False)
        
        self.output = GCNConv(128, 1, add_self_loops = False, normalize = True, bias =True)
        self.output_2 = GCNConv(128, 1, add_self_loops = False, normalize = True, bias =False)
        
        
        
        

    def forward(self, x, edge_index_center, edge_index_neighbor):
        # Layer 1: GCN + ReLU
        
        
        #print("this is edge_weight: ", edge_weight[0:20])
            

        x1 = self.conv1(x, edge_index_center)
        x2 = self.conv1_2(x, edge_index_neighbor)
        x = torch.relu(x1 + x2)

        x1 = self.conv2(x, edge_index_center)
        x2 = self.conv2_2(x, edge_index_neighbor)
        x = torch.relu(x1 + x2)
        
        x1 = self.conv3(x, edge_index_center)
        x2 = self.conv3_2(x, edge_index_neighbor)
        x = torch.relu(x1 + x2)
        
        x1 = self.conv4(x, edge_index_center)
        x2 = self.conv4_2(x, edge_index_neighbor)
        x = torch.relu(x1 + x2)
        
        x1 = self.conv5(x, edge_index_center)
        x2 = self.conv5_2(x, edge_index_neighbor)
        x = torch.relu(x1 + x2)
        
        x1 = self.conv6(x, edge_index_center)
        x2 = self.conv6_2(x, edge_index_neighbor)
        x = torch.relu(x1 + x2)
        
        x1 = self.conv7(x, edge_index_center)
        x2 = self.conv7_2(x, edge_index_neighbor)
        x = torch.relu(x1 + x2)
       
        
        x1 = self.output(x, edge_index_center)
        x2 = self.output_2(x, edge_index_neighbor)
        x = x1 + x2

        return x
    
    
###################Train Begins#####################################

net = GCN(edges_per_site)
net.to(device)

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")



#print(list(net.parameters()))


optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
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
        
        temp_coordinate = Q[0].requires_grad_(False)
        temp_force = force[0].requires_grad_(False)
        
        #print(temp_coordinate.shape)
        
        #print(x_temp)
       
        optimizer.zero_grad()
        
        force_prediction = net(temp_coordinate.reshape(-1,1), edge_index_center, edge_index_neighbor).view(-1)
        
        #print("shape of force: ", force_prediction.shape)
        
        
        loss = loss_func(force_prediction, temp_force)
        
        loss_perstep.append(loss.item())
        loss.backward()
        optimizer.step()
        print("This is loss: ", loss.item())
 
    average_loss = sum(loss_perstep) / len(loss_perstep)
    if (average_loss < saved_loss ):                                   #((epoch+1)%100)==0: or (epoch+1) % 1000 == 0
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict()},"./model/gnn_6"+".pt")
        saved_loss = average_loss
        best_epoch = []
        best_epoch.append(epoch)
        with open('gnn_6.csv','a',newline='') as file:
            writer = csv.writer(file)
            writer.writerows([best_epoch])    
    elif ((epoch+1) % 50 == 0):
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict()},"./model/gnn_6_temp"+str(epoch)+".pt")
    
    scheduler.step(average_loss)
    info_per_epoch=[]
    info_per_epoch.append(epoch)
    info_per_epoch.append(optimizer.param_groups[-1]['lr'])
    info_per_epoch.append(average_loss)
    with open('loss_gnn_6.csv','a',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([info_per_epoch]) 

