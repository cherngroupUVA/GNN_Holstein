import pandas as pd
import numpy as np
import torch

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import math

import time
# from main import Lsize
# from main import neighbor_2d
# #from main import neighbor_ramp
# from main import kT
# from main import num_steps
# from main import par_steps
# from main import Q_data
# from main import velocity_data
# from main import device

import csv




#################parameters###############################################
modelname = 'gnn_6'

dir_out = './ml_simulation/200/0/'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #is this correct?
print(device)
Lsize = 200
lattice_num = Lsize * Lsize
num_site = Lsize * Lsize

kT = 0.1
num_steps = 25001  # total steps of simulation?
par_steps = 100   # time step gap?
Q_data = None
#Q_data = np.loadtxt("./chenchen/data_output/simu/c0.dat", usecols=(0))
velocity_data = None


cutoff = 4
ramp = 6#9#13
edges_per_site = 1 + 4
file_name = "./neighbor/" + str(Lsize) + "_ramp_" + str(ramp) + ".csv"


# a = 1.6169721175977387
# b = 0.03230160447062808



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
       
        edge_index_neighbor[0,i] = neighbor_2d[onsite_index,0]
        edge_index_neighbor[1,i] = neighbor_2d[onsite_index,i % (edges_per_site-1) + 1]
        
    return edge_index_center, edge_index_neighbor
    
#############End of Funciton###########################

################run some functions############################


neighbor_2d, num_input, neighbor_type, sample_neighbor = read_neighbor_list("./neighbor/" + str(Lsize) + "_ramp_" + str(ramp) + ".csv", ramp, Lsize)

print(neighbor_2d.shape)

edge_index_center, edge_index_neighbor = find_edges(neighbor_2d, edges_per_site)

#print(edge_tensor[0, -21:])
#print(edge_tensor[1, -21:])

####################NETWORK#######################################


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
#############################set up model################################################


#torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


net = GCN(edges_per_site)
net.to(device)
model_load = torch.load('./model/' + modelname + '.pt',map_location=torch.device(device) )
net.load_state_dict(model_load['model'])
net = net.eval()

###################################dynamic model#######################################################
class holstein_dynamics:

    def __init__(self, model, neighbor_2d, Q=None, velocity=None, Lsize=40, kT=0.1):
        self.ts = Lsize * Lsize
        self.ls = Lsize
        self.gamma = 0.2
        self.dt = 0.05
        self.kT = kT
        self.kT_rand = 5.
        self.k0 = 1.0
        self.k1 = 0.18
        self.G = 3.5
        self.mass = 5.0
        self.neighbor_2d = neighbor_2d

        self.a_x = math.exp(-self.gamma * self.dt); # 
        self.b_x = math.sqrt( (1. - pow(self.a_x, 2)) / self.mass );

        self.net = model
        
        if Q is not None:
            self.Q = Q
            print("This is Q: ", Q)
        else:
            self.Q = np.random.randn(self.ts) * math.sqrt(self.kT_rand / self.k1)

        if velocity is not None:
            self.velocity = velocity
        else:
            self.velocity = np.random.randn(self.ts) * math.sqrt(self.kT_rand / self.mass)

        self.force = self.calc_force(self.Q)
        
        self.occ = self.calc_occ(self.Q)


    def calc_force(self, Q):
        
        Q_tensor = torch.tensor(Q, dtype=torch.float64).view(-1,1).to(device)
    
        force_prediction = self.net(Q_tensor, edge_index_center, edge_index_neighbor).reshape(-1)
        
        return force_prediction.cpu().detach().numpy()

    def calc_force_classical(self, Q):  # this classical force should be the classiclal elastic force, which is - spring_constant * Q_i - interaction_neighbor_occ * TOTAL_j Q_j
        Q_2d = Q.reshape(self.ls, self.ls)

        up = np.roll(Q_2d, 1, 0)
        down = np.roll(Q_2d, -1, 0)
        left = np.roll(Q_2d, -1, 1)
        right = np.roll(Q_2d, 1, 1)        

        force_classical = -self.k0 * Q_2d  - self.k1 * (up + down + right + left)  # THIS IS THE EPRESSION TO CALCULATE THE ELASTIC FORCE
        force_classical = force_classical.reshape(-1)
        return force_classical


    def calc_occ(self, Q):
        force_classical = self.calc_force_classical(Q)
        occ = (self.force - force_classical) / self.G + 0.5
        return occ        


    def step(self):
        for i in range(self.ts):
            self.Q[i] += self.velocity[i] * self.dt + 0.5 * (self.force[i] / self.mass) * self.dt * self.dt

        force_prev = self.force
        self.force = self.calc_force(self.Q)
        self.occ = self.calc_occ(self.Q)

        for i in range(self.ts):
            self.velocity[i] += 0.5 * self.dt * (self.force[i] + force_prev[i]) / self.mass
        for i in range(self.ts):
            self.velocity[i] = self.a_x * self.velocity[i] + self.b_x * math.sqrt(self.kT) * np.random.randn()



#####################################end of dyanmic model###########################################


lat_sys = holstein_dynamics(net, neighbor_2d, Q_data, velocity_data, Lsize, kT)

start_time = time.time()
for i in range(num_steps+1):
    #print("this is i: ", i)
    if i%par_steps==0:
        np.savetxt(dir_out + "c" + str(i) + ".dat", np.c_[lat_sys.Q, lat_sys.force, lat_sys.occ], delimiter = '\t')
    lat_sys.step()

print("duration time (minutes): ", (time.time() - start_time) / 60.)
