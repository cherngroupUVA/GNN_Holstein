# Repository Structure and Data Description

This repository contains the code and data used in the paper:

“Graph neural network force fields for adiabatic dynamics of lattice Hamiltonians”

The project develops machine-learning surrogate force fields for the adiabatic dynamics of lattice Hamiltonians, demonstrated using the semiclassical Holstein model on a square lattice.

Three neural-network architectures are implemented:

Descriptor-based Fully Connected Network (FC)
A multilayer perceptron that incorporates the D4 lattice point-group symmetry through symmetry-adapted descriptors of the local environment.

Graph Neural Network (GNN)
A graph convolutional network (GCN) operating directly on the lattice graph. Symmetry preservation is achieved through local message passing and parameter sharing.

Energy-based Message-Passing Neural Network (MPNN)
A Behler–Parrinello–type framework in which the network predicts local energies. Forces are then obtained from the gradient of the total energy.

The FC and GNN models directly predict local forces, while the MPNN predicts local energies, from which forces are computed.

-------------------------------------------------
Training Data
-------------------------------------------------
The training_data zip file contains four .pt files:

Q_random_200.pt
200 random initial lattice configurations

force_random_200.pt
Force snapshots corresponding to Q_random_200.pt

Q_random_400.pt
400 lattice snapshots sampled from the post-quench dynamical evolution

force_quench_400.pt
Force snapshots corresponding to Q_random_400.pt

All data are defined on a 40 × 40 square lattice.

-------------------------------------------------
Code
-------------------------------------------------

The code directory contains two subfolders:

train/ — Training the ML Model

This directory contains the scripts used to train each neural-network model.

FC_train_script.py
Training script for the descriptor-based fully connected network.

GNN_train_script.py
Training script for the graph neural network (GNN) model.

MPNN_train_script.py
Training script for the energy-based message-passing neural network (MPNN).


dynamic/ — Dynamical Simulation

This directory contains scripts used to run Langevin dynamics simulations using the trained machine-learning force fields.

simulation_script.py
Main script for performing large-scale dynamical simulations on a 200 × 200 lattice.

-------------------------------------------------
Neighbor Information
-------------------------------------------------

These files define the lattice connectivity used by the machine-learning models to construct local environments and perform message passing.
