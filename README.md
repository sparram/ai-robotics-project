# Safe Autonomous Driving : Comparison of MPC-CBF and RL controllers

**Author:** Santiago Parra  
**Year:** 2026

This project develops a MPC controller combined with a Control Barrier Function (CBF), and compares it with a RL based controller for autonomous driving with Metadrive.

https://github.com/user-attachments/assets/074b175a-760f-4ef3-8d26-a1c635e1ba4f

## Formulation of the problem

The MPC controller is a classic controller an intuitive perspective for control driving. Given a time horizon T, the idea is to solve the following optimization problem for every time step of the simulation:

$\int_0^T $