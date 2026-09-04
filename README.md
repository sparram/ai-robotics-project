# Safe Autonomous Driving : Comparison of MPC-CBF and RL controllers

**Author:** Santiago Parra  
**Year:** 2026

This project develops a MPC controller combined with a Control Barrier Function (CBF), and compares it with a RL based controller for autonomous driving with Metadrive.

<img src="src/media/mpc_cbf_demo.gif" width="600" alt="MPC-CBF Demo">

## MPC-CBF Controller : Formulation of the problem

The MPC controller is a classic controller an intuitive perspective for control driving. Given a time horizon T, the idea is to solve the following optimization problem for every time step of the simulation:

$\min J(u) = \sum_{k=1}^N \omega_P J_{pos}(x_k) + \omega_S J_{speed}(x_k) + \omega_U J_{control}(u_k)$

Where $x_k$ represents the state of the vehicle and $u_k$ represents the sequence of actions / controls that guide the vehicle. The weights $\omega_P$, $\omega_S$ and $\omega_U$ penalize the position, the speed and the control respectively.

This cost function represents the desired behaviour of the vehicle (go straight in the lane, go at a reasonable speed, don't do sudden moves, etc).
The optimization should be subject to the following constraints:

- The dynamics of the system: Represents the reaction of the vehicle when we choose an action $$x_{k+1} = f(x_k, u_k)$$
- The Control Barrier Function (CBF): We create a barrier around each obstacle that the vehicle cannot cross, we represent this behaviour by a positive barrier function $h(x) \geq 0$. In order to guarantee the positivity, when solving the optimization problem we impose:
  $$h(x_{k+1}) \ge (1 - \gamma) h(x_k)$$
- Limits of the control: As the control represents a real life action, it should be bounded (for example, the steer cannot exceed 45° or the acceleration has a maximum limit). We represent this by a normalized constraint for the controls
  $$-1 \leq u_k \leq 1 \hspace{10pt}$$

Which are taken for each $k = 1, 2, \dots N$ i.e for each instant of the time horizon.

In our implementation, we perform a LTV approximation in order to transform the optimization problem into an Quadratic Programming (QP) problem. In the end we solve a system of the form:

$$\min \frac{1}{2} U^T P U + q^T U$$ 

Subject to a linear constraint $l \leq A U \leq u$ that encapsulates the CBF constraint and the physical limit of the control.

## RL Controller

- In addition to the MPC-CBF, we trained a **Proximal Policy Optimization (PPO)** controller using the `stable-baselines3` package with the Metadrive environment. 
- We trained the model using a MLP Policy over 50 generated scenarios with traffic density of 0.15 over 250000 timesteps. 
- The MLP receives the Metadrive observations (state of the vehicle + LiDAR measurements) and returns the chosen control (steer, acceleration) for the case.
- The cost function for the learning process is the standard PPO cost function. We provide to PPO the  default reward function coming in `MetaDriveEnv`, which is defined as:

$$ R = c_1 R_{driving} + c_2 R_{speed} + R_{terminate}$$

Where $R_{driving}$ and $R_{speed}$ motivate the desired behaviour of position and speed of the vehicle respect to reference target values, respectively (similar to $J_{pos}$, $J_{speed}$ in the MPC-CBF formulation). $R_{terminate}$ contains a set of rewards: the success of the episode, penalization to crashes, etc.

We consider here the default weight configuration ($c_1$, $c_2$, etc) which can be consulted in the Metadrive docs.

**Note:** Here the crash constraint is given as a penalty on the reward function. This is called a soft-formulation, compared to the explicit constraint formulation given in the MPC-CBF.

## Experiments

For the experiments, we considered a $N = 15$, $\Delta t = 0.1$, a $\gamma = 0.2$ and the Kinematic Bycicle model, where the control is given by the acceleration and steer of the vehicle.