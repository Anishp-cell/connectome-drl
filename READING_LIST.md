# Embodied Connectomics — Reading List & Study Guide

> **Purpose:** This document is the canonical bibliography and self-study syllabus for the
> Embodied Connectomics pipeline. Every team member should read the papers in the order
> presented, completing the study prompts before writing any code in the corresponding module.

---

## How to Use This Document

| Symbol | Meaning |
|--------|---------|
| 🔴 | **Must-read** — foundational; blocks all downstream work |
| 🟡 | **Should-read** — deepens understanding of a specific module |
| 🟢 | **Nice-to-have** — supplementary context or advanced extension |

Papers within each section are ordered by **recommended reading sequence**, not publication date.

---

## 1. Connectomics & the MaleCNS Primary Literature

These papers define the biological ground truth that all connectome-constrained policies are built upon.

### 1.1 🔴 The MaleCNS Whole-CNS Connectome

> **Citation:**
> Januszewski, M., Jain, V., et al. / Cambridge Connectomics / Google Research / HHMI Janelia Research Campus.
> "A complete connectome of the male *Drosophila* central nervous system."
> *Cell* (September 3, 2026).

**Focus Areas:**
- Structure of the intact neck connective bridging the brain and VNC.
- Identification and classification of sexually dimorphic neuron populations.
- Organization of descending neuron (DN) classes: `DNa01`, `DNa02`, `DNg13`, and their VNC targets.
- Synapse detection methodology, proofreading pipeline, and cell type annotation standards.

**Study Prompts:**
1. How many descending neurons cross the neck connective, and what fraction target leg motor neurons vs. wing motor neurons?
2. What is the average synapse count per DN→VNC-interneuron connection? How should we set our pruning threshold?
3. Which DN classes are sexually dimorphic, and could dimorphism modulate locomotion or escape?
4. Describe the data schema exposed via NeuPrint. What fields correspond to synapse weight, cell type, and hemisphere?

---

### 1.2 🔴 The FlyWire Whole-Brain Connectome

> **Citation:**
> Dorkenwald, S., Matsliah, A., Sterling, A. R., Schlegel, P., et al.
> "Neuronal wiring diagram of an adult brain."
> *Nature*, **634**, 124–138 (2024).
> DOI: [10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)

**Focus Areas:**
- Methodological foundation for automated segmentation and proofreading of EM volumes.
- Synapse detection accuracy and false positive/negative rates.
- Cell type annotation conventions and their mapping to NeuPrint type labels.
- Comparison of female (FlyWire) vs. male (MaleCNS) brain-level wiring.

**Study Prompts:**
1. What is the estimated synapse detection recall and precision? How does this error propagate into our adjacency matrix A_ij?
2. Compare the neuron counts and major neuropil boundaries between FlyWire (female) and MaleCNS (male). What are the key dimorphic differences?
3. How are "cell types" defined — by morphology, connectivity, or both? What implications does this have for our graph node labels?

---

### 1.3 🔴 Ventral Nerve Cord Architecture (MANC)

> **Citation (Primary):**
> Marin, E. C., Buld, L., Theiss, M., et al.
> "Connectomics of the *Drosophila* male ventral nerve cord."
> *bioRxiv* (2023); later revised in *eLife*.
>
> **Citation (Companion):**
> Cheong, H. S. J., Boone, K. N., Bennett, M. M., et al.
> "Synaptic architecture of leg and wing premotor control networks in *Drosophila*."
> *bioRxiv* / *eLife* (2023–2024).

**Focus Areas:**
- Identification of leg motor neuron pools (T1–T3 segments) and their premotor interneuron inputs.
- Sensory feedback loops: proprioceptive and mechanosensory afferents in the VNC.
- The hierarchical organization: DNs → VNC interneurons → motor neurons → muscles.

**Study Prompts:**
1. How many motor neurons innervate each leg segment (coxa, trochanter, femur, tibia, tarsus)? Map these to the FlyGym actuator names.
2. Which premotor interneurons are shared across ipsilateral and contralateral leg pairs? What does this imply for gait coupling?
3. Describe the sensory feedback pathways from leg mechanoreceptors back to VNC circuits. How might we model these as observation inputs?

---

### 1.4 🟡 Descending Neuron Function & Behavioral Roles

> **Citation:**
> Namiki, S., Dickinson, M. H., Wong, A. M., Card, G. M., & Korff, W.
> "The functional organization of descending sensory-motor pathways in *Drosophila*."
> *eLife*, **7**, e34272 (2018).
> DOI: [10.7554/eLife.34272](https://doi.org/10.7554/eLife.34272)

**Focus Areas:**
- Functional classification of DN types (DNa, DNb, DNc, DNd, DNg, DNp).
- `DNa01` and `DNa02` roles in steering and turning.
- `DNg13` role in the giant-fiber-mediated escape response.
- Optogenetic activation phenotypes that serve as ground truth for our lesion experiments.

**Study Prompts:**
1. If you silence `DNa01` bilaterally, what behavioral deficit do you expect? Design the corresponding in-silico lesion.
2. What is the latency from visual looming detection to escape jump initiation through the `DNg13` pathway?
3. How do DN population dynamics differ between walking, grooming, and flight? Which DN classes should we activate for each behavior?

---

### 1.5 🟢 The Giant Fiber Escape Circuit

> **Citation:**
> Card, G. M. & Dickinson, M. H.
> "Visually mediated motor planning in the escape response of *Drosophila*."
> *Current Biology*, **18**(17), 1300–1307 (2008).
> DOI: [10.1016/j.cub.2008.07.094](https://doi.org/10.1016/j.cub.2008.07.094)

**Focus Areas:**
- The GF (Giant Fiber) → TTMn/DLMn escape jump circuit as a canonical sensorimotor reflex arc.
- Visual looming detection in the Lobula → LPLC2 → GF pathway.
- How this maps to our Outcome B (Visual Escape Closed Loop).

**Study Prompts:**
1. What is the minimum number of neurons in the GF escape pathway from retinal input to leg extension? Can you trace this in the MaleCNS connectome?
2. How would you construct a looming stimulus in MuJoCo that reliably triggers the escape circuit?

---

## 2. Biomechanics & Simulation Engine

### 2.1 🔴 FlyGym / NeuroMechFly v2

> **Citation:**
> Wang-Chen, S., Stimpfling, V. A., Ozdil, P. G., Marchand-Maillet, L., Ramdya, P., et al.
> "NeuroMechFly v2: Simulating embodied sensorimotor control in adult *Drosophila*."
> *Nature Methods* (2024) / *bioRxiv* preprint.
> GitHub: [https://github.com/NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym)

**Focus Areas:**
- MuJoCo-based biomechanical model: joint limits, mass properties, adhesion actuators.
- The Gymnasium-compatible API: `observation_space`, `action_space`, `step()`, `reset()`.
- Preprogrammed CPG-based gaits (tripod, tetrapod, wave) as behavioral baselines.
- Ground reaction force (GRF) sensing and tarsal contact detection.

**Study Prompts:**
1. How many degrees of freedom (DOFs) does the FlyGym model expose per leg? List each joint name and its range of motion.
2. What is the default simulation timestep, and how does it relate to the control timestep? What is the expected sim-to-real time ratio on a single CPU core?
3. Describe the adhesion actuator model. Under what conditions does tarsal adhesion engage/disengage?
4. How does the observation space encode proprioceptive vs. exteroceptive information?

---

### 2.2 🟡 NeuroMechFly v1 (Original Biomechanical Model)

> **Citation:**
> Lobato-Rios, V., Ramalingasetty, S. T., Ozdil, P. G., Arreguit, J., Ijspeert, A. J., & Ramdya, P.
> "NeuroMechFly, a neuromechanical model of adult *Drosophila melanogaster*."
> *Nature Methods*, **19**, 620–627 (2022).
> DOI: [10.1038/s41592-022-01466-7](https://doi.org/10.1038/s41592-022-01466-7)

**Focus Areas:**
- Original morphological reconstruction from micro-CT data.
- Validation of joint torques against experimental measurements.
- Comparison of v1 (Bullet-based) vs. v2 (MuJoCo-based) dynamics.

---

### 2.3 🔴 MuJoCo: Physics Engine Foundations

> **Citation:**
> Todorov, E., Erez, T., & Tassa, Y.
> "MuJoCo: A physics engine for model-based control."
> *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 5026–5033 (2012).
> DOI: [10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109)

**Focus Areas:**
- Generalized coordinate representation and minimal coordinate dynamics.
- Constraint formulation: contact, friction cones, joint limits.
- Soft constraint solvers and their implications for biomechanical accuracy.
- Actuator models: position, velocity, and torque-controlled joints.

**Study Prompts:**
1. Explain the difference between `position`, `velocity`, and `motor` actuators in MuJoCo. Which does FlyGym use?
2. How does MuJoCo's contact model handle simultaneous multi-point contact (e.g., tripod stance)?
3. What is the computational cost model for `mj_step()`? How does it scale with the number of contacts?

---

## 3. Reinforcement Learning & Motor Control

### 3.1 🔴 Proximal Policy Optimization (PPO)

> **Citation:**
> Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.
> "Proximal Policy Optimization Algorithms."
> *arXiv preprint arXiv:1707.06347* (2017).

**Focus Areas:**
- The clipped surrogate objective and its relationship to trust region methods.
- Generalized Advantage Estimation (GAE-lambda) and bias-variance tradeoff.
- Implementation details for continuous action spaces: diagonal Gaussian policies.
- Hyperparameter sensitivity: clip ratio epsilon, learning rate schedule, number of epochs per update.

**Study Prompts:**
1. Derive the clipped surrogate loss. Why does clipping prevent catastrophic policy updates?
2. Write pseudocode for the GAE-lambda advantage computation. What values of lambda and gamma are appropriate for locomotion tasks with dense rewards?
3. For a 42-dimensional continuous action space (fly joint targets), what are the practical considerations for the policy's output distribution?

---

### 3.2 🔴 DEP-RL: Embodied Exploration for Musculoskeletal Systems

> **Citation:**
> Yoon, D., Choi, H., & Lee, J. (primary); building on Ruckert, E. & d'Avella, A.
> "DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems."
> *International Conference on Learning Representations (ICLR)* (2023).

**Focus Areas:**
- Why standard Gaussian exploration (epsilon-greedy or OU noise) fails catastrophically in high-DOF musculoskeletal bodies.
- The DEP (Differential Extrinsic Plasticity) controller: local anti-Hebbian sensorimotor learning that self-organizes coordinated joint movements.
- Integration of DEP-generated exploration with PPO reward optimization.
- The exploration-exploitation phase transition.

**Study Prompts:**
1. Explain the DEP update rule: C_{t+1} = C_t + epsilon (x_{t+1} * y_t^T - C_t). What does the matrix C represent?
2. Why does DEP naturally produce coordinated multi-joint exploration instead of independent joint noise?
3. Design a training curriculum: How many environment steps should use DEP exploration before transitioning to PPO?

---

### 3.3 🟡 Generalized Advantage Estimation (GAE)

> **Citation:**
> Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P.
> "High-Dimensional Continuous Control Using Generalized Advantage Estimation."
> *International Conference on Learning Representations (ICLR)* (2016).
> *arXiv preprint arXiv:1506.02438*.

**Focus Areas:**
- Formal derivation of GAE(gamma, lambda).
- Understanding how lambda interpolates between Monte Carlo and TD(0) advantage estimates.
- Practical guidelines for tuning gamma and lambda in locomotion tasks.

---

### 3.4 🟡 Actor-Critic Methods for Continuous Control

> **Citation:**
> Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D.
> "Continuous control with deep reinforcement learning." (DDPG)
> *International Conference on Learning Representations (ICLR)* (2016).
> *arXiv preprint arXiv:1509.02971*.

**Focus Areas:**
- Deterministic policy gradient theorem for continuous action spaces.
- Experience replay and target networks (contrast with PPO's on-policy approach).
- Batch normalization in actor-critic architectures.

**Study Prompts:**
1. Compare PPO (stochastic, on-policy) with DDPG (deterministic, off-policy) for the fly locomotion task. What are the tradeoffs?
2. Why might on-policy PPO be preferred for biomechanical simulation where reward shaping changes frequently?

---

## 4. Graph Neural Networks & Sparse Neural Architectures

### 4.1 🔴 Message Passing Neural Networks (MPNNs)

> **Citation:**
> Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E.
> "Neural Message Passing for Quantum Chemistry."
> *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 1263–1272 (2017).

**Focus Areas:**
- The message-passing framework: message, aggregate, update functions.
- How connectome adjacency naturally defines the message-passing topology.
- Relationship between GNN depth and the receptive field in the connectome graph.

**Study Prompts:**
1. If the connectome subgraph has diameter d, how many GNN layers are needed for information to propagate from sensory inputs to motor outputs?
2. How should edge weights (synapse counts) be incorporated into the message function?

---

### 4.2 🟡 PyTorch Geometric (PyG) Framework

> **Citation / Reference:**
> Fey, M. & Lenssen, J. E.
> "Fast Graph Representation Learning with PyTorch Geometric."
> *ICLR Workshop on Representation Learning on Graphs and Manifolds* (2019).
> Documentation: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)

**Focus Areas:**
- `torch_geometric.data.Data` and `torch_geometric.data.Batch` for batched graph processing.
- Built-in `GCNConv`, `GATConv`, `MessagePassing` base classes.
- Sparse adjacency representation via `edge_index` tensors.
- Integration with standard PyTorch training loops.

---

### 4.3 🟡 Sparse Networks & Lottery Ticket Hypothesis

> **Citation:**
> Frankle, J. & Carlin, M.
> "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks."
> *International Conference on Learning Representations (ICLR)* (2019).

**Focus Areas:**
- The hypothesis that sparse subnetworks can match dense network performance.
- Relationship to our approach: the connectome provides a biologically-discovered "lottery ticket."
- Pruning strategies and their parallels to synapse count thresholding.

**Study Prompts:**
1. Is the biological connectome a "winning ticket"? How would you test this hypothesis by comparing a random sparse mask of the same density to the connectome mask?
2. What is the expected sparsity ratio of our DN->VNC subgraph, and how does it compare to lottery ticket sparsity levels?

---

## 5. Insect Neuroscience & Locomotion

### 5.1 🟡 Central Pattern Generators in Insect Locomotion

> **Citation:**
> Bidaye, S. S., Machacek, C., Wu, Y., & Dickson, B. J.
> "Neuronal control of *Drosophila* walking direction."
> *Science*, **344**(6179), 97–101 (2014).
> DOI: [10.1126/science.1249964](https://doi.org/10.1126/science.1249964)

**Focus Areas:**
- Moonwalker descending neurons (MDNs) and their role in backward walking.
- Descending neuron control of gait initiation, speed, and direction.
- How CPG-level gait generation is modulated by descending commands.

---

### 5.2 🟡 Drosophila Leg Coordination & Gait Patterns

> **Citation:**
> DeAngelis, B. D., Zavatone-Veth, J. A., & Clark, D. A.
> "The manifold structure of limb coordination in walking *Drosophila*."
> *eLife*, **8**, e46409 (2019).
> DOI: [10.7554/eLife.46409](https://doi.org/10.7554/eLife.46409)

**Focus Areas:**
- Low-dimensional manifold structure of inter-leg coordination.
- Quantitative gait metrics: duty factor, phase offsets, stride frequency.
- Use as ground truth for evaluating learned gaits.

**Study Prompts:**
1. What gait phase relationships define a tripod gait? How would you compute a "tripod coordination index" from simulation data?
2. At what walking speed does the fly transition from wave gait to tripod gait? Can our RL agent discover this transition?

---

### 5.3 🟢 Proprioceptive Feedback in Drosophila Locomotion

> **Citation:**
> Mamiya, A., Gurung, P., & Tuthill, J. C.
> "Neural coding of leg proprioception in *Drosophila*."
> *Neuron*, **100**(3), 636–650 (2018).
> DOI: [10.1016/j.neuron.2018.09.009](https://doi.org/10.1016/j.neuron.2018.09.009)

**Focus Areas:**
- Femoral chordotonal organ: encoding of joint angle, velocity, and acceleration.
- Club, hook, and claw sensory neuron subtypes.
- Implications for designing observation inputs in FlyGym.

---

## 6. Computational Neuroscience Methods

### 6.1 🟡 Connectome-Constrained Neural Network Modeling

> **Citation:**
> Shiu, P. K., Sterne, G. R., Spiller, N., et al.
> "A *Drosophila* computational brain model reveals sensorimotor processing."
> *Nature* (2024).

**Focus Areas:**
- Using connectome weights to initialize or constrain neural network models.
- Functional validation through behavioral predictions.
- Relationship between structural connectivity and functional dynamics.

**Study Prompts:**
1. How do the authors handle the mismatch between synapse counts (structural) and synaptic efficacy (functional)?
2. What normalization strategies are applied to the raw adjacency matrix before use as network weights?

---

### 6.2 🟢 Whole-Brain Calcium Imaging & Neural Dynamics

> **Citation:**
> Aimon, S., Katsuki, T., Jia, T., Grosenick, L., Broxton, M., Deisseroth, K., Sejnowski, T. J., & Greenspan, R. J.
> "Fast near-whole-brain imaging in adult *Drosophila* during responses to stimuli and behavior."
> *PLOS Biology*, **17**(2), e2006732 (2019).

**Focus Areas:**
- Functional neural dynamics that complement structural connectomics.
- Stimulus-evoked activity patterns as validation targets for simulation.

---

## 7. Software & API References

| Resource | URL | Purpose |
|----------|-----|---------|
| NeuPrint Python Client | [neuprint-python](https://github.com/connectome-neuprint/neuprint-python) | Querying MaleCNS connectome data |
| NeuPrint Explorer (Web) | [neuprint.janelia.org](https://neuprint.janelia.org) | Interactive connectome exploration |
| FlyGym Repository | [flygym](https://github.com/NeLy-EPFL/flygym) | Biomechanical simulation framework |
| FlyGym Documentation | [flygym.github.io](https://flygym.github.io/flygym/) | API docs for FlyGym/NeuroMechFly v2 |
| PyTorch Geometric | [pytorch-geometric](https://pytorch-geometric.readthedocs.io/) | GNN layers and sparse operations |
| MuJoCo Documentation | [mujoco](https://mujoco.readthedocs.io/) | Physics engine API and MJCF reference |
| Codex (MaleCNS) | [codex.flywire.ai](https://codex.flywire.ai) | Cell type annotations browser |

---

## Recommended Reading Order

The dependency graph below shows the optimal reading sequence. Papers connected by arrows should be read in parent-first order.

**Track A — Simulation & RL:**
1. MuJoCo (Todorov 2012)
2. FlyGym / NMF v2 (Wang-Chen 2024)
3. PPO (Schulman 2017)
4. GAE (Schulman 2016)
5. DEP-RL (Yoon 2023)

**Track B — Connectomics:**
6. FlyWire Connectome (Dorkenwald 2024)
7. MaleCNS (Januszewski 2026)
8. MANC / VNC (Marin & Cheong 2023)
9. DN Function (Namiki 2018)
10. Escape Circuit (Card 2008)

**Track C — Architectures:**
11. MPNN (Gilmer 2017)
12. PyG (Fey 2019)
13. Lottery Ticket (Frankle 2019)

**Track D — Biology & Validation:**
14. Insect Locomotion (Bidaye 2014, DeAngelis 2019)
15. Connectome Models (Shiu 2024)

### Suggested Weekly Plan

| Week | Papers | Module Work |
|------|--------|-------------|
| **Week 1** | MuJoCo, FlyGym/NMF v2, PPO | Phase 0: Scaffolding, FlyGym verification |
| **Week 2** | FlyWire, MaleCNS, MANC/VNC | Phase 1: Connectome extraction |
| **Week 3** | DN Function, MPNN, PyG, Lottery Ticket | Phase 2: Policy architecture |
| **Week 4** | GAE, DEP-RL | Phase 3-4: FlyGym integration & training |
| **Week 5** | Escape Circuit, Insect Locomotion, Connectome Models | Phase 5: Lesion experiments |

---

## Appendix: Key Equations to Internalize

### PPO Clipped Objective
```
L_CLIP(theta) = E_t [ min( r_t(theta) * A_t,  clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]
```

### Generalized Advantage Estimation
```
A_t^GAE(gamma,lambda) = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

### DEP Controller Update
```
C_{t+1} = C_t + epsilon * ( x_{t+1} * y_t^T - C_t )
```

### Masked Linear Layer
```
Y = (W ⊙ M) X + b,    M_ij in {0, 1}
```

### Locomotion Reward
```
R_t = v_forward - c_energy * sum_j ||tau_j||^2 - c_roll * theta_roll^2
```

---

*Last updated: 2026-09-14*
*Maintainer: Embodied Connectomics Research Team*
