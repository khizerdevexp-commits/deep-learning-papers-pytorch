# Physics-Informed ML: Papers and Roadmap

A practical roadmap for building industry-ready AI-for-science skills through Physics-Informed Neural Networks (PINNs), failure-mode analysis, neural operators, and publication-quality project work.

This repository is organized around a clear progression from foundational PINN implementations to operator learning and real-world scientific ML research.

---

## Overview

This roadmap is designed for a hands-on, reproducible, and publication-oriented learning path in physics-informed machine learning. The goal is not just to read papers, but to implement them from scratch, compare against classical baselines, and produce work with clear evidence and measurable results.

Each phase ends with something you can publish, because public, verifiable work is what gets you hired. Verify paper titles and details against arXiv or Google Scholar as you go.

---

## Part 1: Papers List (in implementation order)

### Phase 1: Core PINNs

1. Raissi, Perdikaris, Karniadakis (2019). Physics-informed neural networks.
   - The original paper.
   - Reproduce Burgers' equation (forward) and an inverse problem.

2. Lu et al. (2021). DeepXDE.
   - Read the design, then compare your code against it.

3. Baydin et al. (2018). Automatic differentiation in machine learning: a survey.
   - Know how the higher-order derivatives in your loss actually work.

### Phase 2: Why PINNs Fail, and Fixes

4. Krishnapriyan et al. (2021). Characterizing possible failure modes in physics-informed neural networks.
5. Wang, Teng, Perdikaris (2021). Understanding and mitigating gradient flow pathologies in PINNs.
6. Wang, Yu, Perdikaris (2022). When and why PINNs fail to train: a neural tangent kernel perspective.
7. Tancik et al. (2020). Fourier features let networks learn high-frequency functions in low dimensional domains.
8. Wang et al. (2023). An expert's guide to training physics-informed neural networks.
   - The most practical paper in this list.

### Phase 3: Neural Operators

9. Lu et al. (2021). DeepONet. Learning nonlinear operators.
10. Li et al. (2021). Fourier Neural Operator for parametric partial differential equations.
    - Implement FNO from scratch, including the spectral convolution layer.
11. Wang, Wang, Perdikaris (2021). Learning the solution operator of parametric PDEs with physics-informed DeepONets.
12. Kovachki et al. (2023). Neural operator: learning maps between function spaces.
    - The theory behind the operator approach.

### Phase 4: Benchmarks and Honest Evaluation

13. Takamoto et al. (2022). PDEBench.
14. Grossmann et al. (2024). Can physics-informed neural networks beat the finite element method?
15. Hao et al. (2023). PINNacle. A PINN benchmark.

### Phase 5: Industry-Relevant Frontier

16. Brandstetter et al. (2022). Message passing neural PDE solvers.
17. Pathak et al. (2022) and Lam et al. (2023). FourCastNet and GraphCast.
    - Weather models that attract industry attention.
18. Wu et al. (2024) and Herde et al. (2024). Transolver and Poseidon.
    - Optional: PDE foundation models.

### Surveys to Read Alongside

- Karniadakis et al. (2021). Physics-informed machine learning (Nature Reviews Physics).
- Cuomo et al. (2022). Scientific machine learning through physics-informed neural networks.

---

## Part 2: Roadmap (about 6 months)

Mark items with `[x]` as you complete them.

### Weeks 1-4: PINN Fundamentals (Phase 1)

Watch first:
- Maziar Raissi lectures
- Ben Moseley PINN talks
- Steve Brunton (physics-informed ML)
- Karpathy Zero to Hero (habits)

- [ ] Implement a PINN in PyTorch from scratch for the 1D heat equation
- [ ] Implement a PINN for Burgers' equation
- [ ] Implement an inverse problem (infer a parameter from data)
- [ ] Write a classical baseline for each (finite differences or spectral, NumPy/SciPy)
- [ ] Deliverable: repo with clean code, plots, and a short write-up comparing PINN and classical accuracy and runtime

### Weeks 5-9: Failure Modes and Fixes (Phase 2)

Watch first:
- Ben Moseley (when PINNs work and fail)
- Paris Perdikaris (gradient pathology and NTK talks)
- Machine Learning & Simulation (JAX)

- [ ] Reproduce a failure case (e.g., high-wavenumber convection from Krishnapriyan)
- [ ] Fix it with loss balancing, Fourier features, or curriculum training
- [ ] Learn JAX by reimplementing one experiment in it
- [ ] Deliverable: blog post, "Why my PINN failed and what fixed it," with ablations

### Weeks 10-15: Neural Operators (Phase 3)

Watch first:
- Zongyi Li and Anima Anandkumar (FNO talks)
- Steve Brunton (neural operators)
- Perdikaris (physics-informed DeepONets)

- [ ] Implement DeepONet from scratch on Burgers and Darcy flow
- [ ] Implement FNO from scratch
- [ ] Build a physics-informed variant
- [ ] Deliverable: benchmark across PINN, DeepONet, FNO, and a classical solver (accuracy, speed, data needs, generalization)

### Weeks 16-21: Your Original Project (Phases 4-5)

Watch first:
- NeurIPS/ICML ML-for-science workshop recordings
- Nathan Kutz, Chris Rackauckas (SciML)

- [ ] Pick one question from your own physics background (e.g., does enforcing a conservation law improve long-time stability?)
- [ ] Test it properly using PDEBench data, reporting results honestly, including where ML loses to classical solvers
- [ ] Deliverable: arXiv preprint or submission to an ML-for-science workshop (NeurIPS or ICML)

### Weeks 22-26: Production Skills and Job Search

Watch first:
- NVIDIA Developer (PhysicsNeMo, AI-for-science talks)

- [ ] Learn industry tooling: PhysicsNeMo, the neuraloperator library, mixed precision, multi-GPU training, deployment
- [ ] Contribute a fix or feature to an open-source library such as DeepXDE or neuraloperator
- [ ] Start applying from about week 16, not week 26; interviews are feedback

---

## Part 3: Practical Notes

- Start applying early. Target AI-for-science teams, simulation and CFD companies, weather and climate startups, materials and drug discovery, science groups at large companies, and "ML engineer, scientific computing" roles.
- Keep one public repo per phase with a good README. Recruiters look at these before anything else.
- Don't oversell PINNs in interviews. Showing you know their limits, and when a neural operator or classical solver is the better choice, demonstrates real expertise.
- Lean on your MS in physics. Your physics background is a differentiator in this field, not just a credential.

---

## Part 4: YouTube Channels

Recommended from memory. Check that each channel is still active and look at recent uploads before relying on it.

### Best Overall for This Roadmap

- Steve Brunton (Eigensteve): Scientific ML, physics-informed learning, SINDy, dynamical systems, neural operators. Suits a physics background. Use for Phases 1 and 3.
- Machine Learning & Simulation (Felix Koehler): Numerical PDE solvers, JAX, differentiable physics, with code walkthroughs. Useful for classical baselines and learning JAX.

### Phases 1-2: PINNs

- Maziar Raissi: Original author of the PINN paper. Watch his lectures alongside the paper.
- Ben Moseley: Short, practical talks on PINNs, including when they work and when they don't. Fits the Phase 2 failure-modes work.
- Brown University / Karniadakis group lectures: Search for "Physics-Informed Machine Learning" and DeepXDE talks.

### Phase 3: Neural Operators

- Anima Anandkumar and Zongyi Li talks: Search for "Fourier Neural Operator" talks, including NVIDIA GTC and conference recordings.
- Paris Perdikaris: Physics-informed DeepONets, gradient pathology and NTK papers.

### Supporting Skills

- Andrej Karpathy: Zero to Hero. Deep implementation habits and debugging.
- Nathan Kutz: Data-driven science and engineering lectures.
- Chris Rackauckas: Julia/SciML talks and MIT scientific machine learning lectures. Differentiable-programming angle.
- Yannic Kilcher and Umar Jamil: Paper walkthroughs for reading new papers quickly.
- NVIDIA Developer: PhysicsNeMo and industry AI-for-science talks. Helps with Phase 5 and the job search.

### How to Use Them

- Watch a lecture first, then read the paper, then implement from scratch. Don't watch passively.
- Use videos for intuition and papers for details. Don't copy their code.
- Rebuild it yourself, then compare.
- Conference talks (NeurIPS, ICML ML-for-science workshops) are free on YouTube and show what the field is currently working on.

---

## Project Goals

This repository is meant to support a structured progression:

- Learn the foundations of PINNs and automatic differentiation.
- Diagnose and mitigate common failure modes.
- Move from PINNs to neural operators and advanced scientific ML.
- Produce reproducible, publication-worthy results.
- Build a portfolio that demonstrates practical AI-for-science capability.

---

## Suggested Execution Pattern

1. Read the paper and identify the key equations.
2. Re-implement the model in PyTorch from scratch.
3. Compare against a classical baseline.
4. Run ablations and report failures honestly.
5. Write a clear summary and publish the results.

---

## Final Note

The goal is not to chase the latest trend, but to become capable of solving real scientific ML problems with a transparent, rigorous, and reproducible workflow.

This roadmap is intentionally practical: every milestone is designed to be turned into a public repository, blog post, or scientific preprint.
