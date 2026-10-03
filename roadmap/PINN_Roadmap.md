# Physics-Informed Neural Networks: Papers and Roadmap

A comprehensive implementation plan for building expertise in Physics-Informed Machine Learning—from core concepts to industry-ready applications.

**Goal**: Each phase ends with something publishable. Public, verifiable work is what gets you hired.

> **Note**: Verify paper titles and details against arXiv or Google Scholar as you progress.

---

## Part 1: Papers List (Implementation Order)

### Phase 1: Core PINNs

1. **Raissi, Perdikaris, Karniadakis (2019)**. Physics-informed neural networks.
   - The original foundational paper
   - Reproduce Burgers' equation (forward problem) and an inverse problem

2. **Lu et al. (2021)**. DeepXDE.
   - Study the design patterns
   - Compare your implementation against the library

3. **Baydin et al. (2018)**. Automatic differentiation in machine learning: a survey.
   - Understand how higher-order derivatives in your loss function work

### Phase 2: Why PINNs Fail and Fixes

4. **Krishnapriyan et al. (2021)**. Characterizing possible failure modes in physics-informed neural networks.

5. **Wang, Teng, Perdikaris (2021)**. Understanding and mitigating gradient flow pathologies in PINNs.

6. **Wang, Yu, Perdikaris (2022)**. When and why PINNs fail to train: a neural tangent kernel perspective.

7. **Tancik et al. (2020)**. Fourier features let networks learn high-frequency functions in low dimensional domains.

8. **Wang et al. (2023)**. An expert's guide to training physics-informed neural networks.
   - ⭐ The most practical paper in this list

### Phase 3: Neural Operators

9. **Lu et al. (2021)**. DeepONet. Learning nonlinear operators.

10. **Li et al. (2021)**. Fourier Neural Operator for parametric partial differential equations.
    - Implement FNO from scratch, including the spectral convolution layer

11. **Wang, Wang, Perdikaris (2021)**. Learning the solution operator of parametric PDEs with physics-informed DeepONets.

12. **Kovachki et al. (2023)**. Neural operator: learning maps between function spaces.
    - Deep dive into the theory behind the operator approach

### Phase 4: Benchmarks and Honest Evaluation

13. **Takamoto et al. (2022)**. PDEBench.

14. **Grossmann et al. (2024)**. Can physics-informed neural networks beat the finite element method?

15. **Hao et al. (2023)**. PINNacle. A PINN benchmark.

### Phase 5: Industry-Relevant Frontier

16. **Brandstetter et al. (2022)**. Message passing neural PDE solvers.

17. **Pathak et al. (2022) and Lam et al. (2023)**. FourCastNet and GraphCast.
    - Weather models that attract significant industry attention

18. **Wu et al. (2024) and Herde et al. (2024)**. Transolver and Poseidon.
    - Optional: PDE foundation models

### Surveys to Read Alongside

- **Karniadakis et al. (2021)**. Physics-informed machine learning (*Nature Reviews Physics*).
- **Cuomo et al. (2022)**. Scientific machine learning through physics-informed neural networks.

---

## Part 2: 6-Month Roadmap

> Track your progress by marking items with `[x]` as you complete them.

### Weeks 1-4: PINN Fundamentals (Phase 1)

**Watch First:**
- Maziar Raissi lectures
- Ben Moseley PINN talks
- Steve Brunton (Physics-informed ML)
- Andrej Karpathy Zero to Hero (learning habits)

**Tasks:**
- [ ] Implement a PINN in PyTorch from scratch for the 1D heat equation
- [ ] Implement a PINN for Burgers' equation
- [ ] Implement an inverse problem (infer a parameter from data)
- [ ] Write a classical baseline for each (finite differences or spectral, NumPy/SciPy)
- [ ] **Deliverable**: Repository with clean code, plots, and a short write-up comparing PINN and classical accuracy/runtime

### Weeks 5-9: Failure Modes and Fixes (Phase 2)

**Watch First:**
- Ben Moseley (when PINNs work and fail)
- Paris Perdikaris (gradient pathology and NTK talks)
- Machine Learning & Simulation (JAX)

**Tasks:**
- [ ] Reproduce a failure case (e.g., high-wavenumber convection from Krishnapriyan et al.)
- [ ] Fix it with loss balancing, Fourier features, or curriculum training
- [ ] Learn JAX by reimplementing one experiment in it
- [ ] **Deliverable**: Blog post "Why my PINN failed and what fixed it," with ablation studies

### Weeks 10-15: Neural Operators (Phase 3)

**Watch First:**
- Zongyi Li and Anima Anandkumar (FNO talks)
- Steve Brunton (neural operators)
- Paris Perdikaris (physics-informed DeepONets)

**Tasks:**
- [ ] Implement DeepONet from scratch on Burgers and Darcy flow
- [ ] Implement FNO from scratch
- [ ] Build a physics-informed variant
- [ ] **Deliverable**: Benchmark across PINN, DeepONet, FNO, and a classical solver (accuracy, speed, data needs, generalization)

### Weeks 16-21: Your Original Project (Phases 4-5)

**Watch First:**
- NeurIPS/ICML ML-for-science workshop recordings
- Nathan Kutz lectures
- Chris Rackauckas (SciML)

**Tasks:**
- [ ] Pick one question from your physics background (e.g., "Does enforcing a conservation law improve long-time stability?")
- [ ] Test it properly using PDEBench data, reporting results honestly—including where ML loses to classical solvers
- [ ] **Deliverable**: arXiv preprint or submission to an ML-for-science workshop (NeurIPS or ICML)

### Weeks 22-26: Production Skills and Job Search

**Watch First:**
- NVIDIA Developer (PhysicsNeMo, AI-for-science talks)

**Tasks:**
- [ ] Learn industry tooling: PhysicsNeMo, neuraloperator library, mixed precision, multi-GPU training, deployment
- [ ] Contribute a fix or feature to an open-source library (e.g., DeepXDE, neuraloperator)
- [ ] Start applying from about week 16, not week 26; interviews are feedback

---

## Part 3: YouTube Channels & Resources

### Best Overall for This Roadmap

- **Steve Brunton (Eigensteve)**: Scientific ML, physics-informed learning, SINDy, dynamical systems, neural operators. Perfect for a physics background. *Use for Phases 1 and 3.*
- **Machine Learning & Simulation (Felix Koehler)**: Numerical PDE solvers, JAX, differentiable physics, with code walkthroughs. Useful for classical baselines and learning JAX.

### Phases 1-2: PINNs Foundations

- **Maziar Raissi**: Original PINN author. Watch alongside the paper.
- **Ben Moseley**: Short, practical talks on PINNs—when they work and when they don't.
- **Brown University / Karniadakis Group**: Search "Physics-Informed Machine Learning" and DeepXDE talks.

### Phase 3: Neural Operators

- **Anima Anandkumar and Zongyi Li**: Search "Fourier Neural Operator" talks (NVIDIA GTC, conferences).
- **Paris Perdikaris**: Physics-informed DeepONets, gradient pathology, and NTK papers.

### Supporting Skills

- **Andrej Karpathy**: Zero to Hero—deep implementation habits and debugging.
- **Nathan Kutz**: Data-driven science and engineering lectures.
- **Chris Rackauckas**: Julia/SciML talks and MIT scientific machine learning lectures.
- **Yannic Kilcher & Umar Jamil**: Paper walkthroughs for reading new papers quickly.
- **NVIDIA Developer**: PhysicsNeMo and industry AI-for-science talks. *Essential for Phase 5 and job search.*

### How to Use These Resources

✅ Watch a lecture first → read the paper → implement from scratch  
✅ Use videos for intuition; papers for details  
✅ Don't copy their code; rebuild it yourself, then compare  
✅ Conference talks (NeurIPS, ICML ML-for-science workshops) are free on YouTube and show current field trends

---

## Part 4: Practical Notes for Job Success

### Strategy
- **Start applying early**. Target AI-for-science teams, simulation/CFD companies, weather/climate startups, materials and drug discovery, science groups at large companies, and "ML engineer, scientific computing" roles.
- Keep one public repository per phase with a good README. *Recruiters look at these before anything else.*
- **Don't oversell PINNs** in interviews. Showing you know their limits—and when a neural operator or classical solver is the better choice—demonstrates real expertise.
- Lean on your physics background. It's a differentiator in this field, not just a credential.

### Key Mindsets
- **Fail fast, learn publicly**: Each phase should produce a publishable artifact.
- **Benchmark honestly**: Include where ML loses to classical methods.
- **Understand the theory**: Gradient pathology, operator learning, and spectral methods aren't optional.

---

## Getting Started

1. Clone this repository
2. Pick **Phase 1** papers and start with Week 1-4 tasks
3. Create a new subdirectory for each phase (e.g., `phase1-core-pinns/`)
4. Push clean, documented code and a short write-up at the end of each phase
5. Share on Twitter, arXiv, and LinkedIn—recruiters find candidates this way

Good luck! 🚀

---

*Last updated: 2026*  
*Based on roadmap for AI-for-science roles focused on Physics-Informed Machine Learning*
