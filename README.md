# Dual Phase-Field Modeling of Nacre's Asymmetric Mechanical Strength

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9+-blue.svg)](https://julialang.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.mechmat.2026.XXXXXX-brightgreen)](https://doi.org/10.1016/j.mechmat.2025.XXXXXX)

This repository contains the computational implementation of the phase-field fracture framework described in:

> **Vigliotti, A.** (2025). "Multiscale Phase-Field Analysis of Nacre's Asymmetric Mechanical Strength: A Dual Damage Field Approach with Variational Irreversibility Constraints." *Mechanics of Materials*. https://doi.org/10.1016/j.mechmat.2025.XXXXXX

## Overview

Nacre (mother-of-pearl) exhibits remarkable mechanical asymmetry: its compressive strength exceeds tensile strength by factors of 4-5×. This Julia package implements a computational framework that explains this behavior through phase-field modeling with:

- **Dual independent damage fields** for aragonite tablets (brittle, AT1) and organic matrix (ductile, AT2)
- **KKT-based damage irreversibility** enforcement without history field tracking
- **Periodic boundary conditions** for RVE homogenization
- **Automatic differentiation** for exact tangent matrices
- **Staggered Newton-Raphson** solution scheme

### Key Results

The simulations reproduce:
- ✅ Compression-to-tension strength ratios of 4-5× (experimental range: 2.7-5.0×)
- ✅ Phase-separated failure modes: matrix damage under tension, tablet fragmentation under compression
- ✅ Orientation-independent asymmetry across loading directions
- ✅ Biaxial failure envelopes with distinctive tension-compression topology

---

## Table of Contents

- [Theoretical Background](#theoretical-background)
  - [Phase-Field Fracture Theory](#phase-field-fracture-theory)
  - [Dual Damage Field Formulation](#dual-damage-field-formulation)
  - [Periodic Boundary Conditions](#periodic-boundary-conditions)
- [Numerical Implementation](#numerical-implementation)
  - [Solution Algorithm](#solution-algorithm)
  - [Damage Irreversibility](#damage-irreversibility)
  - [Multiscale Homogenization](#multiscale-homogenization)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
  - [Basic Uniaxial Test](#basic-uniaxial-test)
  - [Boundary Condition Specification](#boundary-condition-specification)
  - [Advanced Multi-Axial Loading](#advanced-multi-axial-loading)
- [Mesh Requirements](#mesh-requirements)
- [Post-Processing](#post-processing)
- [Performance Considerations](#performance-considerations)
- [Citation](#citation)
- [License](#license)

---

## Theoretical Background

### Phase-Field Fracture Theory

Phase-field methods regularize sharp crack discontinuities as diffuse damage zones, enabling variational fracture formulation without explicit crack tracking. The total free energy functional is:

$$ G(\mathbf{u}, d) = \int_\Omega \left[ g(d)\phi^+(\varepsilon) + \phi^-(\varepsilon) + \Psi(d) \right] \mathrm{d}\Omega $$

where:
- $u$ : displacement field
- $d \in [0,1]$ : scalar damage variable (0=intact, 1=failed)
- $g(d) = (1-d)^2$ : degradation function reducing load-bearing capacity
- $\phi^+, \phi^-$: tensile and compressive strain energy densities
- $\psi(d)=(G_c/2l_0)\left(d^n + l_0^2\|\nabla d\|^2\right)$: crack surface energy density

#### AT1 vs AT2 Models

The exponent $n$ distinguishes model behavior:

| Model | n | Behavior | Application |
|-------|---|----------|-------------|
| **AT1** | 1 | Sharp elastic limit, rapid damage onset | Brittle materials (aragonite tablets) |
| **AT2** | 2 | Gradual damage from loading start | Ductile materials (organic matrix) |

**Key parameters:**
- $G_c$: Critical energy release rate (material toughness)
- $l_0$: Regularization length controlling damage zone width
- For physically meaningful results: $l_0$ ≪ Lcharacteristic (geometric features)

### Dual Damage Field Formulation

Standard single-field phase-field models fail for composites with vastly different constituent properties due to artificial damage diffusion across interfaces. This implementation employs **independent damage fields** for each phase:

$$ G(\mathbf{u}, d_\text{t}, d_\text{m}) = \int_{\Omega_\text{t}} \left[ (1-d_\text{t})^2\phi_\text{t}^+ + \phi_\text{t}^- + \Psi_t(d_\text{t}) \right] \mathrm{d}\Omega + \int_{\Omega_\text{m}} \left[ (1-d_\text{m})^2\phi_\text{m}^+ + \phi_\text{m}^- + \Psi_m(d_\text{m}) \right] \mathrm{d}\Omega $$

**Coupling mechanism:**
- Damage fields $d_t$ and $d_m$ evolve independently within their domains
- Phases remain coupled through displacement field $u$ (mechanical equilibrium)
- Stress redistribution from damaged regions naturally transfers to intact phase

**Physical justification:**
- Prevents unphysical damage propagation from weak matrix into strong tablets
- Captures phase-specific failure physics (brittle vs ductile)
- Enables strength ratios > 7× between constituents

### Periodic Boundary Conditions

The Representative Volume Element (RVE) employs periodic boundary conditions for homogenization:

#### Displacement Periodicity

$$ \boldsymbol{u}(\mathbf{r} + \mathbf{a}_i) - \boldsymbol{u}(\mathbf{r}) = \boldsymbol{\varepsilon}_M \cdot \mathbf{a}_i $$

where:
- $\mathbf{a}_i$: lattice vectors defining RVE periodicity
- $\boldsymbol{\varepsilon}_M$: prescribed macroscopic strain tensor (6 components in Voigt notation)

**Implementation:** `u = B₀ × u₀ + Bϵ × ϵₘ`
- **B₀**: maps independent nodal DOFs to full field
- **Bϵ**: couples macroscopic strain to boundary displacements

#### Damage Periodicity
$$
d(r + a_i) = d(r)    \forall r \in \partial \Omega
$$

**Implementation:** `d = B₀d × d₀`
- Pure periodicity (no macroscopic gradient)
- Separate constraint matrices for tablets and matrix

#### Macroscopic Stress Recovery
Once equilibrium is reached for prescribed **ϵₘ**, the conjugate macroscopic stress is:

```
σₘ = (1/V_RVE) ∂G/∂ϵₘ = (1/V_RVE) Bϵᵀ · r_u
```

where **r_u** is the displacement residual vector.

---

## Numerical Implementation

### Solution Algorithm

The code employs a **staggered scheme** alternating between displacement and damage solutions:

```
for each load step n:
    1. Apply boundary conditions: ϵₘⁿ = (n/N) × ϵₘ_target
    2. Solve displacement: min_u G(u, d_t^(n-1), d_m^(n-1))
    3. Solve tablet damage: min_{d_t≥d_t^(n-1)} G(uⁿ, d_t, d_m^(n-1))
    4. Solve matrix damage: min_{d_m≥d_m^(n-1)} G(uⁿ, d_tⁿ, d_m)
    5. Final displacement update
    6. Compute macroscopic stress σₘⁿ
```

#### Newton-Raphson for Displacement

At each staggered iteration, displacement solves:
```
K_uu × Δu = -r_u
```
with stabilized tangent: **K_eff = λT·M + K_uu**

**Adaptive tangent updates:**
- Rebuild when residual increases (non-monotonic convergence)
- Maintains quadratic convergence near solution
- Typical iterations: 2-5 per load step

#### Active Set Method for Damage

Damage fields solve constrained minimization with irreversibility:
```
minimize    G(u, d)
subject to  d ≥ d_old
```

**KKT conditions** identify active constraints:
- If ∂G/∂d < 0: constraint active → d = d_old (no damage healing)
- If ∂G/∂d ≥ 0: constraint inactive → update d

Only the **active set** (typically 1-10% of DOFs) requires linear solve.

### Damage Irreversibility

Traditional history field methods track maximum strain energy:
```
H(x,t) = max_{τ≤t} φ⁺(ϵ(x,τ))
```
Issues: artificial energy clamping, numerical instabilities.

**This implementation** uses variational approach:
- Irreversibility enforced via **inequality constraints** in optimization
- **Lagrange multipliers** naturally emerge from KKT conditions
- No history field storage required
- Robust for both AT1 and AT2 formulations

### Multiscale Homogenization

The framework bridges microscale damage evolution to macroscale constitutive response:

**Microscale (RVE level):**
- Detailed geometry: tablets, matrix, interfaces
- Field resolution: damage zones, crack paths

**Macroscale (stress-strain):**
- Effective properties: σₘ(ϵₘ), Cₘ(ϵₘ)
- Homogenized damage: V*_t, V*_m (volume fractions)
- Outputs: 6×6 tangent stiffness, failure envelopes

**Separation of scales requirement:**
- RVE size >> l₀ (damage regularization)
- Macroscale features >> RVE size
- For nacre: l₀ = 0.01 μm, tablet = 5 μm, RVE = 15 μm

---

## Installation

### Prerequisites

- **Julia ≥ 1.9** ([download](https://julialang.org/downloads/))

### Installation from GitHub

The package can be installed directly from the Julia REPL:

```julia
using Pkg
Pkg.add("AD4SM")
Pkg.add(url="https://github.com/avigliotti/nacre-dual-damage" )
```

### Testing the Installation

After installation, you can test that the package loads correctly:

```julia
using NacreDualDamage
# No error indicates successful loading
```

---

## Usage Examples

### Basic Uniaxial Test

To run the examples, first copy the example files to your current working directory:

```julia
using NacreDualDamage

# Copy example files to current directory
sPackageRoot = joinpath(dirname(pathof(NacreDualDamage)), "..")
cp(joinpath(sPackageRoot, "examples"), "./examples", force=true)
cd("./examples/" )
```

This creates the following directory structure:
```
examples/
├── mesh_files/              # Contains input geometries (.inp)
├── jld2_files/              # Stores binary output
├── vtk_files/               # Stores visualization files
```

The `mesh_files` directory contains pre-generated meshes for nacre microstructures. For example, the file `nacre_tet1x1x1L0300w0050t0050rp060lc0300.inp` represents a single unit cell with:
- Tablet length: 3 μm
- Tablet thickness: 0.5 μm
- Matrix thickness: 0.02 μm

### Running a Simple Simulation

Here's a basic example of uniaxial tension along the y-direction:

```julia

# Apply 2% strain in the y-direction (component 2 in Voigt notation)
ϵM0 = [NaN, 0.02, NaN, NaN, NaN, NaN];
sPost = "NaNe22NaNt";
results = solve_nacre_model(
    sModelName = "nacre_tet1x1x1L0300w0050t0050rp060lc0300",
    ϵM0        = ϵM0,
    nSteps     = 100,
    sPath      = pwd(),
    sPost      = sPost
);

# Extract stress-strain curve
ϵ22 = results["ϵM"][2, :];  # Strain in direction 22
σ22 = results["σM"][2, :];  # Stress in direction 22

# Plot the results
using Plots
plot(ϵ22, σ22, xlabel="Strain ε₂₂", ylabel="Stress σ₂₂ (MPa)", 
     title="Uniaxial Tensile Response", linewidth=2, grid=true )
```

binary data results will be saved in the file 

`examples/jld2_files/nacre_tet1x1x1L0300w0050t0050rp060lc0300NaNe22NaNt.jld2`

while paraview files will be acessible from the file

`examples/vtk_files/nacre_tet1x1x1L0300w0050t0050rp060lc0300NaNe22NaNt.pvd`


### Boundary Condition Specification

The macroscopic deformation tensor **ϵM0** prescribes boundary conditions in **Voigt notation**:

```julia
ϵM0 = [ϵ₁₁, ϵ₂₂, ϵ₃₃, γ₂₃, γ₁₃, γ₁₂]
```

where γᵢⱼ = 2ϵᵢⱼ are engineering shear strains.

#### Mixed Boundary Conditions

Each component can be either:
1. **Strain-controlled** (prescribed value): `ϵM0[i] = value`
2. **Stress-controlled** (free): `ϵM0[i] = NaN`

**Physical meaning of `NaN`:**
- Component i is **unconstrained** (free to adjust)
- Conjugate stress component σₘ[i] = 0 (homogeneous BC)
- RVE deforms to minimize energy without constraint in that direction

#### Example Loading Conditions

**Uniaxial tension (σ₁₁ prescribed, all others free):**
```julia
ϵM0 = [0.02, NaN, NaN, NaN, NaN, NaN]
# Applies: ϵ₁₁ = 2%
# Result: σ₂₂ = σ₃₃ = σ₂₃ = σ₁₃ = σ₁₂ = 0
# Free: ϵ₂₂, ϵ₃₃ (develop via Poisson effect)
```

**Plane strain (ϵ₃₃ = 0, σ₂₂ = 0):**
```julia
ϵM0 = [0.03, NaN, 0.0, NaN, NaN, NaN]
# Applies: ϵ₁₁ = 3%, ϵ₃₃ = 0
# Result: σ₂₂ = σ₂₃ = σ₁₃ = σ₁₂ = 0
# Free: ϵ₂₂ (adjusts naturally)
# Non-zero: σ₃₃ (reaction from constraint)
```

**Equibiaxial tension, plane stress:**
```julia
ϵM0 = [0.01, 0.01, NaN, NaN, NaN, NaN]
# Applies: ϵ₁₁ = ϵ₂₂ = 1%
# Result: σ₃₃ = σ₂₃ = σ₁₃ = σ₁₂ = 0
# Free: ϵ₃₃ (contracts due to Poisson)
```

**Pure shear:**
```julia
ϵM0 = [NaN, NaN, NaN, NaN, NaN, 0.015]
# Applies: γ₁₂ = 1.5% (engineering shear)
# Result: σ₁₁ = σ₂₂ = σ₃₃ = σ₂₃ = σ₁₃ = 0
# Free: all normal strains
```

### Advanced Multi-Axial Loading

**Compression with rotation:**
```julia
results_comp = solve_nacre_model(
    sModelName = "nacre_tet1x1x1L0300w0050t0050rp060lc0300",
    ϵM0 = [-0.03, NaN, NaN, NaN, NaN, NaN],
    θ = π/6,  # 30° rotation about z-axis
    nSteps = 150
)
```

**Parametric study of biaxial states:**
```julia
# Generate failure envelope
angles = range(0, π/2, length=20)
results = []

for θ in angles
    # Apply strain in direction defined by angle θ
    ϵM0 = [cos(θ), sin(θ), NaN, NaN, NaN, NaN] * 0.05
    
    push!(results, solve_nacre_model(
            sModelName = "nacre_tet1x1x1L0300w0050t0050rp060lc0300",
            ϵM0 = ϵM0,
            nSteps = 100
            )
        )
end
```

### Custom Material Properties

Material properties for tablets and matrix can be customized:

```julia
# Stronger tablets (higher ceramic content)
tablets_custom = let
    l0 = 1e-2  # Regularization length in μm
    E, ν, ϵc = 120.0, 0.25, 0.004  # GPa, -, -
    Gc = 2E * l0 * ϵc^2  # Toughness
    PhaseField{Hooke{Float64}, :ATn}(l0, Gc, Hooke(E, ν, 1.0, small=true), 1)
end

# Softer matrix (higher water content)
matrix_custom = let
    l0 = 1e-2
    E, ν, ϵc = 2.0, 0.45, 0.035
    Gc = 3E * l0 * ϵc^2
    λ, μ = E*ν/(1+ν)/(1-2ν), E/2/(1+ν)
    C1, K = μ/2, λ/2
    PhaseField{NeoHooke{Float64}, :ATn}(l0, Gc, NeoHooke(C1, K, 1.0), 2)
end

results = solve_nacre_model(
    sModelName = "nacre_tet1x1x1L0300w0050t0050rp060lc0300",
    tablets_mat = tablets_custom,
    matrix_mat = matrix_custom,
    ϵM0 = [0.04, NaN, NaN, NaN, NaN, NaN]
)
```

---

## Mesh Requirements

Meshes for this study were generated using [Gmsh](https://gmsh.info/) and saved as Abaqus input files (.inp format). Arbitrary meshes can be used provided they satisfy periodicity requirements:

- Opposite boundaries must have identical tessellation
- Node pairing tolerance: 1e-12 (automatic detection)

### Required Node Sets in Input File

```
*NSET, NSET=left
  <nodes on x=0 boundary>
*NSET, NSET=right
  <nodes on x=Lx boundary>
*NSET, NSET=bottom
  <nodes on y=0 boundary>
*NSET, NSET=top
  <nodes on y=Ly boundary>
*NSET, NSET=front
  <nodes on z=0 boundary>
*NSET, NSET=back
  <nodes on z=Lz boundary>

*NSET, NSET=tablets
  <all nodes in tablet regions>
*NSET, NSET=matrix
  <all nodes in matrix regions>
```

### Required Element Sets

```
*ELSET, ELSET=tablets
  <all elements in tablet regions>
*ELSET, ELSET=matrix
  <all elements in matrix regions>
```

---

## Post-Processing

### Loading Results

```julia
using FileIO

# Load saved data
data = load("./jld2_files/nacre_tet1x1x1L0300w0050t0050rp060lc0300NaNe22NaNt.jld2");

# Available fields
println(keys(data))
# Output: ["ϵM", "σM", "steps", "Vol", "Vd", ...]
```

### Stress-Strain Curves

```julia
using Plots

ϵ22 = data["ϵM"][2, :];  # Strain component 22
σ22 = data["σM"][2, :];  # Stress component 22

plot(ϵ22, σ22, 
     xlabel="Strain ε₂₂", 
     ylabel="Stress σ₂₂ (MPa)",
     title="Uniaxial Tensile Response",
     linewidth=2,
     grid=true,
     legend=false)
```

### Volume-Averaged Damage Evolution

```julia
# Volume fraction of damaged material
Vd_tablets = data["Vd"][1];  # Tablet damage volume fraction
Vd_matrix = data["Vd"][2];   # Matrix damage volume fraction
ϵ = data["ϵM"][2, :];           # Macroscopic strain

plot(ϵ, [Vd_tablets Vd_matrix],
     label=["Tablets" "Matrix"],
     xlabel="Macroscopic Strain",
     ylabel="Damage Volume Fraction",
     title="Damage Evolution",
     linewidth=2,
     grid=true)
```

### Field Visualization with ParaView

Output `.pvd` files can be opened in [ParaView](https://www.paraview.org/) for 3D visualization:

1. Load `./vtk_files/nacre_tet1x1x1L0300w0050t0050rp060lc0300.pvd`
2. Color by damage field (`d_tablets` or `d_matrix`)
3. Use the timeline slider to animate through loading steps

**Visualization tips:**
- Apply "Threshold" filter to show only damaged regions (d > 0.1)
- Use "Glyph" filter for displacement vectors
- Apply "Clip" to reveal internal damage patterns

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Vigliotti2025,
  title = {Multiscale Phase-Field Analysis of Nacre's Asymmetric Mechanical Strength: A Dual Damage Field Approach with Variational Irreversibility Constraints},
  author = {Vigliotti, Andrea},
  journal = {Mechanics of Materials},
  year = {2026},
  volume = {###},
  pages = {###},
  doi = {10.1016/j.mechmat.2026.###}
}
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Academic and Commercial Use:** Free under MIT terms (attribution required).

---

## Contact

**Andrea Vigliotti**  
Innovative Materials Laboratory  
Italian Aerospace Research Centre (CIRA)  
📧 andrea.vigliotti@gmail.com  
🌐 [GitHub Profile](https://github.com/avigliotti)

**Bug Reports and Feature Requests:** Please use the [GitHub Issues](https://github.com/avigliotti/nacre-dual-damage/issues) page.

---

## Acknowledgments

This work was supported by the METMAT project, financed by the Italian Aerospace Research Program (PRORA). Computational resources were provided by CIRA's High Performance Computing facility.

---

**Repository:** https://github.com/avigliotti/nacre-dual-damage  
**Documentation:** [Wiki](https://github.com/avigliotti/nacre-dual-damage/wiki)  
**Paper:** https://authors.elsevier.com/sd/article/S0167-6636(26)00031-1

*Last updated: January 2025*
