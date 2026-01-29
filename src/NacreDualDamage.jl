"""
    NacreDualDamage

A Julia module for phase-field fracture simulation of nacre's brick-and-mortar microstructure
using a dual damage field approach with periodic boundary conditions.

This module implements the computational framework described in:
"Multiscale Phase-Field Analysis of Nacre's Asymmetric Mechanical Strength: 
A Dual Damage Field Approach with Variational Irreversibility Constraints"

# Main Features
- Dual independent damage fields for tablets and matrix
- Staggered solution scheme (displacement-damage alternation)
- KKT-based damage irreversibility enforcement
- Periodic boundary conditions for RVE homogenization
- Parallel element assembly using threading
- Optional PARDISO direct solver support

# External Dependencies
- WriteVTK: Visualization output
- AbaqusReader: Mesh file input
- Pardiso: Optional fast direct solver
- AD4SM: Automatic differentiation for solid mechanics
"""
module NacreDualDamage

using WriteVTK, LinearAlgebra, SparseArrays, Printf, Random
using AbaqusReader, Logging, FileIO, Dates
using Pardiso

using AD4SM

using .Solvers, .Materials, .Elements
using .Elements: getVd

# Public exports
export AD4SM, Solvers, Materials, Elements
export solve_nacre_model

#==============================================================================
                        MAIN SIMULATION DRIVER
==============================================================================#

"""
    solve_nacre_model(; kwargs...) -> Dict

Execute a complete phase-field fracture simulation on a nacre RVE with dual damage fields.

This function orchestrates the entire simulation workflow:
1. Load geometry and setup periodic boundary conditions
2. Initialize displacement and damage fields
3. Execute staggered Newton-Raphson solution for each load step
4. Save visualization (VTK) and binary (JLD2) outputs
5. Return complete simulation history

# Simulation Algorithm

The solver employs a staggered scheme at each load increment:
```
for each load step:
    1. Apply boundary conditions (prescribed strain εₘ)
    2. Solve for displacement u with damage d fixed
    3. Solve for damage d_tablets with u fixed
    4. Solve for damage d_matrix with u fixed
    5. Final displacement update
    6. Store results and check convergence
```

Damage irreversibility is enforced via KKT conditions: dⁿ⁺¹ ≥ dⁿ

# Keyword Arguments

## Model Definition
- `sModelName::String`: Base name of mesh file (default: "platelets_tet1x1x1L1000w0100t0200rp500lct200lcm0250")
- `θ, ψ, ζ::Float64`: Euler angles for RVE rotation (default: 0.0)

## Material Properties
- `matrix_mat::PhaseField`: Matrix phase-field material (default: AT2 neo-Hookean)
  - Default: E=4 GPa, ν=0.27, εc=2.5%, l₀=0.01 μm, n=2
- `tablets_mat::PhaseField`: Tablet phase-field material (default: AT1 Hookean)
  - Default: E=100 GPa, ν=0.27, εc=0.5%, l₀=0.01 μm, n=1

## Loading
- `εM0::Vector`: Applied macroscopic strain [ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂] (default: [0.05, NaN, NaN, NaN, NaN, NaN])
  - Use `NaN` for free components (stress-controlled)
  - Use scalar values for constrained components (strain-controlled)

## Solution Control
- `nSteps::Int`: Number of load increments (default: 200)
- `maxiter::Int`: Maximum global staggered iterations (default: 4)

### Displacement Solver
- `miniteru::Int`: Minimum displacement iterations (default: 1)
- `maxiteru::Int`: Maximum displacement iterations (default: 7)
- `dTolrnorm::Float64`: Residual norm convergence ratio (default: 1.5)
- `dTolu::Float64`: Displacement residual tolerance (default: 5e-6)

### Damage Solver
- `maxiterd::Int`: Maximum damage iterations (default: 1)
- `dTold::Float64`: Damage residual tolerance (default: 1e-6)

## Numerical Parameters
- `λT::Float64`: Mass matrix stabilization parameter (default: 1e-2)
- `busePardiso::Bool`: Use PARDISO sparse direct solver (default: true)

## I/O Control
- `sPath::String`: Working directory (default: pwd())
- `sModelPath::String`: Input mesh directory (default: joinpath(sPath, "mesh_files"))
- `sJLD2Path::String`: Binary output directory (default: joinpath(sPath, "jld2_files"))
- `svtkPath::String`: VTK output directory (default: joinpath(sPath, "vtk_files"))
- `iSaveVTK::Int`: VTK output frequency in steps (default: nSteps÷10)
- `iSavejld2::Int`: JLD2 output frequency (default: iSaveVTK)
- `sPost::String`: Filename postfix (default: "")

# Returns

Dictionary containing complete simulation results:
- `"εM"`: Macroscopic strain history [6 × nSteps]
- `"σM"`: Macroscopic stress history [6 × nSteps]
- `"Vd"`: Volume-averaged damage fractions (tablets, matrix) [2 × nSteps]
- `"steps"`: Saved field snapshots [{iStep, u, d}...]
- `"Vol"`: RVE volumes (tablets, matrix)
- `"θ", "ψ", "ζ"`: Rotation angles
- `"ai"`: Periodic lattice vectors
- `"matrix_mat"`, `"tablets_mat"`: Material definitions
- `"datestart"`: Simulation start timestamp

# Example Usage

```julia
using NacreDualDamage

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

println(keys(results)

# Extract stress-strain curve
using PyPlot

ax = plt.subplot()
ax.plot(results["ϵM"][2,:], results["σM"][2,:])
ax.set_xlabel("\$\epsilon_{22}\$")
ax.set_ylabel("\$\sigma_{22}\$")
```

# Notes

## Mesh Requirements
Input mesh must contain:
- Node sets: "left", "right", "bottom", "top", "front", "back" (for periodicity)
- Node sets: "matrix", "tablets" (for material assignment)
- Element sets: "matrix", "tablets"

## Convergence Behavior
- Displacement solver uses Newton-Raphson with adaptive tangent updates
- Damage solver enforces irreversibility via active set method
- Simulation terminates on convergence failure (state saved for recovery)

"""
function solve_nacre_model(;
                           sModelName::String  = "platelets_tet1x1x1L1000w0100t0200rp500lct200lcm0250",
                           ϵM0         = [1, NaN, NaN, NaN, NaN, NaN]*5e-2,
                           matrix_mat  = let
                             l0        = 1e-2
                             Em,νm,ϵdm = 4.0,0.27,2.5e-2
                             Gcm       = 3Em*l0*ϵdm^2
                             λ,μ       = Em*νm/(1+νm)/(1-2νm), Em/2/(1+νm)
                             C1, K     = μ/2, λ/2
                             PhaseField{NeoHooke{Float64}, :ATn}(l0, Gcm, NeoHooke(C1,K,1.0), 2)
                           end,
                           tablets_mat = let 
                             l0        = 1e-2
                             Ef,νf,ϵdf = 100.0,0.27,5e-3
                             Gcf       = 2Ef*l0*ϵdf^2 
                             PhaseField{Hooke{Float64}, :ATn}(l0, Gcf, Hooke(Ef, νf, 1.0, small=true), 1)
                           end,
                           θ=0., ψ=0., ζ=0.,
                           maxiter::Int       = 4,
                           miniteru::Int      = 1,
                           maxiteru::Int      = 7,
                           dTolrnorm          = 1.5,
                           dTolu              = 5e-6,
                           maxiterd::Int      = 1,
                           dTold              = 1e-6,
                           nSteps::Int        = 200,
                           sPath::String      = pwd(),
                           sModelPath::String = joinpath(sPath, "mesh_files"),
                           sJLD2Path::String  = joinpath(sPath, "jld2_files"),
                           svtkPath::String   = joinpath(sPath, "vtk_files"),
                           iSaveVTK           = nSteps÷10,
                           iSavejld2          = iSaveVTK,
                           busePardiso::Bool  = true,
                           λT                 = 1e-2,
                           sPost::String      = "" )
  # Ensure output directories exist
  mkpath(svtkPath)
  mkpath(sJLD2Path)

  # Initialize simulation
  nDoFs         = 3
  sFileName     = sModelName * sPost
  svtkFileName  = joinpath(svtkPath, sFileName)
  datestart     = Dates.now()
  tinit         = Base.time_ns()

  # Print simulation information
  println("Commencing simulation: \n\t$sFileName\n")
  println("Start time: $datestart \n")
  @show ϵM0

  # Create ParaView collection for time series output
  pvd = if isnan(iSaveVTK) 
    nothing
  else
    println("Writing animation to \n\t$svtkFileName.pvd")
    paraview_collection(svtkFileName)
  end

  # Load and prepare the model with periodic boundary conditions
  println("\nLoading model and setting up periodic boundary conditions...\n")
  nodes, elemsets, 
  B0, Bϵ, B0d, ai = make_the_3Dmodel(joinpath(sModelPath, sModelName), 
                                     tablets_mat, matrix_mat, θ, ψ, ζ)

  # Display material properties
  println("\nMatrix material properties:")
  dump(matrix_mat)
  println("\nTablet material properties:")
  dump(tablets_mat)

  # Display model dimensions and constraint matrix sizes
  println("\n number of nodes          : ", length(nodes), 
          "\n number of tablet elements: ", length(elemsets[1]), 
          "\n number of matrix elements: ", length(elemsets[2]))

  # Calculate volumes of tablet and matrix regions
  Vol = let
    Vtablets = sum(item.V for item in elemsets[1])
    Vmatrix = sum(item.V for item in elemsets[2])
    (Vtablets, Vmatrix)
  end
  @show Vol

  # Set up constraint matrices and degrees of freedom
  BB1     = hcat(B0, Bϵ)
  nNodes  = length(nodes)
  nDoFs0  = size(B0, 2)

  # Identify constrained and free degrees of freedom
  bϵfree  = isnan.(ϵM0)
  iϵcnst  = findall(.!bϵfree)
  bfree   = vcat(trues(nDoFs0), bϵfree)
  ifree   = findall(bfree)
  icnst   = findall(.!bfree)

  # Initialize displacement and damage fields
  u   = zeros(nDoFs, nNodes)
  ru  = zeros(nDoFs, nNodes)
  u1  = zeros(size(BB1, 2))
  d   = (zeros(1, nNodes), zeros(1, nNodes))
  d0  = (zeros(size(B0d[1], 2)), zeros(size(B0d[2], 2)))

  # Store previous states for rollback in case of failure
  uold = copy(u)
  dold = deepcopy(d)

  # Initialize result storage
  nans(i...) = fill(NaN, i...)
  ϵM    = nans(length(ϵM0), nSteps + 1)
  σM    = nans(length(ϵM0), nSteps + 1)
  Vd    = (nans(nSteps + 1), nans(nSteps + 1))
  steps = []

  # Create mass matrix for stabilization
  println("\nAssembling mass matrix...")
  MM = let
    _, _, MM1 = getT(elemsets[1], zeros(size(u)))
    _, _, MM2 = getT(elemsets[2], zeros(size(u)))
    MM1 + MM2
  end

  # Initialize PARDISO solver if requested
  if busePardiso
    println("Initializing PARDISO solver...")
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    pardisoinit(ps)
    fix_iparm!(ps, :N)
  end

  println("\nStarting simulation loop...\n")

  # Main simulation loop
  iStep = 0
  try
    while iStep ≤ nSteps
      # Store previous state for potential rollback
      uold = copy(u)
      dold = deepcopy(d)

      t0 = Base.time_ns()
      LF = iStep / nSteps  # Load factor

      # Apply boundary conditions
      u1[icnst] = ϵM0[iϵcnst] * LF
      u[:] = BB1 * u1

      @printf("Step: %3i/%-4i, Load factor: %.4f \n", iStep, nSteps, LF)
      flush(stdout)

      # Solve displacement field
      if busePardiso
        updtu!(ps, u1, elemsets, u, d, ru, BB1, MM, λT, ifree, maxiteru)
      else
        updtu!(u1, elemsets, u, d, ru, BB1, MM, λT, ifree, maxiteru)
      end

      # Solve damage field (staggered scheme)
      for ii = 1:2
        oldnorm = Inf
        for iterd = 0:maxiterd
          res, updt = updtd!(elemsets[ii], u, d[ii], B0d[ii], d0[ii])
          normres   = maximum(abs.(res), init = 0)
          normupdt  = maximum(abs.(updt), init = 0)

          @printf(" Material: %i, Damage iter: %02i, Residual norm: %.3e, Update norm: %.3e\n",
                  ii, iterd, normres, normupdt)

          # Check convergence criteria
          (oldnorm / normres < dTolrnorm || normres < dTold || normupdt < dTold) && break
          oldnorm = normres
        end
      end

      # Final displacement update after damage update
      if busePardiso
        updtu!(ps, u1, elemsets, u, d, ru, BB1, MM, λT, ifree, maxiteru)
      else
        updtu!(u1, elemsets, u, d, ru, BB1, MM, λT, ifree, maxiteru)
      end

      # Store results
      σM[:, iStep + 1] = transpose(Bϵ) * ru[:] / sum(Vol)
      ϵM[:, iStep + 1] = u1[nDoFs0 + 1:end]
      Vd[1][iStep + 1] = getVd(elemsets[1], d[1]) / Vol[1]
      Vd[2][iStep + 1] = getVd(elemsets[2], d[2]) / Vol[2]

      # Print current state
      println()
      println(" σM = [", join(map(x -> @sprintf("% .3f", x), σM[:, iStep + 1]), ","), "]")
      println(" ϵM = [", join(map(x -> @sprintf("% .3f", x), 100ϵM[:, iStep + 1]), ","), "]×10⁻²")

      # Save visualization data
      if iSaveVTK>0 && iStep % iSaveVTK == 0
        allJs = map(elems -> map(elem -> detJ(elem, u[:, elem.nodes]), elems), elemsets)
        elemspropsets = (Dict("J" => allJs[1]), Dict("J" => allJs[2]))
        nodespropsets = (Dict("u" => u, "d" => d[1]), Dict("u" => u, "d" => d[2]))

        paraview_collection(svtkFileName; append=true) do pvd
          svtmFileName = writeVTKstate(svtkFileName, nodes, elemsets,
                                       nodespropsets, elemspropsets, u,
                                       pvd = pvd, iStep = iStep,
                                       bdef = true, bcenter = true)
          println("\nState written to: \n\t", svtmFileName)
        end
      end

      # Save binary data
      if iSavejld2>0 && iStep % iSavejld2 == 0
        push!(steps, Dict("iStep"=>iStep, "u"=>copy(u), "d"=>deepcopy(d)))
        println("State $iStep saved")
      end

      # Calculate and display timing information
      tnow = Base.time_ns()
      Δt = (tnow - t0) / 1e9
      eltime = (tnow - tinit) / 1e9
      println("Step $iStep/$nSteps completed in ", round(Δt / 60, digits = 2), " mins., ",
              "ELT ", secs2hms(eltime), ", ",
              "ETA ", secs2hms((nSteps - iStep) * Δt), "\n")
      iStep += 1

      flush(stdout)
    end
  catch e
    # Handle errors and roll back to previous state
    u     = uold
    d[1] .= dold[1]
    d[2] .= dold[2]

    error_msg = sprint(showerror, e)
    flush(stdout)
    st = sprint((io, v) -> show(io, "text/plain", v),
                stacktrace(catch_backtrace()))
    @warn "Simulation error:\n$(error_msg)\n$(st)"
    println("Quitting.")
    flush(stdout)

    # Save state before exiting
    if pvd != nothing
      allJs = map(elems -> map(elem -> detJ(elem, uold[:, elem.nodes]), elems), elemsets)
      elemspropsets = (Dict("J" => allJs[1]), Dict("J" => allJs[2]))
      nodespropsets = (Dict("u" => uold, "d" => dold[1]),
                       Dict("u" => uold, "d" => dold[2]))
      paraview_collection(svtkFileName; append=true) do pvd
        svtmFileName = writeVTKstate(svtkFileName, nodes, elemsets,
                                     nodespropsets, elemspropsets, u,
                                     pvd = pvd, iStep = iStep,
                                     bdef = true, bcenter = true)
        println("\nState written to: \n\t", svtmFileName)
      end
    end
    push!(steps, Dict("iStep"=>iStep, "u"=>copy(uold), "d"=>deepcopy(dold)))
  end

  # Final output
  let
    allJs = map(elems -> map(elem -> detJ(elem, u[:, elem.nodes]), elems), elemsets)
    elemspropsets = (Dict("J" => allJs[1]), Dict("J" => allJs[2]))
    nodespropsets = (Dict("u" => u, "d" => d[1]), Dict("u" => u, "d" => d[2]))
    svtmFileName = writeVTKstate(svtkFileName, nodes, elemsets,
                                 nodespropsets, elemspropsets, u,
                                 pvd = nothing, iStep = iStep,
                                 bdef = true, bcenter = true)
  end

  # Save ParaView collection
  if pvd != nothing
    println("\nAnimation written to \n\t$svtkFileName.pvd")
  end

  # Prepare results dictionary
  savedvars = Dict(
                   "ϵM" => ϵM[:, 1:iStep],
                   "σM" => σM[:, 1:iStep],
                   "steps" => steps,
                   "ϵM0" => ϵM0,
                   "sFileName" => sFileName,
                   "sModelName" => sModelName,
                   "Vol" => Vol,
                   "Vd" => (Vd[1][1:iStep], Vd[2][1:iStep]),
                   "θ" => θ,
                   "ψ" => ψ,
                   "ζ" => ζ,
                   "ai" => ai,
                   "iStep" => iStep,
                   "nSteps" => nSteps,
                   "sPost" => sPost,
                   "matrix_mat" => matrix_mat,
                   "tablets_mat" => tablets_mat,
                   "datestart" => string(datestart)
                  )

  # Save results to JLD2 file
  @time let sFileName = joinpath(sJLD2Path, sFileName * ".jld2")
    println("\nSaving binary data to \n\t", sFileName, " ...")
    FileIO.save(sFileName, savedvars)
  end

  # Display completion message
  Δt = (Base.time_ns() - tinit) / 1e9
  println("\nFinished on ", Dates.now(),
          ", Total execution time: ", secs2hms(Δt))

  flush(stdout)
  return savedvars
end

"""
    makerKt!(Φ::Vector{<:adiff.D2}, elems::Vector{<:CEElem}, r)

Assemble tangent stiffness matrix from element contributions.

# Arguments
- `Φ`: Vector of element energy functions
- `elems`: Array of elements
- `r`: Residual vector

# Returns
- Assembled tangent stiffness matrix
"""
function makerKt!(Φ::Vector{<:adiff.D2}, elems::Vector{<:C3DP}, r)

  @assert length(Φ)==length(elems) "length(Φ)!=length(elems)"

  Nt = 0
  for ϕ in Φ
    Nt += length(ϕ.g.v)*length(ϕ.g.v)
  end

  II   = zeros(Int, Nt)
  JJ   = zeros(Int, Nt)  
  Kt   = zeros(Nt)  
  idxs = LinearIndices(r)

  N1 = 1
  for (ii,elem) in enumerate(elems)    
    idxii     = idxs[:, elem.nodes][:]    
    r[idxii] += adiff.grad(Φ[ii]) 
    nii       = length(idxii)
    Nii       = nii*nii
    oneii     = ones(nii)
    idd       = N1:N1+Nii-1
    II[idd]   = idxii * transpose(oneii)
    JJ[idd]   = oneii * transpose(idxii)
    Kt[idd]   = adiff.hess(Φ[ii])
    N1       += Nii
  end

  nDoFstot = length(r)
  Kt = dropzeros(sparse(II,JJ,Kt,nDoFstot,nDoFstot))

  return Kt
end
"""
    makerKtu!(elems::Vector{<:C3DP}, u::Array, d::Array, ru; dmax=0.99)

Assemble displacement tangent stiffness matrix with damage threshold.

# Arguments
- `elems`: Array of elements
- `u`: Displacement field
- `d`: Damage field
- `ru`: Residual vector

# Keywords
- `dmax`: Maximum damage value for inclusion

# Returns
- Tangent stiffness matrix
"""
function makerKtu!(elems::Vector{<:C3DP}, 
                   u::Array, d::Array, ru; dmax=0.99)
  Φu    = Vector{adiff.D2}(undef, length(elems))
  allds = map(elem->getd(elem, d[elem.nodes]), elems)
  allJs = map(elem->detJ(elem, u[:,elem.nodes]), elems)
  idiis = findall(allJs.>0 .&& allds.<dmax)

  Threads.@threads for ii=idiis
    elem   = elems[ii]
    Φu[ii] = getϕ(elem, adiff.D2(u[:,elem.nodes]), d[elem.nodes])
  end

  makerKt!(Φu[idiis], elems[idiis], ru)
end
function makerKtu!(elemsets::NTuple{N,Vector}, u, 
                   d::NTuple{N,Matrix}, ru;
                   dmax=0.99) where N
  Φu    = map(elems->Vector{adiff.D2}(undef, length(elems)), elemsets)
  idiis = map(elems->trues(length(elems)), elemsets)

  for jj in 1:N
    allds = map(elem->getd(elem, d[jj][elem.nodes]),elemsets[jj])
    allJs = map(elem->detJ(elem, u[:,elem.nodes]),  elemsets[jj])
    idiis[jj][:] = allJs.>0 .&& allds.<dmax
  end

  for jj in 1:N
    Threads.@threads for ii=findall(idiis[jj])
      elem       = elemsets[jj][ii]
      Φu[jj][ii] = getϕ(elem, adiff.D2(u[:,elem.nodes]), d[jj][elem.nodes])
    end
  end

  makerKt!(Φu[1][idiis[1]], elemsets[1][idiis[1]], ru) + 
  makerKt!(Φu[2][idiis[2]], elemsets[2][idiis[2]], ru) 
end
"""
    maker!(Φ::Vector{T}, elems, r) where T<:adiff.D1

Assemble residual vector from element contributions.

# Arguments
- `Φ`: Vector of element energy functions
- `elems`: Array of elements
- `r`: Residual vector (modified in-place)

# Returns
- Modified residual vector
"""
function maker!(Φ::Vector{T} where T<:adiff.D1, elems, r)

  idx = LinearIndices(r)
  for ii in axes(elems, 1)    
    idxii     = idx[:, elems[ii].nodes][:]    
    r[idxii] += adiff.grad(Φ[ii]) 
  end
  r
end
"""
    makeru!(elems::Vector{<:C3DP}, u::Array, d::Array, r; dmax=0.99)

Assemble displacement residual vector with damage threshold.

# Arguments
- `elems`: Array of elements
- `u`: Displacement field
- `d`: Damage field
- `r`: Residual vector (modified in-place)

# Keywords
- `dmax`: Maximum damage value for inclusion

# Returns
- Modified residual vector
"""
function makeru!(elemsets::NTuple{N,Vector}, u, 
                 d::NTuple{N,Matrix}, r,
                 dmax=0.99) where N
  Φu    = map(elems->Vector{adiff.D1}(undef, length(elems)), elemsets)
  idiis = map(elems->trues(length(elems)), elemsets)

  for jj in 1:N
    allds = map(elem->getd(elem, d[jj][elem.nodes]),elemsets[jj])
    allJs = map(elem->detJ(elem, u[:,elem.nodes]),  elemsets[jj])
    idiis[jj][:] = allJs.>0 .&& allds.<dmax
  end

  for jj in 1:N
    Threads.@threads for ii=findall(idiis[jj])
      elem       = elemsets[jj][ii]
      Φu[jj][ii] = getϕ(elem, adiff.D1(u[:,elem.nodes]), d[jj][elem.nodes])
    end
  end

  maker!(Φu[1][idiis[1]], elemsets[1][idiis[1]], r)  
  maker!(Φu[2][idiis[2]], elemsets[2][idiis[2]], r) 
end
function makeru!(elems::Vector{<:C3DP}, u::Array, d::Array, r;
                 dmax=0.99)
  Φu    = Vector{adiff.D1}(undef, length(elems))
  allds = map(elem->getd(elem, d[elem.nodes]), elems)
  allJs = map(elem->detJ(elem, u[:,elem.nodes]), elems)
  idiis = findall(allJs.>0 .&& allds.<dmax)

  Threads.@threads for ii=idiis
    elem   = elems[ii]
    Φu[ii] = getϕ(elem, adiff.D1(u[:,elem.nodes]), d[elem.nodes])
  end

  maker!(Φu[idiis], elems[idiis], r)  
end
function updtd!(elems, u, d, B0d, d0)

  Φd    = Vector{adiff.D2}(undef, length(elems))
  allJs = map(elem->detJ(elem, u[:,elem.nodes]), elems)
  idiis = findall(allJs.>0)

  if isempty(idiis)
    return [], []
  end

  Threads.@threads  for ii=idiis
    Φd[ii] = getϕ(elems[ii], u[:,elems[ii].nodes], adiff.D2(d[elems[ii].nodes])) 
  end

  res  = zeros(size(d))
  Ktd  = makerKt!(Φd[idiis], elems[idiis], res)

  res  = transpose(B0d)*res[:]
  idd  = res[:] .< 0 

  if any(idd)
    Ktd      = transpose(B0d)*Ktd*B0d

    updt     = Ktd[idd,idd]\res[idd]
    d0[idd] -= updt 

    d[:]     = B0d*d0
    d[d.<0] .= 0
    d[d.>1] .= 1

    return res[idd], updt
  else
    return [], []
  end

end
"""
    updtu!(u1, elems, u, d, ru, BB1, MM, λT, ifree, maxiter; dTolu=1e-6, maxnormupdtu=1.0)

Update displacement field using Newton iteration.

# Arguments
- `u1`: Global DOF vector
- `elems`: Array of elements
- `u`: Displacement field
- `d`: Damage field
- `ru`: Residual vector
- `BB1`: Constraint matrix
- `MM`: Mass matrix
- `λT`: Stabilization parameter
- `ifree`: Free DOF indices
- `maxiter`: Maximum iterations

# Keywords
- `dTolu`: Displacement tolerance
- `maxnormupdtu`: Maximum update norm

# Returns
- Norm of update
"""
function updtu!(u1, elems, u, d, ru, 
                BB1, MM, λT, ifree, maxiter; 
                dTolu=1e-6, maxnormupdtu=1.0)

  iterupdt = (3maxiter)÷4
  oldnormr = Inf
  normupdt = 0.0
  updt     = zeros(size(ifree))

  t0       = Base.time_ns()
  Ktuff    = let 
    Ktu = makerKtu!(elems, u, d, fill!(ru,0))
    transpose(BB1[:,ifree])*(λT*MM+Ktu)*BB1[:,ifree]
  end

  iter     = 0
  while iter ≤ maxiter
    resu    = transpose(BB1[:,ifree])*ru[:]
    normr   = maximum(resu)-minimum(resu)

    if normr≤dTolu 
      @printf(" iteru: %02i, normresu: %.3e \n", iter, normr)
      break
    elseif normr≥oldnormr
      u1[ifree] += updt
      u[:]       = BB1*u1
      @printf("*normresu: %.3e, updating Ktu \n", normr/oldnormr)
      Ktuff    = let 
        Ktu = makerKtu!(elems, u, d, fill!(ru,0))
        transpose(BB1[:,ifree])*(λT*MM+Ktu)*BB1[:,ifree]
      end
    end
    oldnormr   = normr
    updt       = Ktuff\resu

    normupdt   = maximum(updt)-minimum(updt)
    if normupdt > maxnormupdtu
      return normupdt
    end
    u1[ifree] -= updt
    u[:]       = BB1*u1

    @printf(" iteru: %02i, normresu: %.3e, updt: %.3e, in %.3f sec. \n", 
            iter, normr, normupdt, (Base.time_ns()-t0)/1e9)
    t0         = Base.time_ns()
    makeru!(elems, u, d, fill!(ru, 0))

    flush(stdout)
    iter += 1
  end

  return normupdt
end
function updtu!(ps::MKLPardisoSolver, u1, elems, u, 
                d, ru, BB1, MM, λT, ifree, maxiter; 
                dTolu=1e-6, maxnormupdtu=1)

  iterupdt = (maxiter+1)÷2
  oldnormr = Inf
  normupdt  = 0.0
  updt     = zeros(size(ifree))

  t0       = Base.time_ns()
  Ktu      = makerKtu!(elems, u, d, fill!(ru,0))
  Ktuff    = transpose(BB1[:,ifree])*(λT*MM+Ktu)*BB1[:,ifree]
  resu     = transpose(BB1[:,ifree])*ru[:]

  _Ktuff   = get_matrix(ps, Ktuff, Pardiso.REAL_SYM_POSDEF)
  set_phase!(ps, Pardiso.ANALYSIS)
  pardiso(ps, _Ktuff, resu)
  set_phase!(ps, Pardiso.NUM_FACT)
  pardiso(ps, _Ktuff, resu)

  iter = 0
  while iter ≤ maxiter
    normr    = maximum(resu)-minimum(resu)
    if normr≤dTolu
      @printf(" iteru: %02i, normresu: %.3e \n", iter, normr)
      break
    elseif normr≥oldnormr
      u1[ifree] += updt
      u[:]       = BB1*u1

      @printf("*normresu: %.3e, updating Ktu \n", normr/oldnormr)
      Ktu      = makerKtu!(elems, u, d, fill!(ru,0))
      Ktuff    = transpose(BB1[:,ifree])*(λT*MM+Ktu)*BB1[:,ifree]
      resu     = transpose(BB1[:,ifree])*ru[:]

      _Ktuff   = get_matrix(ps, Ktuff, Pardiso.REAL_SYM_POSDEF)
      set_phase!(ps, Pardiso.ANALYSIS)
      pardiso(ps, _Ktuff, resu)
      set_phase!(ps, Pardiso.NUM_FACT)
      pardiso(ps, _Ktuff, resu)
    end
    oldnormr = normr

    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    pardiso(ps, updt, _Ktuff, resu)

    normupdt   = maximum(updt)-minimum(updt)
    if normupdt > maxnormupdtu
      return normupdt
    end

    u1[ifree] -= updt
    u[:]       = BB1*u1
    @printf(" iteru: %02i, normresu: %.3e, updt: %.3e, in %.3f sec. \n", 
            iter, normr, normupdt, (Base.time_ns()-t0)/1e9)
    t0        = Base.time_ns()
    makeru!(elems, u, d, fill!(ru, 0))
    resu      = transpose(BB1[:,ifree])*ru[:]
    iter +=1 
  end
  set_phase!(ps, Pardiso.RELEASE_ALL)
  pardiso(ps, updt, _Ktuff, resu)

  return normupdt
end
"""
    secs2hms(secs)

Convert seconds to hours:minutes:seconds format.

# Arguments
- `secs`: Time in seconds

# Returns
- Formatted time string (HH:MM:SS)
"""
function secs2hms(secs)
  h, r = divrem(secs, 3600)
  m, s = divrem(r, 60)
  @sprintf("%02i:%02i:%02i",h, m, s)
end
"""
    writeVTKstate(sFileName::String, nodes::Vector{Vector{D}}, elemsets::NTuple{N,Vector{E}}, nodespropsets::NTuple{N,Dict}, elemspropsets::NTuple{N,Dict}, u=[]; iStep=0, pvd=nothing, r0=zeros(size(nodes[1])), bdef=false, bcenter=true) where {N, D<:Number, E<:CPElem}

Write simulation state to VTK file for visualization.

# Arguments
- `sFileName`: Output filename
- `nodes`: Node coordinates
- `elemsets`: Tuple of element sets
- `nodespropsets`: Tuple of node property dictionaries
- `elemspropsets`: Tuple of element property dictionaries
- `u`: Displacement field (optional)

# Keywords
- `iStep`: Time step index
- `pvd`: ParaView data collection
- `r0`: Reference coordinates
- `bdef`: Write deformed configuration
- `bcenter`: Center displacements

# Returns
- Path to created VTK file
"""
function writeVTKstate(sFileName::String, 
                       nodes::Vector{Vector{D}} where D<:Number, 
                       elemsets::NTuple{N,Vector{E} where E<:CPElem},
                       nodespropsets::NTuple{N,Dict},
                       elemspropsets::NTuple{N,Dict},
                       u       = [];
                       iStep   = 0, 
                       pvd     = nothing, 
                       r0      = zeros(size(nodes[1])),
                       bdef    = false,
                       bcenter = true) where N
  cellType = VTKCellTypes.VTK_TETRA
  nNodes   = length(nodes)
  nDoFs    = length(nodes[1])

  points  = zeros(nDoFs, nNodes) 
  if bdef
    u_cg  = if bcenter
      sum([u[:,ii] for ii=1:nNodes])/nNodes
    else
      zeros(nDoFs)
    end
    for ii=1:nNodes
      points[:,ii] = nodes[ii]+r0+u[:,ii] - u_cg
    end
  else
    for ii=1:nNodes
      points[:,ii] = nodes[ii]+r0
    end
  end

  if pvd!==nothing
    sPost = @sprintf("_%03i", iStep)
  else
    sPost = ""
  end

  vtm = WriteVTK.vtk_multiblock(sFileName*sPost)
  for ii = 1:N
    sFNameii = @sprintf("%sb%02i%s", sFileName, ii, sPost)
    cells    = [WriteVTK.MeshCell(cellType, elem.nodes) for elem in elemsets[ii]]
    vtkobj   = WriteVTK.vtk_grid(vtm, sFNameii, points, cells)
    for spropname in keys(nodespropsets[ii])
      WriteVTK.vtk_point_data(vtkobj, nodespropsets[ii][spropname], spropname)
    end
    for spropname in keys(elemspropsets[ii])
      WriteVTK.vtk_cell_data(vtkobj, elemspropsets[ii][spropname], spropname)
    end

    WriteVTK.multiblock_add_block(vtm, vtkobj)
  end

  if pvd !== nothing
    WriteVTK.collection_add_timestep(pvd, vtm, iStep)
  end

  WriteVTK.vtk_save(vtm)  

  return vtm.path
end
#
# functions for making the two phases model with periodic b.c.
#
function make_the_3Dmodel(sModelName::String, 
                          tablets_mat::Material, matrix_mat::Material,
                          θ=0, ψ=0, ζ=0)

  nDoFs    = 3
  # rotation matrix for the model, the sign is reversed in order for θ to point
  # to the direction of the applied strain
  M = [1.0    0.0     0.0;
       0.0    cos(ψ)  sin(ψ); 
       0.0   -sin(ψ)  cos(ψ)] * 
  [cos(ζ)   0.0     sin(ζ); 
   0.0      1.0     0.0;
   -sin(ζ)  0.0     cos(ζ)] * 
  [ cos(θ)  sin(θ)  0.0;
   -sin(θ)  cos(θ)  0.0; 
   0.0      0.0     1.0] 

  println("\n Rotation matrix:")
  for ii=1:size(M,1)
    println("  |", join(map(x->@sprintf("% .3f", x), M[ii,:]), ", "), "|")
  end

  # load model from input file
  model = with_logger(Logging.NullLogger()) do
    AbaqusReader.abaqus_read_mesh(sModelName*".inp")
  end

  nNodes      = length(model["nodes"])
  nodes       = [ model["nodes"][ii][1:nDoFs] for ii in 1:nNodes ]
  elements    = model["elements"]
  node_sets   = model["node_sets"]
  elem_sets   = model["element_sets"]
  matrixnodes = node_sets["matrix"]

  (B0, B0d, Bϵ, (a1,a2,a3)) = make_B_matrices(model, nDoFs=nDoFs, M=M)
  #
  #   constructs elements
  #
  nodes, elemsets   = make_elements(model, tablets_mat, matrix_mat, M)

  println(" ... done")

  return nodes, elemsets, B0, Bϵ, B0d, (a1,a2,a3) 
end
function make_elements(model::Dict, 
                       tablts_mat::PhaseField,
                       matrix_mat::PhaseField,
                       M)

  nDoFs       = 3
  nodes       = [model["nodes"][ii][1:nDoFs] for ii = 1:length(model["nodes"])]
  elements    = model["elements"]
  node_sets   = model["node_sets"]
  elem_sets   = model["element_sets"]

  print("\n constructing elements ... "); flush(stdout)
  t0 = Base.time_ns()

  tablts_elems = Vector{C3DP}(undef,  length(elem_sets["tablets"]))
  Threads.@threads for ii in 1:length(elem_sets["tablets"])
    elem             = elem_sets["tablets"][ii]
    nodesid          = elements[elem]
    elnodes          = map(node->M*node, nodes[nodesid])
    tablts_elems[ii] = Elements.Tet04P(nodesid, elnodes, mat=tablts_mat) 
  end  

  matrix_elems = Vector{C3DP}(undef,  length(elem_sets["matrix"]))
  Threads.@threads for ii in 1:length(elem_sets["matrix"])
    elem             = elem_sets["matrix"][ii]
    nodesid          = elements[elem]
    elnodes          = map(node->M*node, nodes[nodesid])
    matrix_elems[ii] = Elements.Tet04P(nodesid, elnodes, mat=matrix_mat) 
  end 

  Δt = round((Base.time_ns()-t0)/1e9, digits=2)
  println(" done in $Δt sec. \n")

  @show typeof(tablts_elems)
  @show length(tablts_elems)
  @show typeof(matrix_elems)
  @show length(matrix_elems)

  return nodes, (tablts_elems, matrix_elems)
end
"""
    make_B_matrices(model, nDoFs=3, M=I)

Generate B matrices for periodic boundary conditions in 2D or 3D.

# Arguments
- `model`: Model dictionary containing nodes and node sets
- `nDoFs`: Number of degrees of freedom (2 for 2D, 3 for 3D)
- `M`: Transformation matrix to apply to periodicity vectors

# Returns
Tuple containing:
- B0: Constraint matrix for periodic boundary conditions
- B0d: Reduced constraint matrix
- Bϵ: Strain-displacement matrix for periodic boundary conditions
- Tuple of transformed periodicity vectors
"""
function make_B_matrices(model::Dict; M::AbstractMatrix=I, nDoFs::Int=3)
  @assert nDoFs ∈ (2, 3) "nDoFs must be 2 or 3, got $nDoFs"

  nNodes    = length(model["nodes"])
  nodes     = [model["nodes"][ii][1:nDoFs] for ii in 1:nNodes]
  node_sets = model["node_sets"]

  # Validate pair sets
  pair_sets = if nDoFs==2 
    [("left", "right"), ("bottom", "top")] 
  else
    [("left", "right"), ("bottom", "top"), ("front", "back")]
  end

  foreach(pair_sets) do (set1, set2)
    @assert length(node_sets[set1]) == length(node_sets[set2]) 
    "Mismatch between $set1 ($(length(node_sets[set1]))) and $set2 ($(length(node_sets[set2]))) nodes"
  end

  # Find pairs for all relevant directions
  pairs       = [find_pairs(nodes, node_sets[a], node_sets[b]) for (a, b) in pair_sets]
  a_vectors   = first.(pairs)
  pair_groups = last.(pairs)

  # Create BEqs and B matrices
  BEqs  = makeBEqs(vcat(pair_groups...), nNodes)
  B0    = dropzeros!(makeB0(BEqs, nDoFs=nDoFs))
  B0dm  = let
    BB = makeB0(pair_groups, node_sets["matrix"], nNodes, nDoFs=1)  
    Bm = make_Bm(node_sets["matrix"], nNodes, nDoFs=1)
    dropzeros!(Bm*BB)
  end
  B0dt  = let
    BB = makeB0(pair_groups, node_sets["tablets"], nNodes, nDoFs=1)  
    Bm = make_Bm(node_sets["tablets"], nNodes, nDoFs=1)
    dropzeros!(Bm*BB)
  end

  B0d = (B0dt,B0dm)

  # Transform a vectors
  a_vectors = [M * a for a in a_vectors]

  # Create Bϵ matrix
  Beps = if nDoFs == 2
    #     e₁₁, e₂₂, e₁₂
    a -> [a[1] 0 a[2];
          0 a[2] a[1]]
  else
    #     e₁₁, e₂₂, e₃₃, e₂₃, e₁₃, e₁₂
    a -> [a[1] 0 0 0 a[3] a[2];
          0 a[2] 0 a[3] 0 a[1];
          0 0 a[3] a[2] a[1] 0]
  end

  Ba  = makeBa(Tuple(pair_groups), nNodes)
  Bϵi = [Beps(a) for a in a_vectors]
  Bϵ  = dropzeros!(sparse(Ba * vcat(Bϵi...)))

  return (B0, B0d, Bϵ, Tuple(a_vectors))
end
function find_pairs(nodes, set1, set2; bchk = false, dTol=1e-12)
  N1,N2 = length(set1),length(set2)
  @assert N1==N2 @sprintf("length(set1)=%i!=%i=length(set2)",N1,N2)

  a = let
    cg1  = sum(nodes[set1])/N1
    cg2  = sum(nodes[set2])/N2
    cg2-cg1
  end

  pairs = Vector{Pair{Int64,Int64}}(undef, N1)
  for ii1 in 1:N1
    node1 = nodes[set1[ii1]]
    dd    = [norm(a + node1 - nodes[set2[jj]]) for jj in 1:N2]
    ii2   = argmin(dd)
    bchk && @assert dd[ii2] ≤ dTol @sprintf("mininum distance out of tolerance: %.3f ≰ %.3f",
                                            dd[ii2], dTol)
    pairs[ii1] = set1[ii1]=>set2[ii2]
  end

  a, pairs
end
function makeBEqs(all_pairs, nNodes, T=Float64)
  A = begin
    rmrow(A, ii) = A[1:size(A)[1] .!= ii, :]
    A = spzeros(T, size(all_pairs)[1], nNodes)
    for (ii, pair) = enumerate(all_pairs)
      A[ii, pair[1]] = A[ii, pair[2]] = 1
    end
    for ii=1:nNodes
      id_rows = sort(findall(A[:,ii].!=0))
      nrows   = length(id_rows)
      if nrows>1
        for jj=nrows:-1:2
          A[id_rows[1],:] = ((A[id_rows[1],:].==1) .| (A[id_rows[jj],:].==1))
          A = rmrow(A, id_rows[jj])
        end
      end
    end  
    A
  end
  B = begin
    nNodes = size(A)[2]
    q      = [sum(A[:,ii]) for ii=1:nNodes] 
    ifree  = findall(q .==0)
    nfree  = length(ifree)
    B      = spzeros(T, nfree, nNodes)
    for (ii,idx) in enumerate(ifree)
      B[ii, idx] = 1
    end  
    B
  end
  return vcat(A,B)
end
function makeB0(ufree::BitArray{N} where N, T)
  #   nDoFstot = length(ufree)
  N    = sum(ufree)
  idxx = findall(ufree[:])
  sparse(idxx, 1:N, ones(N), length(ufree), N)
end
function makeB0(BEqs::SparseMatrixCSC{TF,TI}; nDoFs=3) where {TF,TI}
  nEqs,nNodes = size(BEqs)
  nDoFstot    = nNodes*nDoFs 
  I           = zeros(TI, nDoFstot)
  J           = zeros(TI, nDoFstot)
  V           = zeros(TF, nDoFstot)

  for ii=1:nNodes
    iirows = (ii-1)*nDoFs
    for qq = BEqs[:,ii].nzind
      iicols = (qq-1)*nDoFs
      for jj = 1:nDoFs
        I[iirows+jj] = iirows+jj
        J[iirows+jj] = iicols+jj
        V[iirows+jj] = one(TF)
      end
    end
  end  
  sparse(I,J,V)
end
function makeB0(a_pairs::Vector{Vector{Pair{Int,Int}}}, 
                set_of_nodes::Vector{Int},
                nNodes::Int; nDoFs=3)

  glob2loc = zeros(Int, nNodes)
  for ii=1:nNodes
    glob2loc[ii] = ii ∈ set_of_nodes
  end

  loc_pairs = [glob2loc[item[1]]=>glob2loc[item[2]] 
               for item in vcat(a_pairs...)]
  bkeep     = [item != (0=>0) for item in loc_pairs]

  BEqs      = makeBEqs(loc_pairs[bkeep], length(set_of_nodes))
  B0        = dropzeros!(makeB0(BEqs, nDoFs=nDoFs))
  return B0
end
function makeBa(pairs, nNodes,
                T=Float64;
                nDoFsu=length(pairs),
                nDoFsω = 0)

  nDoFs = nDoFsu + nDoFsω
  ndirs = length(pairs)
  Ba    = spzeros(T, nNodes*nDoFs, nDoFsu*ndirs)

  for (jj, pairs) in enumerate(pairs)
    for pair in pairs
      iia = (jj-1)*nDoFsu
      ii1 = (pair[1]-1)*nDoFs
      ii2 = (pair[2]-1)*nDoFs
      for ii=1:nDoFsu
        Ba[ii1+ii, iia+ii] = -1/2
        Ba[ii2+ii, iia+ii] = 1/2
      end
    end
  end
  dropzeros!(Ba)
end
function make_Bm(mnodes, nNodes; nDoFs=3)
  II      = spdiagm(0=>ones(nDoFs))
  Nmnodes = length(mnodes)

  Bm = spzeros(nNodes*nDoFs, Nmnodes*nDoFs)
  for (ii,item) in enumerate(mnodes)
    icols   = (ii-1)*nDoFs + 1
    irow    = (item-1)*nDoFs + 1

    Bm[irow:irow+nDoFs-1, icols:icols+nDoFs-1] = II
  end
  dropzeros!(Bm)
end
# function for inertia and mass matrices
function getT(elem::C3DP{P,M} where M,
              udot0::Matrix{T}) where {T,P}
  ϕ   = zero(T) 
  for ii=1:P
    N0 = elem.N[ii]
    d  = [N0⋅udot0[1:3:end], N0⋅udot0[2:3:end], N0⋅udot0[3:3:end]]
    ϕ += elem.mat.mat.ρ*elem.wgt[ii]* (d⋅d)
  end
  ϕ
end
function getT(elems::Vector{C3DP}, 
              udot::Matrix{T}) where T
  nElems = length(elems)

  Φ = Vector{adiff.D2}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getT(elems[ii], adiff.D2(udot[:,elems[ii].nodes]))
  end

  makeϕrKt(Φ, elems, udot)
end
function getd(elem::C3DP{P}, d0::Vector{T}) where {P,T}
  d       = zero(T)
  # N0, wgt = elem.N0, elem.wgt
  for ii=1:P
    d += elem.wgt[ii]*(elem.N[ii]⋅d0)
  end  
  d/elem.V
end

end # module NacreDualDamage
