#### FEA Code to carry out Linear stability analysis on Laminar flames using Thermo-diffusive model###################
###This code reads in baseflow (initial guess of stationary solution from a file and finds accurate stationary solution suing newtons Raphson method
#then calculates eigenvalues and eigenfunction and write them to disk
#Required libraries, dolfinx-real, numpy, pandas, mpi4py slepc4py, petsc4py, scipy 
#Requires initial guess for stationary solutioonfor proper convergence of Newton's method. This has been obtained by Selective frequency Damping code.
# This code can be run in parallel using MPI
#use mpirun -np [no of processes] python3 code.py

#!/usr/bin/env python
# coding: utf-8

import sys
import os
import typing
import basix.ufl
import dolfinx.fem
import dolfinx.fem.petsc
import numpy.typing
import petsc4py.PETSc
import scipy.special
import slepc4py.SLEPc
import ufl
import dolfinx.mesh
from mpi4py import MPI
from dolfinx.io import (VTXWriter, gmshio)
import numpy as np
gdim = 2
mesh_comm = MPI.COMM_WORLD
dtype0=np.float64
print("using dolfinx ",dolfinx.__version__," and \n data type",
      [dolfinx.default_scalar_type,petsc4py.PETSc.ScalarType])
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import log

import multiphenicsx.fem
import multiphenicsx.fem.petsc

Print = petsc4py.PETSc.Sys.Print


##initial guess for base flow loading and defining functions
##load base flows create a interpolation function using scipy library
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
def init(zeta,dam):
    file='./baseflows/le1.6_all/BF2D_Le1.6_phi1_z'+str(zeta)+'_da'+str(dam)+'.dat'
    # load full data file
    df = pd.read_csv(file, sep=r'\s+',header=None)
    ##define x and y and dependent variables
    x = np.linspace(0, 15, 241)
    y = np.linspace(-15, 15, 481)
    matrix = df.values
    #extract values of species concentration and temperature
    temp=matrix[:,2].reshape(481,241)
    fuel=matrix[:,3].reshape(481,241)
    oxy=matrix[:,4].reshape(481,241)
    #interpolate
    Yt_base = RegularGridInterpolator((x,y), np.transpose(temp),bounds_error=False,fill_value=None)
    Yf_base = RegularGridInterpolator((x, y), np.transpose(fuel),bounds_error=False,fill_value=None)
    Yx_base = RegularGridInterpolator((x, y), np.transpose(oxy),bounds_error=False,fill_value=None)
    # Read the inlet porfile of species
    df = pd.read_csv('./baseflows/inlets/inlet_'+str(zeta)+'.dat', sep=r'\s+',header=None)
    matrix = df.values;
    dat=matrix.flatten();
    #interpolate inlet profile
    y = np.linspace(-15, 15, 481)
    from scipy import interpolate as interpolate1
    yf_inlet = interpolate1.interp1d(y, dat,fill_value="extrapolate")
    yx_inlet = interpolate1.interp1d(y, 1.0-dat,fill_value="extrapolate")  ##due to equal Lewis number case 
    return Yt_base,Yf_base,Yx_base,yf_inlet,yx_inlet
    
#######################
##convert python functions to functions which would receive numpy type arguments
def yf_base_eval(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[  # type: ignore[no-any-unimported]
        petsc4py.PETSc.ScalarType]:
    """Return the interpolated fuel mass fraction profile at the inlet."""
    values = np.zeros((x.shape[1]))
    values[:] = Yf_base(np.transpose([x[0],x[1]]))
    return values
def yx_base_eval(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[  # type: ignore[no-any-unimported]
        petsc4py.PETSc.ScalarType]:
    """Return the interpolated  oxy mass fraction profile at the inlet."""
    values = np.zeros((x.shape[1]))
    values[:] = Yx_base(np.transpose([x[0],x[1]]))
    return values
def yt_base_eval(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[  # type: ignore[no-any-unimported]
        petsc4py.PETSc.ScalarType]:
    """Return the interpolated Temperature profile at the inlet."""
    values = np.zeros((x.shape[1]))
    values[:] = Yt_base(np.transpose([x[0],x[1]]))
    return values
#def func for interpolation in boundary conditions for base flow variables
##using bc for baseflow computation using Newtons method
def yf_in_eval(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[  # type: ignore[no-any-unimported]
        petsc4py.PETSc.ScalarType]:
    values = np.zeros((x.shape[1]))
    values[:] = yf_inlet(x[1])
    return values
def yx_in_eval(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[  # type: ignore[no-any-unimported]
        petsc4py.PETSc.ScalarType]:
    values = np.zeros((x.shape[1]))
    values[:] = yx_inlet(x[1])
    return values
def yt_in_eval(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[  # type: ignore[no-any-unimported]
        petsc4py.PETSc.ScalarType]:
    values = np.zeros((x.shape[1]))
    return values

## define all necessary paramters
LeF=petsc4py.PETSc.ScalarType(1.6)
LeX=petsc4py.PETSc.ScalarType(1.6)
#Da=petsc4py.PETSc.ScalarType(100.0)
beta=petsc4py.PETSc.ScalarType(10.0)
gamma=petsc4py.PETSc.ScalarType(5.0)
phi=petsc4py.PETSc.ScalarType(1.0)
InvLeF=1.0/LeF
InvLeX=1.0/LeX


##defines the mesh problem operators with governing equations and boundary conditions
##takes inputs
# sizeno:  (hx=hy=1/mesh_sizeno units .. eg. size=16, means hx=hy=1/16)
# porder: order of finite element
#  output: mesh, functionspaces[YF,YX,YT], operators with bc [F,J,rhs,bc]
def problem(mesh_sizeno,porder,dam):
    Da=petsc4py.PETSc.ScalarType(dam)
    ihx=mesh_sizeno;
    sx=15*ihx;
    sy=2*sx;
    #rectangular mesh
    mesh=dolfinx.mesh.create_rectangle(mesh_comm, np.array([[0,-15.0],[15,15]]),np.array([sx,sy]),\
                             cell_type=dolfinx.mesh.CellType.quadrilateral,dtype=dtype0)
    # Function spaces
    #use Lagrange element
    Vt_element = basix.ufl.element("Lagrange", mesh.basix_cell(), porder)
    ##function spaces
    YF = dolfinx.fem.functionspace(mesh, Vt_element)
    YX=YF.clone()
    YT=YF.clone()
    #functions
    (yf,yx,yt)=(dolfinx.fem.Function(YF,dtype=dtype0),dolfinx.fem.Function(YF,dtype=dtype0),dolfinx.fem.Function(YF,dtype=dtype0))
    (vf, vx, vt) = (ufl.TestFunction(YF),ufl.TestFunction(YX),ufl.TestFunction(YT))
    (yfb,yxb,ytb) = (ufl.TrialFunction(YF),ufl.TrialFunction(YX),ufl.TrialFunction(YT))
    ##interpolate the base_flow onto the dolfinx function (initial guess)
    yf.interpolate(yf_base_eval)
    yx.interpolate(yx_base_eval)
    yt.interpolate(yt_base_eval)
    #governing equations in variational form
    w=Da*(beta**3)*yf*yx*ufl.exp(beta*(yt-1)*(1+gamma)/(1+gamma*yt));
    F=[(ufl.inner(yf.dx(0),vf)+InvLeF*ufl.inner(ufl.grad(yf),ufl.grad(vf))+ufl.inner(w,vf))*ufl.dx,\
       (ufl.inner(yx.dx(0),vx)+InvLeX*ufl.inner(ufl.grad(yx),ufl.grad(vx))+phi*ufl.inner(w,vx))*ufl.dx,\
       (ufl.inner(yt.dx(0),vt)+ufl.inner(ufl.grad(yt),ufl.grad(vt))-(1+phi)*ufl.inner(w,vt))*ufl.dx];
    #Jacobian 
    J = [[ufl.derivative(F[0], yf, yfb), ufl.derivative(F[0], yx, yxb),ufl.derivative(F[0], yt, ytb)],\
        [ufl.derivative(F[1], yf, yfb), ufl.derivative(F[1], yx, yxb),ufl.derivative(F[1], yt, ytb)],\
         [ufl.derivative(F[2], yf, yfb), ufl.derivative(F[2], yx, yxb),ufl.derivative(F[2], yt, ytb)]];
    #boundary conditions
    ##define x=0 inlet bc
    def inlet_bc(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
        return np.isclose(x[0], 0)
    boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, inlet_bc)
    #degree of freedom for boundary facets
    bdofs_YF = dolfinx.fem.locate_dofs_topological(YF, mesh.topology.dim - 1, boundary_facets);
    bdofs_YX = dolfinx.fem.locate_dofs_topological(YX, mesh.topology.dim - 1, boundary_facets);
    bdofs_YT = dolfinx.fem.locate_dofs_topological(YT, mesh.topology.dim - 1, boundary_facets);
    # Set the Boundary conditions
    #fuel
    yf_in=dolfinx.fem.Function(YF,dtype=dtype0)
    yf_in.interpolate(yf_in_eval)
    bcf=dolfinx.fem.dirichletbc(yf_in, bdofs_YF)
    #oxy
    yx_in=dolfinx.fem.Function(YX,dtype=dtype0)
    yx_in.interpolate(yx_in_eval)
    bcx=dolfinx.fem.dirichletbc(yx_in, bdofs_YX)
    #temp
    yt_in=dolfinx.fem.Function(YT,dtype=dtype0)
    yt_in.interpolate(yt_in_eval)
    bct=dolfinx.fem.dirichletbc(yt_in, bdofs_YT)
    #combined bcs
    bc=[bcf,bcx,bct] 
    return mesh,YF,YX,YT,F,J,bc,(yf,yx,yt)

#define functions for setting, solving and monitoring results from  Nonlinear solver: SNES 
# Class for interfacing with the SNES
class NonlinearBlockProblem:
    """Define a nonlinear problem, interfacing with SNES."""
    """Define a nonlinear problem, interfacing with SNES."""
    def __init__(  # type: ignore[no-any-unimported]
            self, F: list[ufl.Form], J: list[list[ufl.Form]],
            solutions: tuple[dolfinx.fem.Function, dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC],
        P: typing.Optional[list[list[ufl.Form]]] = None) -> None:
        self._F = dolfinx.fem.form(F)
        self._J = dolfinx.fem.form(J)
        self._obj_vec = dolfinx.fem.petsc.create_vector_block(self._F)
        self._solutions = solutions
        self._bcs = bcs
        self._P = P

    def create_snes_solution(self) -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
        """
        Create a petsc4py.PETSc.Vec to be passed to petsc4py.PETSc.SNES.solve.
        The returned vector will be initialized with the initial guesses provided in `self._solutions`,
        properly stacked together in a single block vector.
        """
        x = dolfinx.fem.petsc.create_vector_block(self._F)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(x, [YF.dofmap, YX.dofmap,YT.dofmap]) as x_wrapper:
            for x_wrapper_local, sub_solution in zip(x_wrapper, self._solutions):
                with sub_solution.x.petsc_vec.localForm() as sub_solution_local:
                    x_wrapper_local[:] = sub_solution_local
        return x

    def update_solutions(self, x: petsc4py.PETSc.Vec) -> None:  # type: ignore[no-any-unimported]
        """Update `self._solutions` with data in `x`."""
        x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(x, [YF.dofmap, YX.dofmap,YT.dofmap]) as x_wrapper:
            for x_wrapper_local, sub_solution in zip(x_wrapper, self._solutions):
                with sub_solution.x.petsc_vec.localForm() as sub_solution_local:
                    sub_solution_local[:] = x_wrapper_local

    def obj(  # type: ignore[no-any-unimported]
        self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec
    ) -> np.float64:
        """Compute the norm of the residual."""
        self.F(snes, x, self._obj_vec)
        return self._obj_vec.norm()  # type: ignore[no-any-return]

    def F(  # type: ignore[no-any-unimported]
            self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, F_vec: petsc4py.PETSc.Vec
    ) -> None:
        """Assemble the residual."""
        self.update_solutions(x)
        with F_vec.localForm() as F_vec_local:
            F_vec_local.set(0.0)
            dolfinx.fem.petsc.assemble_vector_block(  # type: ignore[misc]
            F_vec, self._F, self._J, self._bcs, x0=x, alpha=-1.0)

    def J(  # type: ignore[no-any-unimported]
        self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, J_mat: petsc4py.PETSc.Mat,
        P_mat: petsc4py.PETSc.Mat
    ) -> None:
        """Assemble the jacobian."""
        J_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(  # type: ignore[misc]
            J_mat, self._J, self._bcs, diagonal=1.0)  # type: ignore[arg-type]
        J_mat.assemble()
        if self._P is not None:
            P_mat.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(  # type: ignore[misc]
                P_mat, self._P, self._bcs, diagonal=1.0)  # type: ignore[arg-type]
            P_mat.assemble()
# Create problem
zeta=0.7
dam=100
#change mesh size by imesh, dx=dy=1.0/imesh and order of element: porder 
imesh=8
porder=1
Yt_base,Yf_base,Yx_base,yf_inlet,yx_inlet=init(zeta,dam)
mesh,YF,YX,YT,F,J,bc,(yf,yx,yt)=problem(imesh,porder,dam)
sol = NonlinearBlockProblem(F, J, (yf, yx, yt), bc)
F_vec = dolfinx.fem.petsc.create_vector_block(sol._F)
J_mat = dolfinx.fem.petsc.create_matrix_block(sol._J)


start_time = MPI.Wtime()


# Solve for base flow
snes = petsc4py.PETSc.SNES().create(mesh.comm)
snes.setType("newtonls")

###Set paramters for Linear Solver
#snes.setTolerances(max_it=20)
snes.setTolerances(rtol=1.0e-4, atol=1.0e-4, stol=1.0e-4, max_it=20)
#snes.getKSP().setType("bcgs")
snes.getKSP().setType("gmres")
#snes.getKSP().getPC().setType("gamg")

snes.getKSP().getPC().setType("hypre")
snes.getKSP().getPC().setHYPREType("boomeramg")
print("max failures allowed default",snes.getMaxKSPFailures())

snes.setMaxKSPFailures(5)
##paramters for exact solver (Warning: May take long time for solution for fine meshes
#snes.getKSP().setType("preonly")
#snes.getKSP().getPC().setType("lu")
#snes.getKSP().getPC().setFactorSolverType("mumps")
snes.setObjective(sol.obj)
snes.setFunction(sol.F, F_vec)
snes.setJacobian(sol.J, J=J_mat, P=None)

# Set the monitor function with F_vec as an argument
def monitor(F_vec, snes, it, norm):
    # Compute the infinity norm of the residual using F_vec
    rnorm_inf = F_vec.norm(petsc4py.PETSc.NormType.NORM_INFINITY)
    print(f"Iteration {it}: Infinity norm of residual = {rnorm_inf}")
snes.setMonitor(lambda snes, it, norm: monitor(F_vec, snes, it, norm))

solution = sol.create_snes_solution()
if mesh_comm.rank==0:
    print("solution vector of size ",solution.size," created successfully")

fnorm=mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(F[0])), op=MPI.SUM)
xnorm=mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(F[1])), op=MPI.SUM)
tnorm=mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(F[2])), op=MPI.SUM)
mesh_comm.Barrier()

if mesh_comm.rank==0:
    print("intial norm of solution is ")
    print("norms of [fuel,oxy,temp] =", [fnorm,xnorm,tnorm])
mesh_comm.Barrier()

num_dofs_global=0
num_dofs_local=0
for Y in [YF,YX,YT]:
    num_dofs_local=num_dofs_local+ (Y.dofmap.index_map.size_local) * Y.dofmap.index_map_bs
    num_dofs_global=num_dofs_global+ Y.dofmap.index_map.size_global * Y.dofmap.index_map_bs
print(f"Number of dofs (owned) by rank {mesh_comm.rank}: {num_dofs_local}")
mesh_comm.Barrier()
print(f"Number of dofs global: {num_dofs_global}")

##execute the Newton solver
snes.solve(None, solution)
mesh_comm.Barrier()
if mesh_comm.rank==0:
# Get the convergence reason
    converged_reason = snes.getConvergedReason()
    print("Converged Reason Code:", converged_reason)

    # Interpret the convergence reason
    if converged_reason > 0:
        print("Converged Reason: Positive value indicates convergence (e.g., CONVERGED_RTOL, CONVERGED_ITS, etc.)")
    elif converged_reason < 0:
        print("Converged Reason: Negative value indicates divergence (e.g., DIVERGED_MAX_IT, DIVERGED_LINEAR_SOLVE, etc.)")
    else:
        print("Converged Reason: Solver has not yet converged or diverged.")
    # Get the tolerances
    rtol, atol, stol, max_it = snes.getTolerances()
    print("Tolerances: rtol={}, atol={}, stol={}, max_it={}".format(rtol, atol, stol, max_it))
if mesh_comm.rank==0:
   if converged_reason < 0:
      print("Newton's code diverged so exitting....")
      sys.exit()
sol.update_solutions(solution)  # TODO can this be safely removed?
solution.destroy()
snes.destroy()
fnorm=mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(F[0])), op=MPI.SUM)
xnorm=mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(F[1])), op=MPI.SUM)
tnorm=mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(F[2])), op=MPI.SUM)
if mesh_comm.rank==0:
    print("norms of [fuel,oxy,temp] =", [fnorm,xnorm,tnorm])

end_time = MPI.Wtime()
print("time_taken= ",end_time-start_time," secs")
#converged slutions
yf.name='yfb'
yx.name='yxb'
yt.name='ytb'
#write all to file                                                                                                                                                                                     
vtxi = VTXWriter(mesh.comm, "./base_flow_z"+str(zeta)+"_Da"+str(dam)+"i"+str(imesh)+"_p"+str(porder)+".bp", [yf,yx,yt], engine="BP4")
vtxi.write(0)
vtxi.close()
###############################################################################################
#Using the baseflow obtained above, Define and Solve the Linear Stability problem (Eigenvalue problem) 
def eigen_problem(YF,YX,YT,yf,yx,yt,dam):
    Da=petsc4py.PETSc.ScalarType(dam)
    #functions for variables of Linearised Governing equations
    (vf, vx, vt) = (ufl.TestFunction(YF),ufl.TestFunction(YX),ufl.TestFunction(YT))
    (yfb,yxb,ytb) = (ufl.TrialFunction(YF),ufl.TrialFunction(YX),ufl.TrialFunction(YT))
    #governing equations in variational form
    w=Da*(beta**3)*yf*yx*ufl.exp(beta*(yt-1)*(1+gamma)/(1+gamma*yt));
    F=[(ufl.inner(yf.dx(0),vf)+InvLeF*ufl.inner(ufl.grad(yf),ufl.grad(vf))+ufl.inner(w,vf))*ufl.dx,\
       (ufl.inner(yx.dx(0),vx)+InvLeX*ufl.inner(ufl.grad(yx),ufl.grad(vx))+phi*ufl.inner(w,vx))*ufl.dx,\
       (ufl.inner(yt.dx(0),vt)+ufl.inner(ufl.grad(yt),ufl.grad(vt))-(1+phi)*ufl.inner(w,vt))*ufl.dx];
#Jacobian 
    J = [[ufl.derivative(F[0], yf, yfb), ufl.derivative(F[0], yx, yxb),ufl.derivative(F[0], yt, ytb)],\
        [ufl.derivative(F[1], yf, yfb), ufl.derivative(F[1], yx, yxb),ufl.derivative(F[1], yt, ytb)],\
         [ufl.derivative(F[2], yf, yfb), ufl.derivative(F[2], yx, yxb),ufl.derivative(F[2], yt, ytb)]];
    rhs= [[ufl.inner(yfb, vf) * ufl.dx,None,None],[None,ufl.inner(yxb, vx) * ufl.dx,None],[None,None,ufl.inner(ytb, vt) * ufl.dx]];
##define x=0 inlet bc
    def inlet_bc(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
        return np.isclose(x[0], 0)
    boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, inlet_bc)
    #degree of freedom for boundary facets
    bdofs_YF = dolfinx.fem.locate_dofs_topological(YF, mesh.topology.dim - 1, boundary_facets);
    bdofs_YX = dolfinx.fem.locate_dofs_topological(YX, mesh.topology.dim - 1, boundary_facets);
    bdofs_YT = dolfinx.fem.locate_dofs_topological(YT, mesh.topology.dim - 1, boundary_facets);
    # Apply Boundary conditions for Linearised variables
    #fuel
    yf_in=dolfinx.fem.Function(YF,dtype=dtype0)
    yf_in.interpolate(yf_in_eval)
    bcf=dolfinx.fem.dirichletbc(yf_in, bdofs_YF)
    #oxy
    yx_in=dolfinx.fem.Function(YX,dtype=dtype0)
    yx_in.interpolate(yx_in_eval)
    bcx=dolfinx.fem.dirichletbc(yx_in, bdofs_YX)

    #temp
    yt_in=dolfinx.fem.Function(YT,dtype=dtype0)
    yt_in.interpolate(yt_in_eval)
    bct=dolfinx.fem.dirichletbc(yt_in, bdofs_YT)
    #combined bcs
    bc=[bcf,bcx,bct]
    return([F,J,rhs,bc])
# Assemble lhs and rhs matrices
def assemble_mats():
    print("assembling A matrices")
    A = dolfinx.fem.petsc.assemble_matrix_block(dolfinx.fem.form(J), bcs=bc)
    A.assemble()
    print("assembled lhs matrix successfully ")
    print(" Assembling rhs matrix...")
    B = dolfinx.fem.petsc.assemble_matrix_block(dolfinx.fem.form(rhs), bcs=bc)
    B.assemble()
    print("assembled rhs matrix successfully ")
    return A,B

##eigen values
##eigen problem
## function to carry out iteration to get eigen values near the real value target
def eig_iterate(neigs,value)->np.array:
    eps = slepc4py.SLEPc.EPS().create(mesh.comm)
    eps.setOperators(A, B)
    eps.setProblemType(slepc4py.SLEPc.EPS.ProblemType.PGNHEP)
    eps.setDimensions(neigs, petsc4py.PETSc.DECIDE, petsc4py.PETSc.DECIDE)
#eps.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.SMALLEST_REAL)
#eps.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.TARGET_REAL)
    eps.setTarget(petsc4py.PETSc.ScalarType(value))
    ##use Shift Invert startegy for locating eigenvalues near the given target
    eps.getST().setType('sinvert')
    ##Set paramters for Linear solver for high accuracy (Exact solver)
    eps.getST().getKSP().setType("preonly")
    eps.getST().getKSP().getPC().setType("lu")
    eps.getST().getKSP().getPC().setFactorSolverType("mumps")
    eps.setTolerances(1e-5,2000);
    eps.solve()
    #Check convergence and print statistics of solver
    assert eps.getConverged() >= 1
    nconv = eps.getConverged()
    Print("Number of converged eigenpairs: %d" % nconv)
    its = eps.getIterationNumber()
    Print("Number of iterations of the method: %i" % its)
    sol_type = eps.getType()
    Print("Solution method: %s" % sol_type)
    nev, ncv, mpd = eps.getDimensions()
    Print("Number of requested eigenvalues: %i" % nev)
    tol, maxit = eps.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

##define real and imaginary part of eigenvector
    vr = dolfinx.cpp.fem.petsc.create_vector_block(
        [(restriction_.dofmap.index_map, restriction_.dofmap.index_map_bs) for restriction_ in [YF,YX,YT]])
    vi = dolfinx.cpp.fem.petsc.create_vector_block(
        [(restriction_.dofmap.index_map, restriction_.dofmap.index_map_bs) for restriction_ in [YF,YX,YT]])
    vrfun_global=[];
    vifun_global=[];
    eig_global=[]
#    Extract the eignevectors and convert to corresponding dolfinx function 
    for i in range(nconv):
        vr = dolfinx.cpp.fem.petsc.create_vector_block(
            [(restriction_.dofmap.index_map, restriction_.dofmap.index_map_bs) for restriction_ in [YF,YX,YT]])
        vi = dolfinx.cpp.fem.petsc.create_vector_block(
            [(restriction_.dofmap.index_map, restriction_.dofmap.index_map_bs) for restriction_ in [YF,YX,YT]])
        eigv = eps.getEigenpair(i, vr, vi)
        er, ei = eigv.real, eigv.imag
        eig_global.append(eigv)
        error = eps.computeError(i)
           # Transform eigenvector into an eigenfunction that can be plotted
        vr.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        vi.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)

        (yf_fun, yx_fun,yt_fun) = (dolfinx.fem.Function(YF), dolfinx.fem.Function(YX),dolfinx.fem.Function(YT))
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(vr, [YF.dofmap, YX.dofmap,YT.dofmap]) as vr_wrapper:
            for vr_wrapper_local, component in zip(vr_wrapper,(yf_fun, yx_fun,yt_fun)):
                with component.x.petsc_vec.localForm() as component_local:
                    component_local[:] = vr_wrapper_local
        vrfun_global.append((yf_fun, yx_fun,yt_fun))
        (yfi_fun, yxi_fun,yti_fun) = (dolfinx.fem.Function(YF), dolfinx.fem.Function(YX),dolfinx.fem.Function(YT));
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(vi, [YF.dofmap, YX.dofmap,YT.dofmap]) as vi_wrapper:
            for vi_wrapper_local, component in zip(vi_wrapper,(yfi_fun, yxi_fun,yti_fun)):
                with component.x.petsc_vec.localForm() as component_local:
                    component_local[:] = vi_wrapper_local
        vifun_global.append((yfi_fun, yxi_fun,yti_fun))            
    return np.array(eig_global),vrfun_global,vifun_global
##function to normalize eigenfunctions
##function using Infinity norm
def normalize_inf(ufr: dolfinx.fem.Function,\
                  ufi: dolfinx.fem.Function,\
                  uxr: dolfinx.fem.Function,\
                  uxi: dolfinx.fem.Function,\
                  utr: dolfinx.fem.Function,\
                  uti: dolfinx.fem.Function ) -> None:
    """Normalize an eigenvector."""
    scaling_operations: list[tuple[  # type: ignore[no-any-unimported]                                                                                                                                            
        dolfinx.fem.Function, typing.Callable[[dolfinx.fem.Function], ufl.Form],
        typing.Callable[[petsc4py.PETSc.ScalarType], petsc4py.PETSc.ScalarType]
    ]] = [

	([ufr,ufi,uxr,uxi,utr,uti], lambda u: np.max((u[4].x.array)*(u[4].x.array) + (u[5].x.array)*(u[5].x.array)), lambda x: np.sqrt(x)),
	                                                                                                                                           
    ]
    for (functionlist, bilinear_form, postprocess) in scaling_operations:
        scalar = postprocess(mesh.comm.allreduce(bilinear_form(functionlist), op=MPI.MAX))
        #scale each function in list with this scalar
        for function in functionlist:
            function.x.petsc_vec.scale(1. / scalar)
            function.x.petsc_vec.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
##function for normalizing the eigenvectors using complex value of eigenvector at location of maximum value of absolute value of T' as the norm                                                                                                                                                                             
def normalize_new(ufr: dolfinx.fem.Function,\
                  ufi: dolfinx.fem.Function,\
                  uxr: dolfinx.fem.Function,\
                  uxi: dolfinx.fem.Function,\
                  utr: dolfinx.fem.Function,\
                  uti: dolfinx.fem.Function ) -> None:
    """Normalize an eigenvector."""
    scaling_operations: list[tuple[  # type: ignore[no-any-unimported]                                                                                                                                            
        dolfinx.fem.Function, typing.Callable[[dolfinx.fem.Function], ufl.Form],
        typing.Callable[[petsc4py.PETSc.ScalarType], petsc4py.PETSc.ScalarType]
    ]] = [
# Scale functions with a W^{1,1} norm to take away possible sign differences.                                                                                                                              
	([ufr,ufi,uxr,uxi,utr,uti], 
	 lambda u: (np.max((u[4].x.array)*(u[4].x.array) + (u[5].x.array)*(u[5].x.array)),np.argmax((u[4].x.array)*(u[4].x.array) + (u[5].x.array)*(u[5].x.array))), lambda x:x),
	                                                                                                                                           
    ]
    for (functionlist, bilinear_form, postprocess) in scaling_operations:
        (local_max_value, local_max_index)=postprocess(bilinear_form(functionlist))
        ##get arrays
        [ufr,ufi,uxr,uxi,utr,uti]=functionlist;
        # Extract the rank of the process that holds the global minimum
        # We need to manually determine the rank using a separate communication step
        if mesh_comm.rank == 0:
            # Gather all local minimum values and their indices
            all_max_values = mesh_comm.gather(local_max_value, root=0)
            all_max_indices = mesh_comm.gather(local_max_index, root=0)
    
            # Find the rank of the process with the global minimum
            global_max_rank = np.argmax(all_max_values)
            global_max_index = all_max_indices[global_max_rank]
        else:
            mesh_comm.gather(local_max_value, root=0)
            mesh_comm.gather(local_max_index, root=0)

        # Broadcast the rank and index of the process with the global minimum
        global_max_rank = mesh_comm.bcast(global_max_rank if mesh_comm.rank == 0 else None, root=0)
        global_max_index = mesh_comm.bcast(global_max_index if mesh_comm.rank == 0 else None, root=0)

        # Broadcast the value of the other array at the global_min_index from the process with the global minimum
        if mesh_comm.rank == global_max_rank:
            # If this process has the global minimum, send the value at global_min_index
            (Tmaxreal,Tmaximag) = (utr.x.array[global_max_index],uti.x.array[global_max_index])
        else:
            # Otherwise, initialize a variable to receive the value
            (Tmaxreal,Tmaximag) =(None,None)
            
        # Broadcast the value to all processes
        Tmaxreal = mesh_comm.bcast(Tmaxreal, root=global_max_rank)
        Tmaximag = mesh_comm.bcast(Tmaximag, root=global_max_rank)

        #form complex eigenfunctions and normalize with value of complex eigenfunction of temperature at max temp location
        uf=(ufr.x.array+(1j)*ufi.x.array)/(Tmaxreal+(1j)*Tmaximag)
        ux=(uxr.x.array+(1j)*uxi.x.array)/(Tmaxreal+(1j)*Tmaximag)
        ut=(utr.x.array+(1j)*uti.x.array)/(Tmaxreal+(1j)*Tmaximag)
        #extract the components
        ufr.x.array[:]=uf.real
        ufi.x.array[:]=uf.imag
        uxr.x.array[:]=ux.real
        uxi.x.array[:]=ux.imag
        utr.x.array[:]=ut.real
        uti.x.array[:]=ut.imag
            #swap from real part to imag and viceversa
##Define eigen problem    
[F,J,rhs,bc]=eigen_problem(YF,YX,YT,yf,yx,yt,dam)
start_time = MPI.Wtime()
#assemble matrices
A,B=assemble_mats()
#solve
eg_it,efr,efi=eig_iterate(500,-1)
end_time = MPI.Wtime()
print("time taken for eigen value calcuations= ",end_time-start_time," secs")
#save eigenvalues
np.save("eigs_reals_z"+str(zeta)+"da"+str(dam)+"_m"+str(imesh)+"_p"+str(porder),eg_it)
numpy.savetxt("eigs_reals_z"+str(zeta)+"da"+str(dam)+"_m"+str(imesh)+"_p"+str(porder)+".csv", eg_it, delimiter=",") ##save as csv
##save eigenfunctions in VTX format to be read by Paraview
#degrees of freedom
num_dofs_global=0
num_dofs_local=0
for Y in [YF,YX,YT]:
    num_dofs_local=num_dofs_local+ (Y.dofmap.index_map.size_local) * Y.dofmap.index_map_bs
    num_dofs_global=num_dofs_global+ Y.dofmap.index_map.size_global * Y.dofmap.index_map_bs
    print(f"Number of dofs (owned) by rank {mesh_comm.rank}: {num_dofs_local}")
    mesh_comm.Barrier()
    print(f"Number of dofs global: {num_dofs_global}")
for kk in range(1):
##save eigenfunctions
    basename="efNum_z"+str(zeta)+"_da"+str(dam)+"_m"+str(imesh)+"_p"+str(porder)
    os.makedirs(basename,exist_ok=True)
    if mesh_comm.rank==0:
       print("no of eigenfunctions recorded =",len(efr)," ",len(efi))
    #define required no of eigenfunctions to save, as it takes lot of disk space
    meig=min(len(efr),len(efi));
   # meig=2; 
    for i in range(meig):
        if np.logical_and(np.iscomplex(eg_it[i]),np.greater(abs(eg_it[i].imag),0.15)):
            if mesh_comm.rank==0:
                print("writing ",i,"th eigen mode with eigen value ",eg_it[i])
            fname=os.path.join(basename,str(i)+".bp");
            #extract real eigenfunctions            
            (ffr,fxr,ftr)=efr[i];
            ffr.name='yfr'
            fxr.name='yxr'
            ftr.name='ytr'
            #extract imaginary part of eigenfunctions
            (ffi,fxi,fti)=efi[i];
            normalize_new(ffr,ffi,fxr,fxi,ftr,fti)
            ffi.name='yfi'
            fxi.name='yxi'
            fti.name='yti'
            #write all to file
            vtxi = VTXWriter(mesh.comm, fname, [ffr,ffi,fxr,fxi,ftr,fti], engine="BP4")
            vtxi.write(i)
            vtxi.close()
mesh_comm.Barrier()    
