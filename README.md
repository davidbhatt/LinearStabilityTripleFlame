# LinearStabilityTripleFlame
Linear Stability of Laminar flames with varying premixedness using Thermo-diffusive model
#### Finite Element Code to carry out Linear stability analysis on Laminar flames using Thermo-diffusive model###################
###This code reads in baseflow (initial guess of stationary solution from a file and finds accurate stationary solution suing newtons Raphson method
#then calculates eigenvalues and eigenfunction and write them to disk
#Required libraries, dolfinx-real, numpy, pandas, mpi4py slepc4py, petsc4py, scipy 
#Requires initial guess for stationary solutioonfor proper convergence of Newton's method. This has been obtained by Selective frequency Damping code.
# This code can be run in parallel using MPI
#use mpirun -np [no of processes] python3 [code.py]
