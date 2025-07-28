
###################################################
written by Prateek Bansal and Vahid Keshavarzzadeh 
Last updated: March 2021
###################################################

This folder includes MATLAB codes to replicate the simulation results presented in Tables 1 to 3 of the paper titled as 
"Designed Quadrature to Approximate Integrals in Maximum Simulated Likelihood Estimation". This folder has two sub folders 
and files in each of these folders are described below: 
1. Generate DQ: To generate DQ rules
2. Simulation: To replicate Monte Carlo study 

#####################################
########## 1. Generate DQ ###########
#####################################

main.m: The user needs to specify the dimension, order of polynomial integration, and the number of nodes. DQ rules will be output in an excel file. 

-------
Changing from Gaussian to Uniform case: 
-------
In the current setting the code generates Gaussian nodes but for the uniform case, the user should make the following changes: 
1. In the function quad_int_mul_u_sens.m, pol_mul_g should be replaced by pol_mul_jacobi for uniform case. 
2. In the generator.m, the fixed regularization parameters are provided within the generator.m file. For uniform case, uncomment the uniform part and comment the Gaussian part.  

-------
Remarks: 
-------
Changing the number of nodes and regularizations affect the results. 
As a general comment, higher number of nodes are more likely to yielf convergence at a given error tolerance. 
In other words, if the number of nodes do not work at a given polynomial order, the user can repeat the design with higher number of nodes. 
Regularization can also impact the solution. Using higher regularization makes the code slower but it produces more stable solution. 


#####################################
########### 2. Simulation ###########
#####################################

Tables 1, 2, and 3 correspond to random parameters (dimension of integral) 3, 5, and 10, respectively. All three Simulation studies 
follow the same naming conventions, and therefore, we describe specific files related to the Monte Carlo Study of Mixed Logit 
Model with 5 random parameters. We write all codes with analytical gradients of the loglikelihood.  

DIAG_5rand.m: Main code for diagonal variance-covariance matrix (includes data generation and estimation).   
NONdiag_5rand.m:  Main code for full (non-diagonal) variance-covariance matrix(includes data generation and estimation).     

-------
Inputs for above codes: 
-------

flexll_grad.m: Function to compute likelihood and gradient of the likelihood function for Halton and MLHS.  
flexll2008_grad.m: Function to compute likelihood and gradient of the likelihood function for Designed Quadrature. 
makedraws.m: Function to generate standard normal draws from Halton and MLHS. 
multiprod.m: Function performs multiple matrix products, with array expansion enabled.
sim_plan_dq.xlsx: Simulation plan for DQ
TO_d_5_accu_6_node_50.xlsx: DQ rules for accuracy (order) 6, and 50 nodes (consistent naming convention), generated using main.m in folder "Generate DQ".

-------
Outputs from above codes: 
-------

DIAG_5rand_output.mat: Output of the Monte Carlo study for diagonal variance-covariance matrix.  
NONdiag_5rand_output.mat: Output of the Monte Carlo study for full (non-diagonal) variance-covariance matrix. 
Simple arithmetic operation on these output files will generate results of Table 2.  

 
   




 
