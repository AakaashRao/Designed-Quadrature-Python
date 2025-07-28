% October 12, 2020
clc;
clear;
Sim = 50; %Number of resample
J = 5; % number of alternatives
N = 1000; % number of individuals
NT = 5*N;  % number of time periods observed for each individual
nrx = 3; % number of x variables with random coefficients
ide_par = nrx+nrx*(nrx+1)*.5 ; % number of parameters
mean1 = reshape(.25:.75/(nrx-1):1,nrx,1);
A = [1.5	0.5	0.5	
	0.5	1	0.5
	0.5	0.5	1.5];
      
NROWS = NT*J;
NCOLS = 4 + nrx;
data = zeros(NROWS,NCOLS);
rng(10);

%%Personid (col1)
col1 = repmat(1:N,NROWS/N,1);
data(:,1) = reshape(col1,NROWS,1);

%choice situation id (col2)
col2 = repmat(1:NT,J,1);
data(:,2) = reshape(col2,NROWS,1);

%Generating Covariates
data(:,(4:(NCOLS-1))) = normrnd(0,1,[NROWS nrx]);

%alternative numbering (col 8)
col_end =transpose(repmat(1:J,NT,1));
data(:,NCOLS) = col_end(:);

Xr = data(:,4:(NCOLS-1));

L_A = chol(A,'lower'); % cholesky of true variance covariance matrix
ind_lowtri = tril(ones(nrx))==1;
temp = L_A';
chol_para= temp(ind_lowtri');
parameter = [mean1;chol_para];

R_set_M = [30,50,100,1000]; 
R_set_H = [30,50,100,1000]; 

% MLHS
Mmean = zeros(ide_par,length(R_set_M),Sim);
Mstd = zeros(ide_par,length(R_set_M),Sim);
Mfval = zeros(length(R_set_M),Sim); %  draws x number of simulations
Mtime = zeros(length(R_set_M),Sim); % draws x number of simulations
Miter = zeros(length(R_set_M),Sim); % draws x number of simulations
Mfun = zeros(length(R_set_M),Sim); % draws x number of simulations

% Halton
Hmean = zeros(ide_par,length(R_set_H),Sim);
Hstd = zeros(ide_par,length(R_set_H),Sim);
Hfval = zeros(length(R_set_H),Sim); %  draws x number of simulations
Htime = zeros(length(R_set_H),Sim); % draws x number of simulations
Hiter = zeros(length(R_set_H),Sim); % draws x number of simulations
Hfun = zeros(length(R_set_H),Sim); % draws x number of simulations

% Quadrature
Q_plan = xlsread('sim_plan_dq.xlsx','d=3');

Qmean = zeros(ide_par,length(Q_plan(:,1)),Sim);
Qstd = zeros(ide_par,length(Q_plan(:,1)),Sim);       
Qfval = zeros(length(Q_plan(:,1)),Sim);
Qtime = zeros(length(Q_plan(:,1)),Sim);
Qiter = zeros(length(Q_plan(:,1)),Sim); % draws x number of simulations
Qfun = zeros(length(Q_plan(:,1)),Sim); % draws x number of simulations

for SS =1:Sim
    SS  
    
    %%%%%%% Data generation starts %%%%%%%%
    %Creating parameters
    
    data(:,3) = 0 ;
    rng(SS) %setting seed 

    temp_par = mvnrnd(mean1,A,N);
    params = kron(temp_par,ones(NROWS/N,1));
    error=evrnd(0,1,[NROWS,1]); %Different realizations of error term for each dataset
    utility = sum(data(:,4:(NCOLS-1)).*params,2) + error;
    utility_wo_err = sum(data(:,4:(NCOLS-1)).*params,2); 
    
    k=1;
    j=1;
    keep_ind = zeros(NT,1);
    keep_ind_wo_err = zeros(NT,1);
    while j <= NROWS
        [~,ind] = max(utility(j:(j+J-1)));
        data(j+ind-1,3)=1;
        keep_ind(k) = ind;
        
        [~,ind_wo_err] = max(utility_wo_err(j:(j+J-1)));
        keep_ind_wo_err(k) = ind_wo_err;
        j=j+J;
        k=k+1;
    end
    %tabulate(keep_ind)
    diff_err = sum(keep_ind_wo_err ~= keep_ind)/NT; % proportion of time different choice is made    
    Y = data(:,3);
    options=optimset('LargeScale','off','GradObj','on','DerivativeCheck','off','Display','off');
    initial = .5*rand(length(parameter),1).*parameter;
        
    %%%%%%% Data generation ends %%%%%%%%
    %%% MLHS   
    for ndraw = 1:length(R_set_M)
        R = R_set_M(ndraw); %Number of simulation Draws
        
        dr_sn = makedraws(nrx,N,4,R); % type=4 for MLHS        
        fnew = @(x) flexll_grad(x, Xr,Y, J, N, R, NT, nrx,dr_sn);
        tic
        [paramhat,fval,~,output,~,hessian]=fminunc(fnew,initial,options);
        Mtime(ndraw,SS) = toc;
        Mmean(:,ndraw,SS) = paramhat;
        Mstd(:,ndraw,SS) = sqrt(diag(inv(hessian)));        
        Mfval(ndraw,SS) = fval;   
        Miter(ndraw,SS) = output.iterations;
        Mfun(ndraw,SS) = output.funcCount;
    end   
    %%% shuffled and scrambled Halton    
    for ndraw = 1:length(R_set_H)
        R = R_set_H(ndraw); %Number of simulation Draws        
        dr_sn = makedraws(nrx,N,3,R); % type=3 for shuffled and scrambled Halton        
        fnew = @(x) flexll_grad(x, Xr,Y, J, N, R, NT, nrx,dr_sn);
        tic
        [paramhat,fval,~,output,~,hessian]=fminunc(fnew,initial,options);
        Htime(ndraw,SS) = toc;
        Hmean(:,ndraw,SS) = paramhat;
        Hstd(:,ndraw,SS) = sqrt(diag(inv(hessian)));        
        Hfval(ndraw,SS) = fval;
        Hiter(ndraw,SS) = output.iterations;
        Hfun(ndraw,SS) = output.funcCount;
    end   
    % Quadrature
    for ndraw = 1:length(Q_plan(:,1))
        acc=Q_plan(ndraw,1); R=Q_plan(ndraw,2);
        filename = sprintf('TO_d_%d_accu_%d_node_%d.xlsx', nrx,acc,R);
        XW = xlsread(filename);
        node = XW(:,1:nrx);
        weight = XW(:,end);
        tic
        fnew = @(x) flexll2008_grad(x,Xr,Y, J, N, R, NT, nrx,node, weight);
        [paramhat,fval,~,output,~,hessian]=fminunc(fnew,initial,options);
        Qtime(ndraw,SS) = toc;
        Qmean(:,ndraw,SS) = paramhat;
        Qstd(:,ndraw,SS) = sqrt(diag(inv(hessian)));        
        Qfval(ndraw,SS) = fval;
        Qiter(ndraw,SS) = output.iterations;
        Qfun(ndraw,SS) = output.funcCount;
    end 
end
save('NONdiag_3rand_output.mat')

 
