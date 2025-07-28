function [ll, g]=flexll2008_grad(para,Xr,Y, J, N, R, NT, nrx,node, weight)

RCmean_l(:,1) = para(1:nrx);

temp = zeros(nrx,nrx);
chol_vec(:,1) = para((nrx+1):end);
ind_lowtri = tril(ones(nrx))==1;
temp(ind_lowtri') = chol_vec; %upper triangular matrix
chol1= temp'; %L11, L21,L22,L31,L32,L33, INSTEAD OF L11,L21,L31, L22,L32,L33

var = chol1*node';
beta = bsxfun(@plus,RCmean_l,var); % nrx X  R 

T= NT/N;
vRC = Xr*beta; 

v = reshape(vRC ,[J T*N*R]);
expv = exp(v);
pr = reshape(bsxfun(@rdivide,expv,sum(expv,1)),[J*T*N R]); % prob of choosing an alternative for each draw
like = reshape(prod(reshape(pr(Y==1,:),[T N R]),1),[N R]); % Conditional likelihood of each person and draw
ll_temp = like*weight;  % unconditional loglikelihood %N x 1
ll = -sum(log(ll_temp));


%% gradient
temp = repmat(node',[1 1 N]);
%temp = reshape(dr_sn,[nrx R N]);
dr_sn_o = permute(temp,[3 2 1]);

% Derivative of random coefficient w.r.t mean and cholesky parameters
g_beta_gama_chol = zeros(N,R,nrx,nrx+nrx*(nrx+1)*.5);
g_beta_gama_chol(:,:,:,1:nrx) = repmat(reshape(eye(nrx), [1 1 nrx nrx]),[N R 1 1]); % adding identity
for j=1:nrx
    temp1 = zeros(N,R,1,nrx*(nrx+1)*.5);
    temp2 = reshape(dr_sn_o(:,:,1:j),[N R 1 j]);
    ind1 = j*(j-1)*.5+1;
    ind2 = j*(j+1)*.5;
    temp1(:,:,1,ind1:ind2) =  temp2 ;
    g_beta_gama_chol(:,:,j,(nrx+1):(nrx+nrx*(nrx+1)*.5)) = temp1;
end
g_beta_gama_chol = permute(g_beta_gama_chol,[4 3 1 2]); % (nrx+nrx*(nrx+1)*.5) x nrx x N x R


%wrt beta (random parameter)
Xr_choose = Xr(Y==1,:)'; % nrx  x (N*T)
l_beta = repmat(reshape(Xr_choose,[nrx,T,N,1]),[1,1,1,R]); % nrx x T x N x R (new)

Xr_trans = Xr';% nrx  x (J*T*N)
Xr_trans_re =repmat(reshape(Xr_trans,[nrx,J,T,N,1]),[1,1,1,1,R]);% nrx  x J x T x N x R
prob = repmat(reshape(pr,[1 J T N R]),[nrx 1 1 1 1]); %nrx  x J x T x N x R (new)
r_beta = squeeze(sum(prob.*Xr_trans_re,2)); %nrx  x T x N x R (new)

temp_g_beta= squeeze(sum(l_beta-r_beta,2));%nrx  x N x R (new)
like_c_beta = repmat(reshape(like,[1 N R]),[nrx 1 1]); % nrx x N x R
g_like_beta = like_c_beta.*temp_g_beta;% nrx x N x R (new);
g_like_beta_re = reshape(g_like_beta,[nrx 1 N R]);
g_nu_chol= squeeze(multiprod(g_beta_gama_chol, g_like_beta_re));%(nrx+nrx*(nrx+1)*.5) x N x R
%g_nu_chol_temp =g_nu_chol_temp  + squeeze(sum(permute(g_nu_chol,[3 1 2]) .* weight, 1)); %(nrx+nrx*(nrx+1)*.5) x N

g_nu_chol_temp = 0 ;
for r=1:R
g_nu_chol_temp =g_nu_chol_temp  + squeeze(g_nu_chol(:,:,r)*weight(r)) ; %(nrx+nrx*(nrx+1)*.5) x N
end


denom_uncond_like = repmat(ll_temp,[1 (nrx+nrx*(nrx+1)*.5)]); %N x (nrx+nrx*(nrx+1)*.5)
g = -sum(g_nu_chol_temp ./denom_uncond_like',2);
end
