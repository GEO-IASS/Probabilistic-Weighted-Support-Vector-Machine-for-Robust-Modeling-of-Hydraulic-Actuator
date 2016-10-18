function [ y ] = sim_robust_weighted_lssvm( xy,p,xy_all,yita,u_a,sig )
if isempty(p)
    y=[];
else
kernel=kernel_matrix(xy_all(:,1:end-1),'RBF_kernel',sig,xy(:,1:end-1));
kernel=kernel';
y=yita*kernel*u_a-yita*kernel*p(1:end-1,:)+p(end,:);
end
end