function [ p,xy_all,yita,u_a] = train_robust_weighted_lssvm( xy_train,lanbada,gama,cluster_num,resample,sig_new,window_width)
train_num=size(xy_train,1);
e_all=[];
sig_all=[];
save_model=[];
rand_store=[];
for i=1:cluster_num
    rand_resample=randi([1 train_num],resample,1);
    if i==1
	rand_store=rand_resample;
	else
	rand_store=[rand_store,rand_resample];
	end
    xy_cluster=xy_train(rand_resample,:);
    cluster.resample(i).xy=xy_cluster;
    model=initlssvm(xy_cluster(:,1:end-1),xy_cluster(:,end),'f',[],[],'RBF_kernel','o');
    model=tunelssvm(model,'simplex','crossvalidatelssvm',{10,'mae'});
	gam=model.gam;
	sig2=model.kernel_pars;
    sig_all=[sig_all;sig2];
    cluster.resample(i).sig2=sig2;
    [alpha,b] = trainlssvm({xy_cluster(:,1:end-1),xy_cluster(:,end),'f',gam,sig2,'RBF_kernel','o'});
    cluster.resample(i).alpha=alpha;
    y_resample_predict = simlssvm({xy_cluster(:,1:end-1),xy_cluster(:,end),'f',gam,sig2,'RBF_kernel','o'},{alpha,b},xy_train(:,1:end-1));
    e_cluster=sum((xy_train(:,end)-y_resample_predict).^2);
    cluster.resample(i).e_cluster=e_cluster;
    e_all=[e_all;e_cluster];
end
 
u=zeros(cluster_num,1);
for parzon=1:cluster_num
    parameter_in_window=-((e_all(parzon)-e_all)./window_width).^2;
    pdf_temp=exp(parameter_in_window);
    pdf=sum(pdf_temp);
    u(parzon)=pdf;
end
u=u./(sum(u));
[sort_u,sort_i]=sort(u,'descend');
xy_all=[];
alpha_all=[];
u_all=[];
sig_all=[];
for i=1:10  %只取了前10个
    loop=sort_i(i);
    xy_all=[xy_all;cluster.resample(loop).xy];
    alpha_all=[alpha_all;cluster.resample(loop).alpha];
    sig_all=[sig_all;cluster.resample(loop).sig2];
    u_every=repmat(u(loop),resample,1);
    u_all=[u_all;u_every];
end
% sig_new=sig_all'*u;
U=diag(1./(gama.*u_all));
yita=1/(lanbada+sum(u));
kernel=kernel_matrix(xy_all(:,1:end-1),'RBF_kernel',sig_new);
matrix_num=size(U,1);
matrix_top=[(yita*kernel+U),-repmat(1,matrix_num,1)];

matrix_below=[ones(1,matrix_num),0];
front=[matrix_top;matrix_below];
u_a=alpha_all.*u_all;
matrix_behind_shang=kernel*u_a*yita-xy_all(:,end);
matrix_behind=[matrix_behind_shang;0];
inf_place=isinf(front);
inf_num=find(inf_place);
nan_place=isnan(front);
nan_num=find(nan_place);
if isempty(inf_num)&&isempty(nan_num)
p_all=pinv(front)*matrix_behind;
p=p_all;
else
    p=[];
end

end
