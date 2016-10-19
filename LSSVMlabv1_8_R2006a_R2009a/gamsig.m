clear
clc
uy=load('D:\matlab\license_standalone.dat\LSSVMlabv1_8_R2006a_R2009a\uy.txt');
for i=1:2998
    x(i,:)=[uy(:,i);uy(:,i+1)];
end
for j=1:2998
    y(j)=uy(5,j+2);
end
y=y';
type='function estimation';
[gam,sig2]=tunelssvm({x,y,type,[],[],'RBF_kernel'},[1e-3 1e5 0.01 3],'leaveoneoutlssvm',{'mse'})