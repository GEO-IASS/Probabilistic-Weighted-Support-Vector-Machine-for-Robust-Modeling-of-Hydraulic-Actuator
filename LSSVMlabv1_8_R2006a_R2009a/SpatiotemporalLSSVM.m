clear
clc
x=0:pi/20:pi;
t=0:0.01:5;
dx=pi/20;
dt=0.01;
C=50*(exp(-4/5)+exp(-4));
y=zeros(length(x),length(t));
y(1,:)=0;y(end,:)=0;y(:,1)=0;
syms T X
for i=1:4
    b(i)=heaviside(X-pi/4*(i-1))-heaviside(X-pi/4*i);
    u(i)=1.1+5*sin(0.1*T+0.1*i);
end
for l=1:4
    B(l,:)=subs(b(l),X,[0:pi/20:pi]);
    U(l,:)=subs(u(l),T,[0:0.01:5]);
end
data=B'*U;
for i=2:20
    for j=1:500
        y(i,j+1)=y(i,j)+dt/dx^2*(y(i+1,j)-2*y(i,j)+y(i-1,j))+dt*C+dt*2*(data(i,j)-y(i,j));
    end
end
input_x=x';
output_T=y;
gam=5;
sig2=10;
type='function approximation';
[alpha,b] = trainlssvm({input_x,output_T,type,gam,sig2,'RBF_kernel'});