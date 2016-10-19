clear
clc
y=Y_PCA_design_oven(160,4,500);
for i=1:9
    yy(i,:)=y(i,:)-mean(y);
end
C=yy'*yy/8;
[m,n]=eig(C);
% C为协方差矩阵
m=flipud(m');
% a的每一行为一个特征向量，第一行对应特征值最大，依次递减
for i=1:2
    phi(i,:)=m(i,:)*y';
end
p=norm(phi);
phi=phi/p;
a=phi*y;
% KL分解得到a(t)
u=500+20*(rand(1,7001)-0.5);
input_t=[a(:,1:2500);u(1:2500)]';
output_y1=a(1,2:2501)';
output_y2=a(2,2:2501)';
gam_1=2000;sig2_1=3000;
gam_2=2000;sig2_2=3000;
type='function approximation';
[alpha_1,b_1] = trainlssvm({input_t,output_y1,type,gam_1,sig2_1,'RBF_kernel'});
[alpha_2,b_2] = trainlssvm({input_t,output_y2,type,gam_2,sig2_2,'RBF_kernel'});
input=input_t(1,:);
for i=1:5000
    pre_y1=simlssvm({input_t,output_y1,type,gam_1,sig2_1,'RBF_kernel','preprocess'},{alpha_1,b_1},input);
    pre_y2=simlssvm({input_t,output_y2,type,gam_2,sig2_2,'RBF_kernel','preprocess'},{alpha_2,b_2},input);
    input=[pre_y1 pre_y2 u(i+1)];
    pre_1(i)=pre_y1;
    pre_2(i)=pre_y2;
end
pre_1=[a(1,1) pre_1];
pre_2=[a(2,1) pre_2];
YY=phi'*[pre_1;pre_2];
figure(1)
plot(a(1,:))
hold on
plot(pre_1,'r')
legend('measured','predicted')
xlabel('time T');ylabel('Temporal coefficent a1(t)')
figure(2)
plot(a(2,:))
hold on
plot(pre_2,'r')
legend('measured','predicted')
xlabel('time T');ylabel('Temporal coefficent a2(t)')
figure(3)
mesh([1:5001],[1:9],YY)
xlabel('time T');ylabel('space X');zlabel('Predicted Temperature y(x,t)')
figure(4)
mesh([1:5001],[1:9],y)
xlabel('time T');ylabel('space X');zlabel('Measured Temperature y(x,t)')
dif=max(YY)-min(YY);
r=1:5001;
for i=1:5001
    if r(i)<=1001
        Y(i)=0.199*(r(i)-1);
    else
        Y(i)=199+1/4000*(r(i)-1001);
    end
end
err=mean(YY)-Y;