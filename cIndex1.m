function ObjVal = cIndex (X,Dx)
% [Nind Nvar]=size(Chrom);
global eT eH eL eA T H L A GlobalParams
%close all
%clear all
%clc

%Food = 'Group1New.xlsx';
% filename = 'Data_knn.xlsx';
%data2 = xlsread(Food);

u1 = .45;
u2 = .30;
u3 = .15;
u4 = .10;

%u1 = .25;
%u2 = .25;
%u3 = .25;
%u4 = .25;

%for j=1:8
T=Dx(:,1); % Column A of Excel
H=Dx(:,2); %Cloum B
L=Dx(:,3);% Cloumn C
A=Dx(:,4); % Cloumn D
    %x1=data2 (j,5);% Optimized Temperature by ABC Algorithm you have to optimize using Optimization algorithm
    %x2=data2 (j,6);% Optimized Humidity by ABC Algorithm you have to optimize using Optimization algorithm
    %x3=data2 (j,7);% Optimized Illumination by ABC Algorithm you have to optimize using Optimization algorithm
    %x4=data2 (j,8);% Optimized Air Quality of CO2 by ABC Algorithm you have to optimize using Optimization algorithm
x1 = X(:,1);
x2 = X(:,2);
x3 = X(:,3);
x4 = X(:,4);
eT=T-x1;
eH=H-x2;
eL=L-x3;
eA=A-x4;
%ObjVal  = (u1*(1-(eT/T)*(eT/T))+ u2*(1-(eH/H)*(eL/H)) + u3*(1-(eL/L)*(eL/L)) + u4*(1-(eA/A)*(eA/A)));
ObjVal  = (u1*(1-((eT/x1)*(eT/x1))^2)+ u2*(1-((eH/x2)*(eL/x2))^2) + u3*(1-((eL/x3)*(eL/x3))^2) + u4*(1-((eA/x4)*(eA/x4))^2));
%dlmwrite('T1.txt',T,'-append','delimiter','\t');
%dlmwrite('T2.txt',H,'-append','delimiter','\t');
%dlmwrite('T3.txt',L,'-append','delimiter','\t');
%dlmwrite('T4.txt',A,'-append','delimiter','\t');
%dlmwrite('T5.txt',eT,'-append','delimiter','\t');
%dlmwrite('T6.txt',eH,'-append','delimiter','\t');
%dlmwrite('T7.txt',eL,'-append','delimiter','\t');
%dlmwrite('T8.txt',eA,'-append','delimiter','\t');
%dlmwrite('T9.txt',ObjVal ,'-append','delimiter','\t');
%fprintf('ObjVal=%g eT=%g eL=%g eA=%g\n',ObjVal,eT,eL,eA),T,H,L,A;
    % Also Calculate RMSE of Cloumn A and Optimized Temperature
    % RMSE of Clumn B and Optimized Humidity
    % RMSE of Column C and Optimized Illumination
    % RMSE of Column D and Optimized CO2














%%
% T=data(:,1)
% L=data(:,2)
% A1=data(:,3)
%     u1 = .30;
%     u2 = .30;
%     u3 = .40;
% for j=3:136
%     T=data (1,j);
%     L=data (2,j);
%     A1=data (3,j); 
%     x1=73;
%     x2=800;
%     x3=800;
%     ObjVal (j) = (u1*(1-(eT/x1)*(eT/x1)) + u2*(1-(eL/x2)*(eL/x2)) + u3*(1-(eA/x3)*(eA/x3)));
% end
    
% for j=1:172
%     T=data (1,j);
%     L=data (2,j);
%     A1=data (3,j);






























%%
% for i=1:Nind
%     
%     x1=Chrom (i,1);
%     x2=Chrom (i,2);
%     x3=Chrom (i,3);
%     eT=T-x1;
%     eL=L-x2;
%     eA=A1-x3;
    
    
    
%     eT = x1-T;
%     eL = x2-L;
%     eA = x3-A;
%     dlmwrite('T.txt',eT,'delimiter','\t');
%     dlmwrite('L.txt',eL,'delimiter','\t');
%     dlmwrite('A.txt',eA,'delimiter','\t');

% 
%     ObjVal (i) = (u1*(1-(eT/x1)*(eT/x1)) + u2*(1-(eL/x2)*(eL/x2)) + u3*(1-(eA/x3)*(eA/x3)));
%     
%     
    
% dlmwrite('T.txt',T(j),'delimiter','\t');
% dlmwrite('T1.txt',L(j),'delimiter','\t');
% dlmwrite('T2.txt',A1(j),'delimiter','\t');
% end 

%  dlmwrite('T.txt',GlobalParams (j),'delimiter','\t');
% end
% size (x1)
% size (x2)
% size (x3)
