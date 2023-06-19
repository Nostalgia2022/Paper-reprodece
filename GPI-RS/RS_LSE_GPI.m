clc
clear

Nt=32;
K=4;

alpha=0.5;

tolerance=1e-4;

noise=ones(K,1);

rng(2)
H=1/sqrt(2)*(randn(Nt,K)+1j*randn(Nt,K));

transmit_SNr=20;
Pt=db2pow(transmit_SNr);

P_c=Pt*0.6;
P_p=(Pt-P_c)/K; 

E=[];

[U,~,~]=svd(H);
p_c=U(:,1)*sqrt(P_c);
P_k=H./vecnorm(H)*sqrt(P_p);

p_all=[p_c;P_k(:)];


[A_KKT,B_KKT,A_c,B_c,A_k,B_k]=deal(zeros(Nt*(K+1)));
for k=1:K
    A_c(:,:,k)=kron(eye(Nt+1),H(:,k)*H(:,k)')+noise(k)/Pt*eye(Nt*(K+1));
    B_c(:,:,k)=A_c(:,:,k);
    B_c(1:Nt,1:Nt,k)=A_c(1:Nt,1:Nt,k)-H(:,k)*H(:,k)';

    A_k(:,:,k)=B_c(:,:,k);
    B_k(:,:,k)=A_k(:,:,k);
    B_k(k*Nt+1:Nt*(k+1),k*Nt+1:k*Nt+Nt,k)=A_k(k*Nt+1:Nt*(k+1),k*Nt+1:k*Nt+Nt,k)...
    -H(:,k)*H(:,k)';
end

gamma_c=zeros(K,1);
gamma_k=zeros(K,1);


flag=1;
p_all_past=p_all;
test_past=0;
count=0;
maxcount=1000;
tic
while(flag)

for k=1:K
    gamma_c(k)=quad_form(p_all,A_c(:,:,k))/quad_form(p_all,B_c(:,:,k));
    gamma_k(k)=quad_form(p_all,A_k(:,:,k))/quad_form(p_all,B_k(:,:,k));
end
lambda=(1/K*sum(exp(-1./alpha.*log2(gamma_c))))^(-alpha/log2(exp(1)))*cumprod(gamma_k);
[N,D]=numden(sym(lambda(K)));
N=double(N);
D=double(D);

con=sum(exp(1/(-alpha)*log2(gamma_c)));

for k=1:K
    A_KKT(:,:,k)=exp(1/(-alpha)*log2(gamma_c(k)))/(con*quad_form(p_all,A_c(:,:,k)))*A_c(:,:,k)+1/quad_form(p_all,A_k(:,:,k))*A_k(:,:,k);

    B_KKT(:,:,k)=exp(1/(-alpha)*log2(gamma_c(k)))/(con*quad_form(p_all,B_c(:,:,k)))*B_c(:,:,k)+1/quad_form(p_all,B_k(:,:,k))*B_k(:,:,k);

end

A_kkt=N*sum(A_KKT,3);
B_kkt=D*sum(B_KKT,3);

new_p_all=B_kkt\A_kkt*p_all;
p_all=new_p_all/norm(new_p_all);

e=norm(p_all-p_all_past);
E=[E;e];
% for k=1:K
%     gamma_c(k)=quad_form(p_all,A_c(:,:,k))/quad_form(p_all,B_c(:,:,k));
%     gamma_k(k)=quad_form(p_all,A_k(:,:,k))/quad_form(p_all,B_k(:,:,k));
% end
% 
% E=[E;min(log2(gamma_c))+sum(log2(gamma_k))];
test=floor(count/50);
switch test-test_past==0
    case 0
        alpha=alpha+0.5;
        test_past=test;
    case 1
        test_past=test;
end
if norm(p_all-p_all_past)<tolerance
    flag=0;
else
    p_all_past=p_all;
    count=count+1;
end

if count>=1000
    break
end

end
toc

for k=1:K
    gamma_c(k)=quad_form(p_all,A_c(:,:,k))/quad_form(p_all,B_c(:,:,k));
    gamma_k(k)=quad_form(p_all,A_k(:,:,k))/quad_form(p_all,B_k(:,:,k));
end

Y=min(log2(gamma_c))+sum(log2(gamma_k))


