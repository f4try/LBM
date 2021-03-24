% LBM- 1-D1Q3, diffusion equation D1Q3, with variable alpha
clear
m=101;
w0=4./6.;
w1=1./6.;
c2=1./3.;
w2=w1;
dx=1.0;
rho=zeros(m);f0=zeros(m);f1=zeros(m);f2=zeros(m);
x=zeros(m);fluxq=zeros(m);flux=zeros(m);
tk=zeros(m);dtkdx=zeros(m);cpr=zeros(m);alpha=zeros(m);
omega=zeros(m);
x(1)=0.0;
for i=1:m-1
  x(i+1)=x(i)+dx;
end
for i=1:m
  tk(i)=20.+30./(2.*x(i)+1.);
  alpha(i)=tk(i)/100.;
  omega(i)=1/(3.*alpha(i)+0.5);
end
twall=1.0;
nstep=1500;
for i=1:m
  f0(i)=w0*rho(i);
  f1(i)=w1*rho(i);
  f2(i)=w1*rho(i);
end
%Collision:
for k1=1:nstep
  for i=1:m
    feq0=w0*rho(i);
    feq=w1*rho(i);
    f0(i)=(1-omega(i))*f0(i)+omega(i)*feq0;
    f1(i)=(1-omega(i))*f1(i)+omega(i)*feq;
    f2(i)=(1-omega(i))*f2(i)+omega(i)*feq;
  end
  % Streaming:
  for i=1:m-1
    f1(m-i+1)=f1(m-i);
    f2(i)=f2(i+1);
  end
  %Boundary condition:
  f1(1)=twall-f2(1)-f0(1);
  f1(m)=f1(m-1);
  f2(m)=f2(m-1);
  f0(m)=f0(m-1);
  for j=1:m
    rho(j)=f1(j)+f2(j)+f0(j);
  end
end
%Flux:
for k=1:m
  flux(k)=tk(k)*omega(k)*(f1(k)-f2(k))/c2;
end
for k=1:m-1
  fluxq(k)=tk(k)*(rho(k)-rho(k+1));
end
fluxq(m)=fluxq(m-1);
figure(1)
plot(x,rho)
title("Temperature, nstep=200")
xlabel("X")
ylabel("T")
figure(2)
plot(x,flux,"o",x,fluxq,"x")
title("Flux")
xlabel("X")
ylabel("Flux")