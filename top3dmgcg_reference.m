%%%% 3D TOPOLOGY OPTIMIZATION CODE, MGCG ANALYSIS %%%%
% nelx - number of elements in x
% nely - number of elements in y
% nelz - number of elements in z
% volfrac - number of elements in x
% penal - number of elements in x
% rmin - 
% ft   - 
% nl -
% cgtol -
% cgmax -
% cgmin -
function top3dmgcg(nelx,nely,nelz,volfrac,penal,rmin,ft,nl,cgtol,cgmax,cgmin)
%
% example run command:
%
% top3dmgcg(64,32,32,0.12,3,2.4,1,4,1e-5,100,1)
% top3dmgcg(24,12,12,0.12,3,2.4,1,3,1e-5,100,1) 
% top3dmgcg(2,4,6,0.12,3,1.6,1,2,1e-5,100,1)
% 
% MATERIAL PROPERTIES
close all
E0 = 1;
Emin = 1e-6;
nu = 0.3;
%% PREPARE FINITE ELEMENT ANALYSIS
KE = Ke3D(nu);
% Prepare fine grid
nelem = nelx*nely*nelz;

% number of nodes
nx = nelx+1; 
ny = nely+1; 
nz = nelz+1;

% size of state matrix and vectors
ndof = 3*nx*ny*nz;


nodenrs(1:ny,1:nz,1:nx) = reshape(1:ny*nz*nx,ny,nz,nx);
edofVec(1:nelem,1) = reshape(3*nodenrs(1:ny-1,1:nz-1,1:nx-1)+1,nelem,1);
edofMat(1:nelem,1:24) = repmat(edofVec(1:nelem),1,24) + ...
    repmat([0 1 2 3*ny*nz+[0 1 2 -3 -2 -1] -3 -2 -1 ...
    3*ny+[0 1 2] 3*ny*(nz+1)+[0 1 2 -3 -2 -1] 3*ny+[-3 -2 -1]],nelem,1);


iK = reshape(kron(edofMat(1:nelem,1:24),ones(24,1))',576*nelem,1);
jK = reshape(kron(edofMat(1:nelem,1:24),ones(1,24))',576*nelem,1);


% Prologation operators
Pu=cell(nl-1,1);
for l = 1:nl-1
    [Pu{l,1}] = prepcoarse(nelz/2^(l-1),nely/2^(l-1),nelx/2^(l-1));
end
% Define loads and supports (cantilever)
%F = sparse(3*nodenrs(1:nely+1,1,nelx+1),1,-sin((0:nely)/nely*pi),ndof(1),1); % Sine load, bottom right
%F = sparse(3*nodenrs(1:nely+1,1,nelx+1),1,[-0.5; -ones(nely-1,1); -0.5],ndof(1),1); % constant load
F = sparse(3*nodenrs(1:nely+1,1,nelx+1),1,-ones(nely+1,1),ndof(1),1); % constant loa

U = zeros(ndof,1);
fixeddofs = 1:3*(nely+1)*(nelz+1);
% Null space elimination of supports
N = ones(ndof,1); N(fixeddofs) = 0; Null = spdiags(N,0,ndof,ndof);
%% PREPARE FILTER
iH = ones(nelx*nely*nelz*(2*(ceil(rmin)-1)+1)^3,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
    for k1 = 1:nelz
        for j1 = 1:nely
            e1 = (i1-1)*nely*nelz + (k1-1)*nely + j1;
            for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
                for k2 = max(k1-(ceil(rmin)-1),1):min(k1+(ceil(rmin)-1),nelz)
                    for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                        e2 = (i2-1)*nely*nelz + (k2-1)*nely + j2;
                        k = k + 1;
                        iH(k) = e1;
                        jH(k) = e2;
                        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2));
                    end
                end
            end
        end
    end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);
%% INITIALIZE ITERATION
x = volfrac*ones(nelem(1),1);
xPhys = x;
loop = 0;
change = 1;


%% START ITERATION
while change > 1e-2 && loop < 100
    loop = loop+1;
    %% FE-ANALYSIS
    
    K = cell(nl,1);
    sK = reshape(KE(:)*(Emin+xPhys(:)'.^penal*(E0-Emin)),576*nelem,1);
    K{1,1} = sparse(iK,jK,sK);
    K{1,1} = Null'*K{1,1}*Null - (Null-speye(ndof,ndof));
    for l = 1:nl-1
        K{l+1,1} = Pu{l,1}'*(K{l,1}*Pu{l,1});
    end
    
    Lfac = chol(K{nl,1},'lower'); Ufac = Lfac';
    [cgiters,cgres,U] = mgcg(K,F,U,Lfac,Ufac,Pu,nl,1,...
        cgtol,cgmax);

    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    ce = sum(U(edofMat)*KE.*U(edofMat),2);
    c = sum(sum((Emin+xPhys.^penal*(E0-Emin)).*ce));
    dc = -penal*(E0-Emin)*xPhys.^(penal-1).*ce;
    dv = ones(nelem(1),1);
    %% FILTERING/MODIFICATION OF SENSITIVITIES
    
    if ft == 1
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
    elseif ft == 2
        dc(:) = H*(dc(:)./Hs);
        dv(:) = H*(dv(:)./Hs);
    end
    %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    g = mean(xPhys(:))-volfrac;
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-6
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
        gt=g+sum((dv(:).*(xnew(:)-x(:))));
	if gt>0, l1 = lmid; else l2 = lmid; end
    end
    change = max(abs(xnew(:)-x(:)));
    x = xnew;
    
    %% FILTERING OF DESIGN VARIABLES½1
    if ft == 1,         xPhys = xnew;
    elseif ft == 2,     xPhys(:) = (H*xnew(:))./Hs;
    end   
    %% PRINT RESULTS
    fprintf(' It.:%4i Obj.:%6.3e Vol.:%6.3e ch.:%4.2e relres: %4.2e iters: %4i \n',...
        loop,c,mean(xPhys(:)),change,cgres,cgiters);
    if mod(loop,10)==0
        %% PLOT
        isovals = shiftdim(reshape(xPhys,nely,nelz,nelx),2);
        isovals = smooth3(isovals,'box',1);
        patch(isosurface(isovals,0.5),'FaceColor',[0 0 1],'EdgeColor','none');
        patch(isocaps(isovals,0.5),'FaceColor',[1 0 0],'EdgeColor','none');
        view(3); axis equal tight off; camlight; drawnow
    end
end
%% PLOT
isovals = shiftdim(reshape(xPhys,nely,nelz,nelx),2);
isovals = smooth3(isovals,'box',1);
patch(isosurface(isovals,0.5),'FaceColor',[0 0 1],'EdgeColor','none');
patch(isocaps(isovals,0.5),'FaceColor',[1 0 0],'EdgeColor','none');
view(3); axis equal tight off; camlight;
end


%% FUNCTION mgcg - MULTIGRID PRECONDITIONED CONJUGATE GRADIENTS
function [i,relres,u] = mgcg(A,b,u,Lfac,Ufac,Pu,nl,nswp,...
    tol,maxiter)
r = b - A{1,1}*u;

res0 = norm(b); 
% Jacobi smoother
omega = 0.6;
invD = cell(nl-1,1);
for l = 1:nl-1
    invD{l,1}= 1./spdiags(A{l,1},0);
end
for i=1:1e6 
    z = VCycle(A,r,Lfac,Ufac,Pu,1,nl,invD,omega,nswp);
    %z = invD{1,1}.*r;
    rho = r'*z;
    
    if i==1
        p=z;
    else
        beta=rho/rho_p;
        p=beta*p+z;
    end
    q=A{1,1}*p;
    dpr=p'*q;
    alpha=rho/dpr;
    u=u+alpha*p;
    r=r-alpha*q;
    rho_p=rho;
    relres=norm(r)/res0;
    if relres<tol || i>=maxiter
        break
    end
end
% fprintf('it.: %d, rho: %e \n',i,relres);
end
%% FUNCTION VCycle - COARSE GRID CORRECTION
function z = VCycle(A,r,Lfac,Ufac,Pu,l,nl,invD,omega,nswp)
z = 0*r;
z = smthdmpjac(z,A{l,1},r,invD{l,1},omega,nswp);
d = r - A{l,1}*z;
dh2 = Pu{l,1}'*d;
if (nl == l+1)
    vh2 = Ufac \ (Lfac \ dh2);
else
    vh2 = VCycle(A,dh2,Lfac,Ufac,Pu,l+1,nl,invD,omega,nswp);
end
v = Pu{l,1}*vh2;
z = z + v;
z = smthdmpjac(z,A{l,1},r,invD{l,1},omega,nswp);
end
%% FUNCTIODN smthdmpjac - DAMPED JACOBI SMOOTHER
function [u] = smthdmpjac(u,A,b,invD,omega,nswp)
for i = 1:nswp
    u = u - omega*invD.*(A*u) + omega*invD.*b;
end
end
%% FUNCTION Ke3D - ELEMENT STIFFNESS MATRIX
function KE = Ke3D(nu)
C = [2/9 1/18 1/24 1/36 1/48 5/72 1/3 1/6 1/12];
A11 = [-C(1) -C(3) -C(3) C(2) C(3) C(3); -C(3) -C(1) -C(3) -C(3) -C(4) -C(5);...
    -C(3) -C(3) -C(1) -C(3) -C(5) -C(4); C(2) -C(3) -C(3) -C(1) C(3) C(3);...
    C(3) -C(4) -C(5) C(3) -C(1) -C(3); C(3) -C(5) -C(4) C(3) -C(3) -C(1)];
B11 = [C(7) 0 0 0 -C(8) -C(8); 0 C(7) 0 C(8) 0 0; 0 0 C(7) C(8) 0 0;...
    0 C(8) C(8) C(7) 0 0; -C(8) 0 0 0 C(7) 0; -C(8) 0 0 0 0 C(7)];
A22 = [-C(1) -C(3) C(3) C(2) C(3) -C(3); -C(3) -C(1) C(3) -C(3) -C(4) C(5);...
    C(3) C(3) -C(1) C(3) C(5) -C(4); C(2) -C(3) C(3) -C(1) C(3) -C(3);...
    C(3) -C(4) C(5) C(3) -C(1) C(3); -C(3) C(5) -C(4) -C(3) C(3) -C(1)];
B22 = [C(7) 0 0 0 -C(8) C(8); 0 C(7) 0 C(8) 0 0; 0 0 C(7) -C(8) 0 0;...
    0 C(8) -C(8) C(7) 0 0; -C(8) 0 0 0 C(7) 0; C(8) 0 0 0 0 C(7)];
A12 = [C(6) C(3) C(5) -C(4) -C(3) -C(5); C(3) C(6) C(5) C(3) C(2) C(3);...
    -C(5) -C(5) C(4) -C(5) -C(3) -C(4); -C(4) C(3) C(5) C(6) -C(3) -C(5);...
    -C(3) C(2) C(3) -C(3) C(6) C(5); C(5) -C(3) -C(4) C(5) -C(5) C(4)];
B12 = [-C(9) 0 -C(9) 0 C(8) 0; 0 -C(9) -C(9) -C(8) 0 -C(8); C(9) C(9) -C(9) 0 C(8) 0;...
    0 -C(8) 0 -C(9) 0 C(9); C(8) 0 -C(8) 0 -C(9) -C(9); 0 C(8) 0 -C(9) C(9) -C(9)];
A13 = [-C(4) -C(5) -C(3) C(6) C(5) C(3); -C(5) -C(4) -C(3) -C(5) C(4) -C(5);...
    C(3) C(3) C(2) C(3) C(5) C(6); C(6) -C(5) -C(3) -C(4) C(5) C(3);...
    C(5) C(4) -C(5) C(5) -C(4) -C(3); -C(3) C(5) C(6) -C(3) C(3) C(2)];
B13 = [0 0 C(8) -C(9) -C(9) 0; 0 0 C(8) C(9) -C(9) C(9); -C(8) -C(8) 0 0 -C(9) -C(9);...
    -C(9) C(9) 0 0 0 -C(8); -C(9) -C(9) C(9) 0 0 C(8); 0 -C(9) -C(9) C(8) -C(8) 0];
A14 = [C(2) C(5) C(5) C(4) -C(5) -C(5); C(5) C(2) C(5) C(5) C(6) C(3);...
    C(5) C(5) C(2) C(5) C(3) C(6); C(4) C(5) C(5) C(2) -C(5) -C(5);...
    -C(5) C(6) C(3) -C(5) C(2) C(5); -C(5) C(3) C(6) -C(5) C(5) C(2)];
B14 = [-C(9) 0 0 -C(9) C(9) C(9); 0 -C(9) 0 -C(9) -C(9) 0; 0 0 -C(9) -C(9) 0 -C(9);...
    -C(9) -C(9) -C(9) -C(9) 0 0; C(9) -C(9) 0 0 -C(9) 0; C(9) 0 -C(9) 0 0 -C(9)];
A23 = [C(2) C(5) -C(5) C(4) -C(5) C(5); C(5) C(2) -C(5) C(5) C(6) -C(3);...
    -C(5) -C(5) C(2) -C(5) -C(3) C(6); C(4) C(5) -C(5) C(2) -C(5) C(5);...
    -C(5) C(6) -C(3) -C(5) C(2) -C(5); C(5) -C(3) C(6) C(5) -C(5) C(2)];
B23 = [-C(9) 0 0 -C(9) C(9) -C(9); 0 -C(9) 0 -C(9) -C(9) 0; 0 0 -C(9) C(9) 0 -C(9);...
    -C(9) -C(9) C(9) -C(9) 0 0; C(9) -C(9) 0 0 -C(9) 0; -C(9) 0 -C(9) 0 0 -C(9)];
KE = 1/(1+nu)/(2*nu-1)*([A11 A12 A13 A14; A12' A22 A23 A13'; A13' A23' A22 A12'; A14' A13 A12 A11] +...
    nu*[B11 B12 B13 B14; B12' B22 B23 B13'; B13' B23' B22 B12'; B14' B13 B12 B11]);
end
%% FUNCTION prepcoarse - PREPARE MG PROLONGATION OPERATOR
function [Pu] = prepcoarse(nex,ney,nez)
% Assemble state variable prolongation
maxnum = nex*ney*nez*20;
iP = zeros(maxnum,1); jP = zeros(maxnum,1); sP = zeros(maxnum,1);
nexc = nex/2; neyc = ney/2; nezc = nez/2;
% Weights for fixed distances to neighbors on a structured grid 
vals = [1,0.5,0.25,0.125];
cc = 0;
for nx = 1:nexc+1
    for ny = 1:neyc+1
        for nz = 1:nezc+1
            col = (nx-1)*(neyc+1)+ny+(nz-1)*(neyc+1)*(nexc+1); 
            % Coordinate on fine grid
            nx1 = nx*2 - 1; ny1 = ny*2 - 1; nz1 = nz*2 - 1;
            % Loop over fine nodes within the rectangular domain
            for k = max(nx1-1,1):min(nx1+1,nex+1)
                for l = max(ny1-1,1):min(ny1+1,ney+1)
                    for h = max(nz1-1,1):min(nz1+1,nez+1)
                        row = (k-1)*(ney+1)+l+(h-1)*(nex+1)*(ney+1); 
                        % Based on squared dist assign weights: 1.0 0.5 0.25 0.125
                        ind = 1+((nx1-k)^2+(ny1-l)^2+(nz1-h)^2);
                        cc=cc+1; iP(cc)=3*row-2; jP(cc)=3*col-2; sP(cc)=vals(ind);
                        cc=cc+1; iP(cc)=3*row-1; jP(cc)=3*col-1; sP(cc)=vals(ind);
                        cc=cc+1; iP(cc)=3*row; jP(cc)=3*col; sP(cc)=vals(ind);
                    end
                end
            end
        end
    end
end
% Assemble matrices
Pu = sparse(iP(1:cc),jP(1:cc),sP(1:cc));
end















