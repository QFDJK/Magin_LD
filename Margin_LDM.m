function  out = Margin_LDM(X,Y,pars)

if nargin<3;               pars  = [];                             end
if isfield(pars,'maxit');  maxit = pars.maxit; else; maxit = 5e2;  end
if isfield(pars,'alpha_P');  alpha_P = pars.alpha_P; else; alpha_P =0.5; end
if isfield(pars,'tol');    tol   = pars.tol;   else; tol   = 1e-4; end
if isfield(pars,'C_p');      C_p   = pars.C_p;     else; C_p     = 1e2;    end
if isfield(pars,'mu1');    mu1 = pars.mu1;     else; mu1  = 10;    end
if isfield(pars,'mu2');    mu2 = pars.mu1;     else; mu2  = 10;    end
if isfield(pars,'sigma_k');    sigma_k = pars.sigma_k;     else; sigma_k  = 1e1;  end
if isfield(pars,'max_xi');    max_xi = pars.max_xi;     else; max_xi  = 3;  end


tic;
y = Y;
A = X.*Y;
[m, n] = size(A);
e_m = ones(m,1);

xi_k = zeros(m,1);

wk = ones(n,1);
bk =0;
sk = zeros(m,1);

I = eye(m);  % Identity matrix of size n

if isfield(pars,'Q')
    Q = pars.Q;
else
    Q  = X' * (2 * mu1 * (m * I - y * y') / m^2) * X;
end

%Q = X' * (2 * mu1 * (m * I - y * y') / m^2) * X;

% Calculate G1 and G2
G1 = (mu2 / m) *X'* y;
G2 = (2 / m^2) * X' * (m*e_m - y * (e_m' * y));

% Calculate lambda1 and lambda2
lambda1 = 2 * mu1 * (1 - (e_m' * y)^2/ (m^2));
lambda2 = mu2 * e_m' * y / m;

ATA = A'*A;
ATY = A'*Y;
ATE = A'*e_m;
YTE = Y'*e_m;
k =1;
Reek = [];
reek =0;
O_o =0;
reek0 = 1000;



tic;
while k<maxit
    sigma_C = C_p/sigma_k;
    % solve xi_k-subproblem
    rep_exp = exp(-alpha_P*xi_k);
    R_exp = rep_exp;
    e_xi_k = (e_m - alpha_P*xi_k);
    term_1 = xi_k-sigma_C*R_exp.*(e_xi_k);
    term_2 = A*wk + bk*Y - e_m -sk;
    xi_k = 1/2*(term_1-term_2);
    xi_k = max(xi_k,0);
    xi_k =min( max(xi_k,0),max_xi);
    % solve w_k-subproblem
    term_1 = 1*(eye(n)+sigma_k*(1*eye(n)+ATA)+Q);
    term_2 = G1+bk*G2;
    term_3 = sigma_k*(bk*ATY-ATE + A'*xi_k-A'*sk+wk);
    wk = -term_1\(term_2+term_3);
    wk = 1*wk/norm(wk);

    % solve b_k-subproblem
    term_1 = (lambda1+sigma_k*(1+m));
    term_2 = wk'*G2 -lambda2+sigma_k*( ATY'* wk - YTE +Y'*xi_k - Y'*sk)+bk ;
    bk = -term_2/term_1;


    % solve s_k-subproblem
    sk = (A*wk+bk*Y-e_m+xi_k);
    sk =1/2 *(A*wk+bk*Y-e_m+xi_k + sk);
    sko =sk;
    xi_k = -sk;
    sk = max(sk,0);


    sigma_k = min(1.25*sigma_k,1e4);

    O_k= [xi_k;wk;bk;sk];
    reek = norm(O_k- O_o)^2;
    if reek<tol
        break
    end


    reek0 =reek;
    Reek =[Reek;reek];
    O_o =  O_k;
    k =k+1;
end

out.iter = k;
out.time = toc;
out.w   = wk;
out.b   = bk;
out.reek   = Reek;
out.xi   = xi_k;
end

