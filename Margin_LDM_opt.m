function  out = Margin_LDM_opt(X,Y,pars)

% Parse parameters
if nargin<3;               pars  = [];                           end
if isfield(pars,'maxit');  maxit = pars.maxit; else; maxit = 1e3;  end
% Using 'tau' to match the math, falling back to 'alpha_P' if provided
if isfield(pars,'tau');    tau = pars.tau;     else; tau = 0.01; end
if isfield(pars,'tol');    tol   = pars.tol;   else; tol   = 1e-4; end
if isfield(pars,'C_p');    C_p   = pars.C_p;   else; C_p   = 1e2;    end
if isfield(pars,'mu1');    mu1   = pars.mu1;   else; mu1  = 1;    end
if isfield(pars,'mu2');    mu2   = pars.mu2;   else; mu2  = 1;    end
if isfield(pars,'sigma_k');sigma_k = pars.sigma_k; else; sigma_k  = 1;  end
if isfield(pars,'max_xi'); max_xi = pars.max_xi; else; max_xi  = 5;  end

y = Y;

tic;

% Setup matrices and dimensions
A = X .* Y; % Assuming Y is a column vector and implicit expansion applies
[m, n] = size(A);
e_m = ones(m,1);

% Initialize variables
xi_k = 0.1*randn(m,1);
wk   = ones(n,1);
bk   = 0;

% Compute Q
if isfield(pars,'Q')
    Q = pars.Q;
else
    X_transpose = X'; 
    XtX = X_transpose * X;
    Xty = X_transpose * y;
    Xty_Xtyt = Xty * Xty';
    Q = (2 * mu1 / m^2) * (m * XtX - Xty_Xtyt);
end

% Calculate G1 and G2
G1 = (mu2 / m) * X' * y;
G2 = (2 / m^2) * X' * (m * e_m - y * (e_m' * y));

% Calculate lambda1 and lambda2
lambda1 = 2 * mu1 * (1 - (e_m' * y)^2 / (m^2));
lambda2 = mu2 * e_m' * y / m;

% Precompute A'A for w update
ATA = A' * A;

k = 1;
Reek = [];
O_o = 5e4 * ones(m + n + 1, 1); % Initial dummy state to ensure distance is large

while k < maxit
    
    % ---------------------------------------------------------
    % 1. Solve xi_k-subproblem (Eq 46)
    % ---------------------------------------------------------
    grad_f = zeros(m, 1);
    idx = xi_k > 0;
    % Gradient: C * 2 * xi * exp(-tau * xi^2) * (1 - tau * xi^2)
    grad_f(idx) = C_p * 2 .* xi_k(idx) .* exp(-tau * xi_k(idx).^2) .* (1 - tau * xi_k(idx).^2);
    
    % Update step for xi
    xi_k = 0.5 *(xi_k - (1/sigma_k) * grad_f - (A * wk + bk * y - e_m));
    

    % ---------------------------------------------------------
    % 2. Solve w_k-subproblem (Eq 47)
    % ---------------------------------------------------------
    term_inv = sigma_k * (eye(n) + ATA) + Q;
    num_w    =-G1 + bk * G2 + sigma_k * A' * (bk * y - e_m + xi_k) - sigma_k * wk;
    wk = - (term_inv \ num_w);
    wk = wk/norm(wk);
    % ---------------------------------------------------------
    % 3. Solve b_k-subproblem (Eq 48)
    % ---------------------------------------------------------
    num_b = lambda2 - G2' * wk - sigma_k * y' * (A * wk - e_m + xi_k);
    den_b = lambda1 + sigma_k * m;
    bk    = num_b / den_b;
    
    % ---------------------------------------------------------
    % Penalty Parameter Update & Convergence Check
    % ---------------------------------------------------------
    sigma_k = min(1.05 * sigma_k, 1e3);
    
    O_k = [xi_k; wk; bk];
    reek = norm(O_k - O_o);
    reek1 = reek / norm(O_o);
    Reek = [Reek; reek1];

    if reek1 < 1 * tol
        break
    end

    O_o = O_k;
    k = k + 1;
end

time = toc;

% % ---------------------------------------------------------
% Calculate F1 Score
% ---------------------------------------------------------
y_pred = sign(X * wk + bk);
y_pred(y_pred == 0) = 1; % Handle boundary zeros

TP = sum(y == 1 & y_pred == 1);
FP = sum(y == -1 & y_pred == 1);
FN = sum(y == 1 & y_pred == -1);

if (TP + FP) == 0; Precision = 0; else; Precision = TP / (TP + FP); end
if (TP + FN) == 0; Recall = 0;    else; Recall = TP / (TP + FN);    end
if (Precision + Recall) == 0
    F1 = 0;
else
    F1 = 2 * (Precision * Recall) / (Precision + Recall);
end
out.F1   = F1;

% ---------------------------------------------------------
% Construct Output
% ---------------------------------------------------------
out.iter = k;
out.time = time;
out.w    = wk;
out.b    = bk;
out.reek = Reek;
out.xi   = xi_k;
out.k    = k;

end