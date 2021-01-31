%% Calculating novelty term in expected free energy when learning 'A' matrix concentration parameters
% (which drives parameter exploration)

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all

%-- 'a' = concentration parameters for likelihood matrix 'A'

% small concentration parameter values 
a1 = [.25  1;  
      .75  1]; 
  
% intermediate concentration parameter values 
a2 = [2.5  10;
      7.5  10]; 
  
% large concentration parameter values  
a3 = [25  100;
      75  100]; 
  
% normalize columns in 'a' to get likelihood matrix 'A' (see col_norm
% function at the end of script)
A1 = col_norm(a1);
A2 = col_norm(a2);
A3 = col_norm(a3);
  
% calculate 'a_sum' 
a1_sum = [a1(1,1)+a1(2,1)  a1(1,2)+a1(2,2);
          a1(1,1)+a1(2,1)  a1(1,2)+a1(2,2)]; 
  
a2_sum = [a2(1,1)+a2(2,1)  a2(1,2)+a2(2,2);
          a2(1,1)+a2(2,1)  a2(1,2)+a2(2,2)];
      
a3_sum = [a3(1,1)+a3(2,1)  a3(1,2)+a3(2,2);
          a3(1,1)+a3(2,1)  a3(1,2)+a3(2,2)];

% element wise inverse for 'a' and 'a_sum'
inv_a1 =  [1/a1(1,1)  1/a1(1,2);
           1/a1(2,1)  1/a1(2,2)];
       
inv_a2 =  [1/a2(1,1)  1/a2(1,2);
           1/a2(2,1)  1/a2(2,2)];
       
inv_a3 =  [1/a3(1,1)  1/a3(1,2);
           1/a3(2,1)  1/a3(2,2)];
       
inv_a1_sum =  [1/a1_sum(1,1)  1/a1_sum(1,2);
               1/a1_sum(2,1)  1/a1_sum(2,2)];
       
inv_a2_sum =  [1/a2_sum(1,1)  1/a2_sum(1,2);
               1/a2_sum(2,1)  1/a2_sum(2,2)];
       
inv_a3_sum =  [1/a3_sum(1,1)  1/a3_sum(1,2);
               1/a3_sum(2,1)  1/a3_sum(2,2)];
      
% 'W' term for 'a' matrix
W1 = .5*(inv_a1-inv_a1_sum);
W2 = .5*(inv_a2-inv_a2_sum);
W3 = .5*(inv_a3-inv_a3_sum);

% beliefs over states under a policy at a time point
s_pi_tau = [.9 .1]';

% predictive posterior over outcomes (A*s_pi_tau = predicted o_pi_tau)
A1s = A1*s_pi_tau;
A2s = A2*s_pi_tau;
A3s = A3*s_pi_tau;

% W term multiplied by beliefs over states under a policy at a time point
W1s = W1*s_pi_tau;
W2s = W2*s_pi_tau;
W3s = W3*s_pi_tau;

% compute novelty using dot product function
Novelty_smallCP = dot((A1s),(W1s));
Novelty_intermediateCP = dot((A2s),(W2s));
Novelty_largeCP = dot((A3s),(W3s));


% show results
disp(' ');
disp('Novelty term for small concentration parameter values:');
disp(Novelty_smallCP);
disp(' ');
disp('Novelty term for intermediate concentration parameter values:');
disp(Novelty_intermediateCP);
disp(' ');
disp('Novelty term for large concentration parameter values:');
disp(Novelty_largeCP);
disp(' ');


%% function for normalizing 'a' to get likelihood matrix 'A'
function A_normed = col_norm(A_norm)
aa = A_norm; 
norm_constant = sum(aa,1); % create normalizing constant from sum of columns
aa = aa./norm_constant; % divide columns by constant
A_normed = aa;
end 
