%% Example code for simulating state and outcome prediction errors

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all
%% set up model to calculate state prediction errors
% This minimizes variational free energy (keeps posterior beliefs accurate 
% while also keeping them as close as possible to prior beliefs)

A = [.8 .4;       
     .2 .6];         % Likelihood

B_t1 = [.9 .2; 
        .1 .8];      % Transition prior from previous timestep
    
B_t2 = [.2 .3; 
        .8 .7];      % Transition prior from current timestep
    
o = [1 0]';          % Observation

s_pi_tau = [.5 .5]'; % Prior distribution over states. Note that we here
                     % use the same value for s_pi_tau-1, s_pi_tau, and 
                     % s_pi_tau+1. But this need not be the case.
                     
s_pi_tau_minus_1 = [.5 .5]';

s_pi_tau_plus_1 = [.5 .5]';

v_0 = log(s_pi_tau);      % Depolarization term (initial value)

B_t2_cross_intermediate = B_t2';  % Transpose B_t2

B_t2_cross = spm_softmax(B_t2_cross_intermediate); % Normalize columns in transposed B_t2
                                            
%% Calculate state prediction error (single iteration)

state_error = 1/2*(log(B_t1*s_pi_tau_minus_1)+log(B_t2_cross*s_pi_tau_plus_1))...
              +log(A'*o)-log(s_pi_tau); % state prediction error

v = v_0 + state_error;      % Depolarization

s = (exp(v)/sum(exp(v)));        % Updated distribution over states


disp(' ');
disp('Prior Distribution over States:');
disp(s_pi_tau);
disp(' ');
disp('State Prediction Error:');
disp(state_error);
disp(' ');
disp('Depolarization:');
disp(v);
disp(' ');
disp('Posterior Distribution over States:');
disp(s);
disp(' ');

return
%% set up model to calculate outcome prediction errors 
% This minimizes expected free energy (maximizes reward and
% information-gain)

clear
close all

% Calculate risk (reward-seeking) term under two policies

A = [.9 .1;
     .1 .9];   % Likelihood
 
S1 = [.9 .1]'; % States under policy 1
S2 = [.5 .5]'; % States under policy 2

C = [1 0]';    % Preferred outcomes

o_1 = A*S1;    % Predicted outcomes under policy 1
o_2 = A*S2;    % Predicted outcomes under policy 2
z = exp(-16);  % Small number added to preference distribution to avoid log(0)

risk_1 = dot(o_1,log(o_1) - log(C+z)); % Risk under policy 1

risk_2 = dot(o_2,log(o_2) - log(C+z)); % Risk under policy 2 

disp(' ');
disp('Risk Under Policy 1:');
disp(risk_1);
disp(' ');
disp('Risk Under Policy 2:');
disp(risk_2);
disp(' ');


% Calculate ambiguity (information-seeking) term under two policies

A = [.4 .2;
     .6 .8];   % Likelihood
 
s1 = [.9 .1]'; % States under policy 1
s2 = [.1 .9]'; % States under policy 2


ambiguity_1 = -dot(diag(A'*log(A)),s1); % Ambiguity under policy 1

ambiguity_2 = -dot(diag(A'*log(A)),s2); % Ambiguity under policy 2

disp(' ');
disp('Ambiguity Under Policy 1:');
disp(ambiguity_1);
disp(' ');
disp('Ambiguity Under Policy 2:');
disp(ambiguity_2);
disp(' ');
