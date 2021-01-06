%% Example code for simulated expected free energy precision (beta/gamma) updates
% (associated with dopamine in the neural process theory)

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

% Here you can set the number of policies and the distributions that
% contribute to prior and posterior policy precision

E = [1 1 1 1 1]';                             % Set a fixed-form prior distribution 
                                              % over policies (habits)

G = [12.505 9.51 12.5034 12.505 12.505]';     % Set an example expected 
                                              % free energy distribution over policies

F = [17.0207 1.7321 1.7321 17.0387 17.0387]'; % Set an example variational 
                                              % free energy distribution over 
                                              % policies after a new observation


gamma_0 = 1;                 % Starting expected free energy precision value
gamma = gamma_0;             % Initial expected free energy precision to be updated
beta_prior = 1/gamma;        % Initial prior on expected free energy precision
beta_posterior = beta_prior; % Initial posterior on expected free energy precision
 

for ni = 1:16 % number of variational updates (16)

    % calculate prior and posterior over policies (see main text for 
    % explanation of equations) 

    pi_0 = exp(log(E) - gamma*G)/sum(exp(log(E) - gamma*G)); % prior over policies

    pi_posterior = exp(log(E) - gamma*G - F)/sum(exp(log(E) - gamma*G - F)); % posterior 
                                                                             % over policies
    % calculate expected free energy precision 

    beta_update = (pi_posterior - pi_0)'*-G; % calculate update

    dFdg = beta_posterior - beta_prior + beta_update; % calculate beta gradient 
                                                      % with respect to F
    
    beta_posterior = beta_posterior - dFdg/2; % update posterior precision 
                                              % estimate

    gamma = 1/beta_posterior; % update expected free energy precision

    % simulate dopamine responses

    n = ni;

    gamma_dopamine(n,1) = gamma; % simulated neural encoding of precision
                                 % (beta_posterior^-1) at each iteration of 
                                 % variational updating

    policies_neural(:,n) = pi_posterior; % neural encoding of posterior over policies at 
                                         % each iteration of variational updating

    pi_posterior = pi_posterior;         % record posterior over policies 

end 

%% Show Results

disp(' ');
disp('Policy Prior:');
disp(pi_0);
disp(' ');
disp('Policy Posterior:');
disp(pi_posterior);
disp(' ');
disp('Policy Difference Vector:');
disp(pi_posterior-pi_0);
disp(' ');
disp('Negative Expected Free Energy:');
disp(-G);
disp(' ');
disp('Prior G Precision (Prior Gamma):');
disp(gamma_0);
disp(' ');
disp('Posterior G Precision (Prior Gamma):');
disp(gamma);
disp(' ');

plot(gamma_dopamine);
title('Expected Free Energy Precision Updates (gamma)');
xlabel('Updates');
ylabel('\gamma');

% uncomment if you want to show firing rates encoding beliefs about each
% policy (columns = policies, rows = updates over time)

% disp('Firing rates encoding beliefs over policies:');
% disp(policies_neural');
% disp(' ');
