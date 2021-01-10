%% Variational free energy calculation example

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte

clear all

True_observation = [1 0]'; % Set observation

%% Generative Model

% Specify Prior and likelihood

Prior = [.5 .5]; % Prior distribution p(s)

Likelihood = [.8 .2;
              .2 .8]; % Likelihood distribution p(o|s): columns=states, 
                      % rows = observations

Joint_probability = Prior.*Likelihood; % Joint probability distribution p(o,s)
Predicted_observation_probabilities = Prior*Likelihood; % Predicted observation 
                                                        % probabilities p(o)

True_observation_probability...
    = Predicted_observation_probabilities*True_observation; % Probability of true observation
True_observation_joint_probability...
    = True_observation'*Joint_probability; % Joint probability of states and true observation

%% Bayes theorem: exact posterior

% This is the distribution we want to approximate using variational 
% inference. In many practical applications, we can not solve for this 
% directly.

Posterior = True_observation_joint_probability...
    /True_observation_probability; % Posterior given true observation p(s|o)

disp(' ');
disp('Exact Posterior:');
disp(Posterior);
disp(' ');

%% Variational Free Energy

% Note: q(s) = approximate posterior belief: we want to get this as close as 
% possible to the true posterior p(s|o) after a new observation.

% Different decompisitions of Free Energy (F)

% 1. F=E_q(s)[ln(q(s)/p(o,s))]

% 2. F=E_q(s)[ln(q(s)/p(s))] - E_q(s)[ln(p(o|s))] % Complexity-accuracy
% version

% The first term can be interpreted as a complexity term (the KL divergence 
% between prior beliefs p(s) and approximate posterior beliefs q(s)). In 
% other words, how much beliefs have changed after a bew observation.

% The second term (excluding the minus sign) is the accuracy or (including the 
% minus sign) the entropy (= expected surprisal) of observations given 
% approximate posterior beliefs q(s). Written in this way 
% free-energy-minimisation is equivalent to a statistical Occam's razor, 
% where the agent tries to find the most accurate posterior belief that also
% changes its beliefs as little as possible.

% 3. F=E_q(s)[ln(q(s)) - ln(p(s|o)p(o))]

% 4. F=E_q(s)[ln(q(s)/p(s|o))] - ln(p(o))

% These two versions similarly show F in terms of a difference between
% q(s) and the true posterior p(s|o). Here we focus on #4.

% The first term is the KL divergence between the approximate posterior q(s)  
% and the unknown exact posterior p(s|o), also called the relative entropy. 

% The second term (excluding the minus sign) is the log evidence or (including 
% the minus sign) the surprisal of observations. Note that ln(p(o)) does 
% not depend on q(s), so its expectation value under q(s) is simply ln(p(o)).

% Since this term does not depend on q(s), minimizing free energy means that 
% q(s) comes to approximate p(s|o), which is our unknown, desired quantity.

% 5. F=E_q(s)[ln(q(s))-ln(p(o|s)p(s))]

% We will use this decomposition for convenience when doing variational
% inference below. Note how this decomposition is equivalent to the expression 
% shown in Figure 2 - F=E_q(s)(ln(q(s)/p(o,s)) - because ln(x)-ln(y) = ln(x/y)
% and p(o|s)p(s)=p(o,s)

%% Variational inference

Initial_approximate_posterior = Prior; % Initial approximate posterior distribution.
                                       % Set this to match generative model prior 

% Calculate F
Initial_F = Initial_approximate_posterior(1)*(log(Initial_approximate_posterior(1))...
    -log(True_observation_joint_probability(1)))+Initial_approximate_posterior(2)...
    *(log(Initial_approximate_posterior(2))-log(True_observation_joint_probability(2)));

Optimized_approximate_posterior = Posterior; % Set approximate distribution to true posterior

% Calculate F
Minimized_F = Optimized_approximate_posterior(1)*(log(Optimized_approximate_posterior(1))...
    -log(True_observation_joint_probability(1)))+Optimized_approximate_posterior(2)...
    *(log(Optimized_approximate_posterior(2))-log(True_observation_joint_probability(2)));

% We see that F is lower when the approximate posterior q(s) is closer to 
% the true distribution p(s|o)

disp(' ');
disp('Initial Approximate Posterior:');
disp(Initial_approximate_posterior);
disp(' ');

disp(' ');
disp('Initial Variational Free Energy:');
disp(Initial_F);
disp(' ');

disp(' ');
disp('Optimized Approximate Posterior:');
disp(Optimized_approximate_posterior);
disp(' ');

disp(' ');
disp('Minimized Variational Free Energy:');
disp(Minimized_F);
disp(' ');
