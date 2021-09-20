%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- Message Passing Examples--%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Supplementary Code for: A Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith and Christopher J. Whyte
% We also acknowledge Samuel Taylor for contributing to this example code

% This script provides two examples of (marginal) message passing, based on
% the steps described in the main text. Each of the two examples (sections)
% need to be run separately. The first example fixes all observed
% variables immediately and does not include variables associated with the
% neural process theory. The second example provides observations
% sequentially and also adds in the neural process theory variables. To
% remind the reader, the message passing steps in the main text are:

% 	1. Initialize the values of the approximate posteriors q(s_(?,?) ) 
%      for all hidden variables (i.e., all edges) in the graph. 
% 	2. Fix the value of observed variables (here, o_?).
% 	3. Choose an edge (V) corresponding to the hidden variable you want to 
%      infer (here, s_(?,?)).
% 	4. Calculate the messages, ?(s_(?,?)), which take on values sent by 
%      each factor node connected to V.
% 	5. Pass a message from each connected factor node N to V (often written 
%      as ?_(N?V)). 
% 	6. Update the approximate posterior represented by V according to the 
%      following rule: q(s_(?,?) )? ? ?(s_(?,?))? ?(s_(?,?)). The arrow 
%      notation here indicates messages from two different factors arriving 
%      at the same edge. 
%       6A. Normalize the product of these messages so that q(s_(?,?) ) 
%           corresponds to a proper probability distribution. 
%       6B. Use this new q(s_(?,?) ) to update the messages sent by 
%           connected factors (i.e., for the next round of message passing).
% 	7. Repeat steps 4-6 sequentially for each edge.
% 	8. Steps 3-7 are then repeated until the difference between updates 
%      converges to some acceptably low value (i.e., resulting in stable 
%      posterior beliefs for all edges). 

%% Example 1: Fixed observations and message passing steps

% This section carries out marginal message passing on a graph with beliefs
% about states at two time points. In this first example, both observations 
% are fixed from the start (i.e., there are no ts as in full active inference
% models with sequentially presented observations) to provide the simplest
% example possible. We also highlight where each of the message passing
% steps described in the main text are carried out.

% Note that some steps (7 and 8) appear out of order when they involve loops that
% repeat earlier steps

% Specify generative model and initialize variables

rng('shuffle')

clear
close all

% priors
D = [.5 .5]';

% likelihood mapping
A = [.9 .1;
     .1 .9];
 
% transitions
 B = [1 0;
      0 1];

% number of timesteps
T = 2;

% number of iterations of message passing
NumIterations = 16;

% initialize posterior (Step 1)
for t = 1:T 
    Qs(:,t) = [.5 .5]';
end 

% fix observations (Step 2)
o{1} = [1 0]';
o{2} = [1 0]';

% iterate a set number of times (alternatively, until convergence) (Step 8)
for Ni = 1:NumIterations
    % For each edge (hidden state) (Step 7)
    for tau = 1:T
        % choose an edge (Step 3)
        q = nat_log(Qs(:,tau));
        
        % compute messages sent by D and B (Steps 4) using the posterior
        % computed in Step 6B
        if tau == 1 % first time point
            lnD = nat_log(D);                % Message 1
            lnBs = nat_log(B'*Qs(:,tau+1));  % Message 2
        elseif tau == T % last time point
            lnBs = nat_log(B*Qs(:,tau-1));  % Message 1
        end 
        
        % likelihood (Message 3)
        lnAo = nat_log(A'*o{tau});
        
        % Steps 5-6 (Pass messages and update the posterior)
        % Since all terms are in log space, this is addition instead of
        % multiplication. This corresponds to  equation 16 in the main
        % text (within the softmax)
        if tau == 1
            q = .5*lnD + .5*lnBs + lnAo;
        elseif tau == T
            q = .5*lnBs + lnAo;
        end
        
        % normalize using a softmax function to find posterior (Step 6A)
        Qs(:,tau) = (exp(q)/sum(exp(q))); 
        qs(Ni,:,tau) = Qs(:,tau); % store value for each iteration
    end % Repeat for remaining edges (Step 7)
end % Repeat until convergence/for fixed number of iterations (Step 8)

Qs; % final posterior beliefs over states

disp(' ');
disp('Posterior over states q(s) in example 1:');
disp(' ');
disp(Qs);

figure

% firing rates (traces)
qs_plot = [D' D';qs(:,:,1) qs(:,:,2)]; % add prior to starting value
plot(qs_plot)
title('Example 1: Approximate Posteriors (1 per edge per time point)')
ylabel('q(s_t_a_u)','FontSize',12)
xlabel('Message passing iterations','FontSize',12)


%% Example 2: Sequential observations and simulation of firing rates and ERPs

% This script performs state estimation using the message passing 
% algorithm introduced in Parr, Markovic, Kiebel, & Friston (2019).
% This script can be thought of as the full message passing solution to 
% problem 2 in the pencil and paper exercises. It also generates
% simulated firing rates and ERPs in the same manner as those shown in
% figs. 8, 10, 11, 14, 15, and 16. Unlike example 1, observations are
% presented sequentially (i.e., two ts and two taus).

% Specify generative model and initialise variables

rng('shuffle')

clear

% priors
D = [.5 .5]';

% likelihood mapping
A = [.9 .1;
     .1 .9];
 
% transitions
 B = [1 0;
      0 1];

% number of timesteps
T = 2;

% number of iterations of message passing
NumIterations = 16;

% initialize posterior (Step 1)
for t = 1:T 
    Qs(:,t) = [.5 .5]';
end 

% fix observations sequentially (Step 2)
o{1,1} = [1 0]';
o{1,2} = [0 0]';
o{2,1} = [1 0]';
o{2,2} = [1 0]';

% Message Passing

for t = 1:T 
    for Ni = 1:NumIterations % (Step 8 loop of VMP)
        for tau = 1:T % (Step 7 loop of VMP)
            
            % initialise depolarization variable: v = ln(s)
            % choose an edge (Step 3 of VMP)
            v = nat_log(Qs(:,t));
            
            % get correct D and B for each time point (Steps 4-5 of VMP)
            % using using the posterior computed in Step 6B
            if tau == 1 % first time point
                % past (Message 1)
                lnD = nat_log(D);
                
                % future (Message 2)
                lnBs = nat_log(B'*Qs(:,tau+1));
            elseif tau == T % last time point
                % no contribution from future (only Message 1)
                lnBs  = nat_log(B*Qs(:,tau-1));
            end 
            % likelihood (Message 3)
            lnAo = nat_log(A'*o{t,tau});
            
            % calculate state prediction error: equation 24
            if tau == 1
                epsilon(:,Ni,t,tau) = .5*lnD + .5*lnBs + lnAo - v;
            elseif tau == T
                epsilon(:,Ni,t,tau) = .5*lnBs + lnAo - v;
            end 
            
            % (Step 6 of VMP)
            % update depolarization variable: equation 25
            v = v + epsilon(:,Ni,t,tau); 
            % normalize using a softmax function to find posterior:
            % equation 26 (Step 6A of VMP)
            Qs(:,tau) = (exp(v)/sum(exp(v)));
            % store Qs for firing rate plots
            xn(Ni,:,tau,t) = Qs(:,tau);
        end % Repeat for remaining edges (Step 7 of VMP)
    end % Repeat until convergence/for number of iterations (Step 8 of VMP)
end

Qs; % final posterior beliefs over states

disp(' ');
disp('Posterior over states q(s) in example 2:');
disp(' ');
disp(Qs);

% plots
    
% get firing rates into usable format
num_states = 2;
num_epochs = 2;
time_tau = [1 2 1 2;
            1 1 2 2];      
for t_tau = 1:size(time_tau,2)
    for epoch = 1:num_epochs
        % firing rate 
        firing_rate{epoch,t_tau} = xn(:,time_tau(1,t_tau),time_tau(2,t_tau),epoch);
        ERP{epoch,t_tau} = gradient(firing_rate{epoch,t_tau}')';
   end
end

% convert cells to matrices
firing_rate = spm_cat(firing_rate)';
firing_rate = [zeros(length(D)*T,1)+[D; D] full(firing_rate)]; % add prior for starting value
ERP = spm_cat(ERP);
ERP = [zeros(length(D)*T,1)'; ERP]; % add 0 for starting value

figure

% firing rates
imagesc(t,1:(num_states*num_epochs),64*(1 - firing_rate))
cmap = gray(256);
colormap(cmap)
title('Example 2: Firing rates (Darker = higher value)')
ylabel('Firing rate','FontSize',12)
xlabel('Message passing iterations','FontSize',12)

figure

% firing rates (traces)
plot(firing_rate')
title('Example 2: Firing rates (traces)')
ylabel('Firing rate','FontSize',12)
xlabel('Message passing iterations','FontSize',12)

figure

% ERPs/LFPs
plot(ERP)
title('Example 2: Event-related potentials')
ylabel('Response','FontSize',12)
xlabel('Message passing iterations','FontSize',12)

%% functions

% natural log that replaces zero values with very small values for numerical reasons.
function y = nat_log(x)
y = log(x+exp(-16));
end 
