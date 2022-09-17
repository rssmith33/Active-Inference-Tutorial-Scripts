%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- Simplified Simulation Script --%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte

rng('shuffle')
close all
clear

% This code simulates a single trial of the explore-exploit task introduced 
% in the active inference tutorial using a stripped down version of the model
% inversion scheme implemented in the spm_MDP_VB_X.m script. 

% Note that this implementation uses the marginal message passing scheme
% described in (Parr et al., 2019), and will return very slightly 
% (negligably) different values than the spm_MDP_VB_X.m script in 
% simulation results.

% Parr, T., Markovic, D., Kiebel, S., & Friston, K. J. (2019). Neuronal 
% message passing using Mean-field, Bethe, and Marginal approximations. 
% Scientific Reports, 9, 1889.

%% Simulation Settings

% To simulate the task when prior beliefs (d) are separated from the 
% generative process, set the 'Gen_model' variable directly
% below to 1. To do so for priors (d), likelihoods (a), and habits (e), 
% set the 'Gen_model' variable to 2:

Gen_model = 1; % as in the main tutorial code, many parameters can be adjusted
               % in the model setup, within the explore_exploit_model
               % function starting on line 810. This includes, among
               % others (similar to in the main tutorial script):

% prior beliefs about context (d): alter line 866

% beliefs about hint accuracy in the likelihood (a): alter lines 986-988

% to adjust habits (e), alter line 1145

%% Specify Generative Model

MDP = explore_exploit_model(Gen_model);

% Model specification is reproduced at the bottom of this script (starting 
% on line 810), but see main tutorial script for more complete walk-through

%% Model Inversion to Simulate Behavior
%==========================================================================

% Normalize generative process and generative model
%--------------------------------------------------------------------------

% before sampling from the generative process and inverting the generative 
% model we need to normalize the columns of the matrices so that they can 
% be treated as a probability distributions

% generative process
A = MDP.A;         % Likelihood matrices
B = MDP.B;         % Transition matrices
C = MDP.C;         % Preferences over outcomes
D = MDP.D;         % Priors over initial states    
T = MDP.T;         % Time points per trial
V = MDP.V;         % Policies
beta = MDP.beta;   % Expected free energy precision
alpha = MDP.alpha; % Action precision
eta = MDP.eta;     % Learning rate
omega = MDP.omega; % Forgetting rate

A = col_norm(A);
B = col_norm(B);
D = col_norm(D);

% generative model (lowercase matrices/vectors are beliefs about capitalized matrices/vectors)

NumPolicies = MDP.NumPolicies; % Number of policies
NumFactors = MDP.NumFactors;   % Number of state factors

% Store initial paramater values of generative model for free energy 
% calculations after learning
%--------------------------------------------------------------------------

% 'complexity' of d vector concentration paramaters
if isfield(MDP,'d')
    for factor = 1:numel(MDP.d)
        % store d vector values before learning
        d_prior{factor} = MDP.d{factor};
        % compute "complexity" - lower concentration paramaters have
        % smaller values creating a lower expected free energy thereby
        % encouraging 'novel' behaviour 
        d_complexity{factor} = spm_wnorm(d_prior{factor});
    end 
end 

if isfield(MDP,'a')
    % complexity of a maxtrix concentration parameters
    for modality = 1:numel(MDP.a)
        a_prior{modality} = MDP.a{modality};
        a_complexity{modality} = spm_wnorm(a_prior{modality}).*(a_prior{modality} > 0);
    end
end  

% Normalise matrices before model inversion/inference
%--------------------------------------------------------------------------

% normalize A matrix
if isfield(MDP,'a')
    a = col_norm(MDP.a);
else 
    a = col_norm(MDP.A);
end 

% normalize B matrix
if isfield(MDP,'b')
    b = col_norm(MDP.b);
else 
    b = col_norm(MDP.B);
end 

% normalize C and transform into log probability
for ii = 1:numel(C)
    C{ii} = MDP.C{ii} + 1/32;
    for t = 1:T
        C{ii}(:,t) = nat_log(exp(C{ii}(:,t))/sum(exp(C{ii}(:,t))));
    end 
end 

% normalize D vector
if isfield(MDP,'d')
    d = col_norm(MDP.d);
else 
    d = col_norm(MDP.D);
end 

% normalize E vector
if isfield(MDP,'e')
    E = MDP.e;
    E = E./sum(E);
elseif isfield(MDP,'E')
    E = MDP.E;
    E = E./sum(E);
else
    E = col_norm(ones(NumPolicies,1));
    E = E./sum(E);
end

% Initialize variables
%--------------------------------------------------------------------------

% numbers of transitions, policies and states
NumModalities = numel(a);                    % number of outcome factors
NumFactors = numel(d);                       % number of hidden state factors
NumPolicies = size(V,2);                     % number of allowable policies
for factor = 1:NumFactors
    NumStates(factor) = size(b{factor},1);   % number of hidden states
    NumControllable_transitions(factor) = size(b{factor},3); % number of hidden controllable hidden states for each factor (number of B matrices)
end

% initialize the approximate posterior over states conditioned on policies
% for each factor as a flat distribution over states at each time point
for policy = 1:NumPolicies
    for factor = 1:NumFactors
        NumStates(factor) = length(D{factor}); % number of states in each hidden state factor
        state_posterior{factor} = ones(NumStates(factor),T,policy)/NumStates(factor); 
    end  
end 

% initialize the approximate posterior over policies as a flat distribution 
% over policies at each time point
policy_posteriors = ones(NumPolicies,T)/NumPolicies; 

% initialize posterior over actions
chosen_action = zeros(ndims(B),T-1);
    
% if there is only one policy
for factors = 1:NumFactors 
    if NumControllable_transitions(factors) == 1
        chosen_action(factors,:) = ones(1,T-1);
    end
end
MDP.chosen_action = chosen_action;

% initialize expected free energy precision (beta)
posterior_beta = 1;
gamma(1) = 1/posterior_beta; % expected free energy precision
    
% message passing variables
TimeConst = 4; % time constant for gradient descent
NumIterations  = 16; % number of message passing iterations

% Lets go! Message passing and policy selection 
%--------------------------------------------------------------------------

for t = 1:T % loop over time points  
    
    % sample generative process
    %----------------------------------------------------------------------
    
    for factor = 1:NumFactors % number of hidden state factors
        % Here we sample from the prior distribution over states to obtain the
        % state at each time point. At T = 1 we sample from the D vector, and at
        % time T > 1 we sample from the B matrix. To do this we make a vector 
        % containing the cumulative sum of the columns (which we know sum to one), 
        % generate a random number (0-1),and then use the find function to take 
        % the first number in the cumulative sum vector that is >= the random number. 
        % For example if our D vector is [.5 .5] 50% of the time the element of the 
        % vector corresponding to the state one will be >= to the random number. 

        % sample states 
        if t == 1
            prob_state = D{factor}; % sample initial state T = 1
        elseif t>1
            prob_state = B{factor}(:,true_states(factor,t-1),MDP.chosen_action(factor,t-1));
        end 
        true_states(factor,t) = find(cumsum(prob_state)>= rand,1);
    end 

    % sample observations
    for modality = 1:NumModalities % loop over number of outcome modalities
        outcomes(modality,t) = find(cumsum(a{modality }(:,true_states(1,t),true_states(2,t)))>=rand,1);
    end
    
    % express observations as a structure containing a 1 x observations 
    % vector for each modality with a 1 in the position corresponding to
    % the observation recieved on that trial
    for modality = 1:NumModalities
        vec = zeros(1,size(a{modality},1));
        index = outcomes(modality,t);
        vec(1,index) = 1;
        O{modality,t} = vec;
        clear vec
    end 
    
    % marginal message passing (minimize F and infer posterior over states)
    %----------------------------------------------------------------------
    
    for policy = 1:NumPolicies
        for Ni = 1:NumIterations % number of iterations of message passing  
            for factor = 1:NumFactors
            lnAo = zeros(size(state_posterior{factor})); % initialise matrix containing the log likelihood of observations
                for tau = 1:T % loop over tau
                    v_depolarization = nat_log(state_posterior{factor}(:,tau,policy)); % convert approximate posteriors into depolarisation variable v 
                    if tau<t+1 % Collect an observation from the generative process when tau <= t
                        for modal = 1:NumModalities % loop over observation modalities
                            % this line uses the observation at each tau to index
                            % into the A matrix to grab the likelihood of each hidden state
                            lnA = permute(nat_log(a{modal}(outcomes(modal,tau),:,:,:,:,:)),[2 3 4 5 6 1]);                           
                            for fj = 1:NumFactors
                                % dot product with state vector from other hidden state factors 
                                % (this is what allows hidden states to interact in the likleihood mapping)    
                                if fj ~= factor        
                                    lnAs = md_dot((lnA),state_posterior{fj}(:,tau),fj);
                                    clear lnA
                                    lnA = lnAs; 
                                    clear lnAs
                                end
                            end
                            lnAo(:,tau) = lnAo(:,tau) + lnA;
                        end
                    end
                    % 'forwards' and 'backwards' messages at each tau
                    if tau == 1 % first tau
                        lnD = nat_log(d{factor}); % forward message
                        lnBs = nat_log(B_norm(b{factor}(:,:,V(tau,policy,factor))')*state_posterior{factor}(:,tau+1,policy));% backward message
                    elseif tau == T % last tau                    
                        lnD  = nat_log((b{factor}(:,:,V(tau-1,policy,factor)))*state_posterior{factor}(:,tau-1,policy));% forward message 
                        lnBs = zeros(size(d{factor})); % backward message
                    else % 1 > tau > T
                        lnD  = nat_log(b{factor}(:,:,V(tau-1,policy,factor))*state_posterior{factor}(:,tau-1,policy));% forward message
                        lnBs = nat_log(B_norm(b{factor}(:,:,V(tau,policy,factor))')*state_posterior{factor}(:,tau+1,policy));% backward message
                    end
                    % here we both combine the messages and perform a gradient
                    % descent on the posterior 
                    v_depolarization = v_depolarization + (.5*lnD + .5*lnBs + lnAo(:,tau) - v_depolarization)/TimeConst;
                    % variational free energy at each time point
                    Ft(tau,Ni,t,factor) = state_posterior{factor}(:,tau,policy)'*(.5*lnD + .5*lnBs + lnAo(:,tau) - nat_log(state_posterior{factor}(:,tau,policy)));
                    % update posterior by running v through a softmax 
                    state_posterior{factor}(:,tau,policy) = (exp(v_depolarization)/sum(exp(v_depolarization)));    
                    % store state_posterior (normalised firing rate) from each epoch of
                    % gradient descent for each tau
                    normalized_firing_rates{factor}(Ni,:,tau,t,policy) = state_posterior{factor}(:,tau,policy);                   
                    % store v (non-normalized log posterior or 'membrane potential') 
                    % from each epoch of gradient descent for each tau
                    prediction_error{factor}(Ni,:,tau,t,policy) = v_depolarization;
                    clear v
                end
            end
        end        
      % variational free energy for each policy (F)
      Fintermediate = sum(Ft,4); % sum over hidden state factors (Fintermediate is an intermediate F value)
      Fintermediate = squeeze(sum( Fintermediate,1)); % sum over tau and squeeze into 16x3 matrix
      % store variational free energy at last iteration of message passing
      F(policy,t) = Fintermediate(end);
      clear Fintermediate
    end 
    
    % expected free energy (G) under each policy
    %----------------------------------------------------------------------
    
    % initialize intermediate expected free energy variable (Gintermediate) for each policy
    Gintermediate = zeros(NumPolicies,1);  
    % policy horizon for 'counterfactual rollout' for deep policies (described below)
    horizon = T;

    % loop over policies
    for policy = 1:NumPolicies
        
        % Bayesian surprise about 'd'
        if isfield(MDP,'d')
            for factor = 1:NumFactors
                Gintermediate(policy) = Gintermediate(policy) - d_complexity{factor}'*state_posterior{factor}(:,1,policy);
            end 
        end
         
        % This calculates the expected free energy from time t to the
        % policy horizon which, for deep policies, is the end of the trial T.
        % We can think about this in terms of a 'counterfactual rollout'
        % that asks, "what policy will best resolve uncertainty about the 
        % mapping between hidden states and observations (maximize
        % epistemic value) and bring about preferred outcomes"?
   
        for timestep = t:horizon
            % grab expected states for each policy and time
            for factor = 1:NumFactors
                Expected_states{factor} = state_posterior{factor}(:,timestep,policy);
            end 
            
            % calculate epistemic value term (Bayesian Surprise) and add to
            % expected free energy
            Gintermediate(policy) = Gintermediate(policy) + G_epistemic_value(a(:),Expected_states(:));
            
            for modality = 1:NumModalities
                % prior preferences about outcomes
                predictive_observations_posterior = cell_md_dot(a{modality},Expected_states(:)); %posterior over observations
                Gintermediate(policy) = Gintermediate(policy) + predictive_observations_posterior'*(C{modality}(:,timestep));

                % Bayesian surprise about parameters 
                if isfield(MDP,'a')
                    Gintermediate(policy) = Gintermediate(policy) - cell_md_dot(a_complexity{modality},{predictive_observations_posterior Expected_states{:}});
                end
            end 
        end 
    end 
    
    % store expected free energy for each time point and clear intermediate
    % variable
    G(:,t) = Gintermediate;
    clear Gintermediate
    
    % infer policy, update precision and calculate BMA over policies
    %----------------------------------------------------------------------
    

    % loop over policy selection using variational updates to gamma to
    % estimate the optimal contribution of expeceted free energy to policy
    % selection. This has the effect of down-weighting the contribution of 
    % variational free energy to the posterior over policies when the 
    % difference between the prior and posterior over policies is large
    
    if t > 1
        gamma(t) = gamma(t - 1);
    end
    for ni = 1:Ni 
        % posterior and prior over policies
        policy_priors(:,t) = exp(log(E) + gamma(t)*G(:,t))/sum(exp(log(E) + gamma(t)*G(:,t)));% prior over policies
        policy_posteriors(:,t) = exp(log(E) + gamma(t)*G(:,t) + F(:,t))/sum(exp(log(E) + gamma(t)*G(:,t) + F(:,t))); % posterior over policies
        
        % expected free energy precision (beta)
        G_error = (policy_posteriors(:,t) - policy_priors(:,t))'*G(:,t);
        beta_update = posterior_beta - beta + G_error; % free energy gradient w.r.t gamma
        posterior_beta = posterior_beta - beta_update/2; 
        gamma(t) = 1/posterior_beta;
        
        % simulate dopamine responses
        n = (t - 1)*Ni + ni;
        gamma_update(n,1) = gamma(t); % simulated neural encoding of precision (posterior_beta^-1)
                                      % at each iteration of variational updating
        policy_posterior_updates(:,n) = policy_posteriors(:,t); % neural encoding of policy posteriors
        policy_posterior(1:NumPolicies,t) = policy_posteriors(:,t); % record posterior over policies 
    end 
    
    % bayesian model average of hidden states (averaging over policies)
    for factor = 1:NumFactors
        for tau = 1:T
            % reshape state_posterior into a matrix of size NumStates(factor) x NumPolicies and then dot with policies
            BMA_states{factor}(:,tau) = reshape(state_posterior{factor}(:,tau,:),NumStates(factor),NumPolicies)*policy_posteriors(:,t);
        end
    end
    
    % action selection
    %----------------------------------------------------------------------
    
    % The probability of emitting each particular action is a softmax function 
    % of a vector containing the probability of each action summed over 
    % each policy. E.g. if there are three policies, a posterior over policies of 
    % [.4 .4 .2], and two possible actions, with policy 1 and 2 leading 
    % to action 1, and policy 3 leading to action 2, the probability of 
    % each action is [.8 .2]. This vector is then passed through a softmax function 
    % controlled by the inverse temperature parameter alpha which by default is extremely 
    % large (alpha = 512), leading to deterministic selection of the action with 
    % the highest probability. 
    
    if t < T

        % marginal posterior over action (for each factor)
        action_posterior_intermediate = zeros([NumControllable_transitions(end),1])';

        for policy = 1:NumPolicies % loop over number of policies
            sub = num2cell(V(t,policy,:));
            action_posterior_intermediate(sub{:}) = action_posterior_intermediate(sub{:}) + policy_posteriors(policy,t);
        end
        
        % action selection (softmax function of action potential)
        sub = repmat({':'},1,NumFactors);
        action_posterior_intermediate(:) = (exp(alpha*log(action_posterior_intermediate(:)))/sum(exp(alpha*log(action_posterior_intermediate(:))))); 
        action_posterior(sub{:},t) = action_posterior_intermediate;

        % next action - sampled from marginal posterior
        ControlIndex = find(NumControllable_transitions>1);
        action = (1:1:NumControllable_transitions(ControlIndex)); % 1:number of control states
        for factors = 1:NumFactors 
            if NumControllable_transitions(factors) > 2 % if there is more than one control state
                ind = find(rand < cumsum(action_posterior_intermediate(:)),1);  
                MDP.chosen_action(factor,t) = action(ind);
            end
        end

    end % end of state and action selection   
         
end % end loop over time points

% accumulate concentration paramaters (learning)
%--------------------------------------------------------------------------

for t = 1:T
    % a matrix (likelihood)
    if isfield(MDP,'a')
        for modality = 1:NumModalities
            a_learning = O(modality,t)';
            for  factor = 1:NumFactors
                a_learning = spm_cross(a_learning,BMA_states{factor}(:,t));
            end
            a_learning = a_learning.*(MDP.a{modality} > 0);
            MDP.a{modality} = MDP.a{modality}*omega + a_learning*eta;
        end
    end 
end 
 
% initial hidden states d (priors):
if isfield(MDP,'d')
    for factor = 1:NumFactors
        i = MDP.d{factor} > 0;
        MDP.d{factor}(i) = omega*MDP.d{factor}(i) + eta*BMA_states{factor}(i,1);
    end
end

% policies e (habits)
if isfield(MDP,'e')
    MDP.e = omega*MDP.e + eta*policy_posterior(:,T);
end

% Free energy of concentration parameters
%--------------------------------------------------------------------------

% Here we calculate the KL divergence (negative free energy) of the concentration 
% parameters of the learned distribution before and after learning has occured on 
% each trial. 

% (negative) free energy of a
for modality = 1:NumModalities
    if isfield(MDP,'a')
        MDP.Fa(modality) = - spm_KL_dir(MDP.a{modality},a_prior{modality});
    end
end

% (negative) free energy of d
for factor = 1:NumFactors
    if isfield(MDP,'d')
        MDP.Fd(factor) = - spm_KL_dir(MDP.d{factor},d_prior{factor});
    end
end

% (negative) free energy of e
if isfield(MDP,'e')
    MDP.Fe = - spm_KL_dir(MDP.e,E);
end

% simulated dopamine responses (beta updates)
%----------------------------------------------------------------------
% "deconvolution" of neural encoding of precision
if NumPolicies > 1
    phasic_dopamine = 8*gradient(gamma_update) + gamma_update/8;
else
    phasic_dopamine = [];
    gamma_update = [];
end

% Bayesian model average of neuronal variables; normalized firing rate and
% prediction error
%----------------------------------------------------------------------
for factor = 1:NumFactors
    BMA_normalized_firing_rates{factor} = zeros(Ni,NumStates(factor),T,T);
    BMA_prediction_error{factor} = zeros(Ni,NumStates(factor),T,T);
    for t = 1:T
        for policy = 1:NumPolicies 
            %normalised firing rate
            BMA_normalized_firing_rates{factor}(:,:,1:T,t) = BMA_normalized_firing_rates{factor}(:,:,1:T,t) + normalized_firing_rates{factor}(:,:,1:T,t,policy)*policy_posterior(policy,t);
            %depolarisation
            BMA_prediction_error{factor}(:,:,1:T,t) = BMA_prediction_error{factor}(:,:,1:T,t) + prediction_error{factor}(:,:,1:T,t,policy)*policy_posterior(policy,t);
        end
    end
end

% store variables in MDP structure
%----------------------------------------------------------------------

MDP.T  = T;                                   % number of belief updates
MDP.O  = O;                                   % outcomes
MDP.P  = action_posterior;                    % probability of action at time 1,...,T - 1
MDP.R  = policy_posterior;                    % Posterior over policies
MDP.Q  = state_posterior(:);                  % conditional expectations over N states
MDP.X  = BMA_states(:);                       % Bayesian model averages over T outcomes
MDP.C  = C(:);                                % preferences
MDP.G  = G;                                   % expected free energy
MDP.F  = F;                                   % variational free energy

MDP.s = true_states;                          % states
MDP.o = outcomes;                             % outcomes
MDP.u = MDP.chosen_action;                    % actions

MDP.w  = gamma;                               % posterior expectations of expected free energy precision (gamma)
MDP.vn = BMA_prediction_error(:);             % simulated neuronal prediction error
MDP.xn = BMA_normalized_firing_rates(:);      % simulated neuronal encoding of hidden states
MDP.un = policy_posterior_updates;            % simulated neuronal encoding of policies
MDP.wn = gamma_update;                        % simulated neuronal encoding of policy precision (beta)
MDP.dn = phasic_dopamine;                     % simulated dopamine responses (deconvolved)

%% Plot
%==========================================================================

% trial behaviour
spm_figure('GetWin','Figure 1'); clf    % display behavior
spm_MDP_VB_trial(MDP); 

% neuronal responces
spm_figure('GetWin','Figure 2'); clf    % display behavior
spm_MDP_VB_LFP(MDP,[],1); 

%% Functions
%==========================================================================

% normalise vector columns
function b = col_norm(B)
numfactors = numel(B);
for f = 1:numfactors
    bb{f} = B{f}; 
    z = sum(bb{f},1); %create normalizing constant from sum of columns
    bb{f} = bb{f}./z; %divide columns by constant
end 
b = bb;
end 

% norm the elements of B transpose as required by MMP
function b = B_norm(B)
bb = B; 
z = sum(bb,1); %create normalizing constant from sum of columns
bb = bb./z; % divide columns by constant
bb(isnan(bb)) = 0; %replace NaN with zero
b = bb;
% insert zero value condition
end 

% natural log that replaces zero values with very small values for numerical reasons.
function y = nat_log(x)
y = log(x+exp(-16));
end 

% dot product along dimension f
function B = md_dot(A,s,f)
if f == 1
    B = A'*s;
elseif f == 2
    B = A*s;
end 
end


%--- SPM functions
%==========================================================================

% These functions have been replicated (with permission) from the spm
% toolbox. To aid in understading, some variable names have been changed.

function X = cell_md_dot(X,x)
% initialize dimensions
DIM = (1:numel(x)) + ndims(X) - numel(x);

% compute dot product using recursive sums (and bsxfun)
for d = 1:numel(x)
    s         = ones(1,ndims(X));
    s(DIM(d)) = numel(x{d});
    X         = bsxfun(@times,X,reshape(full(x{d}),s));
    X         = sum(X,DIM(d));
end

% eliminate singleton dimensions
X = squeeze(X);
end 

% epistemic value term (Bayesian surprise) in expected free energy 
function G = G_epistemic_value(A,s)
    
% auxiliary function for Bayesian suprise or mutual information
% FORMAT [G] = spm_MDP_G(A,s)
%
% A   - likelihood array (probability of outcomes given causes)
% s   - probability density of causes

% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_G.m 7306 2018-05-07 13:42:02Z karl $

% probability distribution over the hidden causes: i.e., Q(s)

qx = spm_cross(s); % this is the outer product of the posterior over states
                   % calculated with respect to itself

% accumulate expectation of entropy: i.e., E[lnP(o|s)]
G     = 0;
qo    = 0;
for i = find(qx > exp(-16))'
    % probability over outcomes for this combination of causes
    po   = 1;
    for g = 1:numel(A)
        po = spm_cross(po,A{g}(:,i));
    end
    po = po(:);
    qo = qo + qx(i)*po;
    G  = G  + qx(i)*po'*nat_log(po);
end

% subtract entropy of expectations: i.e., E[lnQ(o)]
G  = G - qo'*nat_log(qo);
    
end 

%--------------------------------------------------------------------------
function A  = spm_wnorm(A)
% This uses the bsxfun function to subtract the inverse of each column
% entry from the inverse of the sum of the columns and then divide by 2.
% 
A   = A + exp(-16);
A   = bsxfun(@minus,1./sum(A,1),1./A)/2;
end 

function sub = spm_ind2sub(siz,ndx)
% subscripts from linear index
% 

n = numel(siz);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1
    vi       = rem(ndx - 1,k(i)) + 1;
    vj       = (ndx - vi)/k(i) + 1;
    sub(i,1) = vj;
    ndx      = vi;
end
end 

%--------------------------------------------------------------------------
function [Y] = spm_cross(X,x,varargin)
% Multidimensional outer product
% FORMAT [Y] = spm_cross(X,x)
% FORMAT [Y] = spm_cross(X)
%
% X  - numeric array
% x  - numeric array
%
% Y  - outer product
%
% See also: spm_dot
% Copyright (C) 2015 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_cross.m 7527 2019-02-06 19:12:56Z karl $

% handle single inputs
if nargin < 2
    if isnumeric(X)
        Y = X;
    else
        Y = spm_cross(X{:});
    end
    return
end

% handle cell arrays

if iscell(X), X = spm_cross(X{:}); end
if iscell(x), x = spm_cross(x{:}); end

% outer product of first pair of arguments (using bsxfun)
A = reshape(full(X),[size(X) ones(1,ndims(x))]);
B = reshape(full(x),[ones(1,ndims(X)) size(x)]);
Y = squeeze(bsxfun(@times,A,B));

% and handle remaining arguments
for i = 1:numel(varargin)
    Y = spm_cross(Y,varargin{i});
end
end 

%--------------------------------------------------------------------------
function [d] = spm_KL_dir(q,p)
% KL divergence between two Dirichlet distributions
% FORMAT [d] = spm_kl_dirichlet(lambda_q,lambda_p)
%
% Calculate KL(Q||P) = <log Q/P> where avg is wrt Q between two Dirichlet 
% distributions Q and P
%
% lambda_q   -   concentration parameter matrix of Q
% lambda_p   -   concentration parameter matrix of P
%
% This routine uses an efficient computation that handles arrays, matrices 
% or vectors. It returns the sum of divergences over columns.
%
% see also: spm_kl_dirichlet.m (for rwo vectors)
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% Will Penny 
% $Id: spm_KL_dir.m 7382 2018-07-25 13:58:04Z karl $

%  KL divergence based on log beta functions
d = spm_betaln(p) - spm_betaln(q) - sum((p - q).*spm_psi(q + 1/32),1);
d = sum(d(:));

return

% check on KL of Dirichlet ditributions
p  = rand(6,1) + 1;
q  = rand(6,1) + p;
p0 = sum(p);
q0 = sum(q);

d  = q - p;
KL = spm_betaln(p) - spm_betaln(q) + d'*spm_psi(q)
kl = gammaln(q0) - sum(gammaln(q)) - gammaln(p0) + sum(gammaln(p)) + ...
    d'*(spm_psi(q) - spm_psi(q0))
end 

%--------------------------------------------------------------------------
function y = spm_betaln(z)
% returns the log the multivariate beta function of a vector.
% FORMAT y = spm_betaln(z)
%   y = spm_betaln(z) computes the natural logarithm of the beta function
%   for corresponding elements of the vector z. if concerned is an array,
%   the beta functions are taken over the elements of the first to mention
%   (and size(y,1) equals one).
%
%   See also BETAINC, BETA.
%   Ref: Abramowitz & Stegun, Handbook of Mathematical Functions, sec. 6.2.
%   Copyright 1984-2004 The MathWorks, Inc. 

% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_betaln.m 7508 2018-12-21 09:49:44Z thomas $

% log the multivariate beta function of a vector
if isvector(z)
    z     = z(find(z)); %#ok<FNDSB>
    y     = sum(gammaln(z)) - gammaln(sum(z));
else
    for i = 1:size(z,2)
        for j = 1:size(z,3)
            for k = 1:size(z,4)
                for l = 1:size(z,5)
                    for m = 1:size(z,6)
                        y(1,i,j,k,l,m) = spm_betaln(z(:,i,j,k,l,m));
                    end
                end
            end
        end
    end
end
end 

%--------------------------------------------------------------------------
function [A] = spm_psi(A)
% normalisation of a probability transition rate matrix (columns)
% FORMAT [A] = spm_psi(A)
%
% A  - numeric array
%
% See also: psi.m
% Copyright (C) 2015 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_psi.m 7300 2018-04-25 21:14:07Z karl $

% normalization of a probability transition rate matrix (columns)
A = bsxfun(@minus, psi(A), psi(sum(A,1)));
end 

%% Set up POMDP model structure

% Please note that the main tutorial script ('Step_by_Step_AI_Guide.m') has
% more thorough descriptions of how to specify this generative model and
% the other parameters that might be included. Below we only describe the
% elements used to specify this specific model. Also, unlike the main
% tutorial script which focuses on learning initial state priors (d), 
% this version also enables habits (priors over policies; e) and separation of  
% the generative process from the generative model for the likelihood function (a).

function MDP = explore_exploit_model(Gen_model)

% Number of time points or 'epochs' within a trial: T
% =========================================================================

% Here, we specify 3 time points (T), in which the agent 1) starts in a 'Start'
% state, 2) first moves to either a 'Hint' state or a 'Choose Left' or 'Choose
% Right' slot machine state, and 3) either moves from the Hint state to one
% of the choice states or moves from one of the choice states back to the
% Start state.

T = 3;

% Priors about initial states: D and d
% =========================================================================

%--------------------------------------------------------------------------
% Specify prior probabilities about initial states in the generative 
% process (D)
% Note: By default, these will also be the priors for the generative model
%--------------------------------------------------------------------------

% For the 'context' state factor, we can specify that the 'left better' context 
% (i.e., where the left slot machine is more likely to win) is the true context:

D{1} = [1 0]';  % {'left better','right better'}

% For the 'behavior' state factor, we can specify that the agent always
% begins a trial in the 'start' state (i.e., before choosing to either pick
% a slot machine or first ask for a hint:

D{2} = [1 0 0 0]'; % {'start','hint','choose-left','choose-right'}

%--------------------------------------------------------------------------
% Specify prior beliefs about initial states in the generative model (d)
% Note: This is optional, and will simulate learning priors over states 
% if specified.
%--------------------------------------------------------------------------

% Note that these are technically what are called 'Dirichlet concentration
% paramaters', which need not take on values between 0 and 1. These values
% are added to after each trial, based on posterior beliefs about initial
% states. For example, if the agent believed at the end of trial 1 that it 
% was in the 'left better' context, then d{1} on trial 2 would be 
% d{1} = [1.5 0.5]' (although how large the increase in value is after 
% each trial depends on a learning rate). In general, higher values 
% indicate more confidence in one's beliefs about initial states, and 
% entail that beliefs will change more slowly (e.g., the shape of the 
% distribution encoded by d{1} = [25 25]' will change much more slowly 
% than the shape of the distribution encoded by d{1} = [.5 0.5]' with each 
% new observation).

% For context beliefs, we can specify that the agent starts out believing 
% that both contexts are equally likely, but with somewhat low confidence in 
% these beliefs:

d{1} = [.25 .25]';  % {'left better','right better'}

% For behavior beliefs, we can specify that the agent expects with 
% certainty that it will begin a trial in the 'start' state:

d{2} = [1 0 0 0]'; % {'start','hint','choose-left','choose-right'}


% State-outcome mappings and beliefs: A and a
% =========================================================================

%--------------------------------------------------------------------------
% Specify the probabilities of outcomes given each state in the generative 
% process (A)
% This includes one matrix per outcome modality
% Note: By default, these will also be the beliefs in the generative model
%--------------------------------------------------------------------------

% First we specify the mapping from states to observed hints (outcome
% modality 1). Here, the rows correspond to observations, the columns
% correspond to the first state factor (context), and the third dimension
% corresponds to behavior. Each column is a probability distribution
% that must sum to 1.

% We start by specifying that both contexts generate the 'No Hint'
% observation across all behavior states:

Ns = [length(D{1}) length(D{2})]; % number of states in each state factor (2 and 4)

for i = 1:Ns(2) 

    A{1}(:,:,i) = [1 1; % No Hint
                   0 0; % Machine-Left Hint
                   0 0];% Machine-Right Hint
end

% Then we specify that the 'Get Hint' behavior state generates a hint that
% either the left or right slot machine is better, depending on the context
% state. In this case, the hints are accurate with a probability of pHA. 

pHA = 1; % By default we set this to 1, but try changing its value to 
          % see how it affects model behavior

A{1}(:,:,2) = [0     0;      % No Hint
               pHA 1-pHA;    % Machine-Left Hint
               1-pHA pHA];   % Machine-Right Hint

% Next we specify the mapping between states and wins/losses. The first two
% behavior states ('Start' and 'Get Hint') do not generate either win or
% loss observations in either context:

for i = 1:2

    A{2}(:,:,i) = [1 1;  % Null
                   0 0;  % Loss
                   0 0]; % Win
end
           
% Choosing the left machine (behavior state 3) generates wins with
% probability pWin, which differs depending on the context state (columns):

pWin = .8; % By default we set this to 1, but try changing its value to 
          % see how it affects model behavior
           
A{2}(:,:,3) = [0      0;     % Null        
               1-pWin pWin;  % Loss
               pWin 1-pWin]; % Win

% Choosing the right machine (behavior state 4) generates wins with
% probability pWin, with the reverse mapping to context states from 
% choosing the left machine:
           
A{2}(:,:,4) = [0      0;     % Null
               pWin 1-pWin;  % Loss
               1-pWin pWin]; % Win
           
% Finally, we specify an identity mapping between behavior states and
% observed behaviors, to ensure the agent knows that behaviors were carried
% out as planned. Here, each row corresponds to each behavior state.
           
for i = 1:Ns(2) 

    A{3}(i,:,i) = [1 1];

end

%--------------------------------------------------------------------------
% Specify prior beliefs about state-outcome mappings in the generative model 
% (a)
% Note: This is optional, and will simulate learning state-outcome mappings 
% if specified.
%--------------------------------------------------------------------------
           
% Similar to learning priors over initial states, this simply
% requires specifying a matrix (a) with the same structure as the
% generative process (A), but with Dirichlet concentration parameters that
% can encode beliefs (and confidence in those beliefs) that need not
% match the generative process. Learning then corresponds to
% adding to the values of matrix entries, based on what outcomes were 
% observed when the agent believed it was in a particular state. For
% example, if the agent observed a win while believing it was in the 
% 'left better' context and the 'choose left machine' behavior state,
% the corresponding probability value would increase for that location in
% the state outcome-mapping (i.e., a{2}(3,1,3) might change from .8 to
% 1.8).

% One simple way to set up this matrix is by:
 
% 1. initially identifying it with the generative process 
% 2. multiplying the values by a large number to prevent learning all
%    aspects of the matrix (so the shape of the distribution changes very slowly)
% 3. adjusting the elements you want to differ from the generative process.

% To simulate learning the hint accuracy we
% can specify:

a{1} = A{1}*200;
a{2} = A{2}*200;
a{3} = A{3}*200;

a{1}(:,:,2) =  [0     0;     % No Hint
               .25   .25;    % Machine-Left Hint
               .25   .25];   % Machine-Right Hint
    

% Controlled transitions and transition beliefs : B{:,:,u} and b(:,:,u)
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions between hidden states
% under each action (sometimes called 'control states'). 
% Note: By default, these will also be the transitions beliefs 
% for the generative model
%--------------------------------------------------------------------------

% Columns are states at time t. Rows are states at t+1.

% The agent cannot control the context state, so there is only 1 'action',
% indicating that contexts remain stable within a trial:

B{1}(:,:,1) = [1 0;  % 'Left Better' Context
               0 1]; % 'Right Better' Context
           
% The agent can control the behavior state, and we include 4 possible 
% actions:

% Move to the Start state from any other state
B{2}(:,:,1) = [1 1 1 1;  % Start State
               0 0 0 0;  % Hint
               0 0 0 0;  % Choose Left Machine
               0 0 0 0]; % Choose Right Machine
           
% Move to the Hint state from any other state
B{2}(:,:,2) = [0 0 0 0;  % Start State
               1 1 1 1;  % Hint
               0 0 0 0;  % Choose Left Machine
               0 0 0 0]; % Choose Right Machine

% Move to the Choose Left state from any other state
B{2}(:,:,3) = [0 0 0 0;  % Start State
               0 0 0 0;  % Hint
               1 1 1 1;  % Choose Left Machine
               0 0 0 0]; % Choose Right Machine

% Move to the Choose Right state from any other state
B{2}(:,:,4) = [0 0 0 0;  % Start State
               0 0 0 0;  % Hint
               0 0 0 0;  % Choose Left Machine
               1 1 1 1]; % Choose Right Machine        
           
%--------------------------------------------------------------------------
% Specify prior beliefs about state transitions in the generative model
% (b). This is a set of matrices with the same structure as B.
% Note: This is optional, and will simulate learning state transitions if 
% specified.
%--------------------------------------------------------------------------
          
% For this example, we will not simulate learning transition beliefs. 
% But, similar to learning d and a, this just involves accumulating
% Dirichlet concentration parameters. Here, transition beliefs are updated
% after each trial when the agent believes it was in a given state at time
% t and and another state at t+1.

% Preferred outcomes: C and c
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the 'prior preferences', encoded here as log
% probabilities. 
%--------------------------------------------------------------------------

% One matrix per outcome modality. Each row is an observation, and each
% columns is a time point. Negative values indicate lower preference,
% positive values indicate a high preference. Stronger preferences promote
% risky choices and reduced information-seeking.

% We can start by setting a 0 preference for all outcomes:

No = [size(A{1},1) size(A{2},1) size(A{3},1)]; % number of outcomes in 
                                               % each outcome modality

C{1}      = zeros(No(1),T); % Hints
C{2}      = zeros(No(2),T); % Wins/Losses
C{3}      = zeros(No(3),T); % Observed Behaviors

% Then we can specify a 'loss aversion' magnitude (la) at time points 2 
% and 3, and a 'reward seeking' (or 'risk-seeking') magnitude (rs). Here,
% rs is divided by 2 at the third time point to encode a smaller win ($2
% instead of $4) if taking the hint before choosing a slot machine.

la = 1; % By default we set this to 1, but try changing its value to 
        % see how it affects model behavior

rs = 4; % By default we set this to 4, but try changing its value to 
        % see how it affects model behavior

C{2}(:,:) =    [0  0   0   ;  % Null
                0 -la -la  ;  % Loss
                0  rs  rs/2]; % win
            
%--------------------------------------------------------------------------
% One can also optionally choose to simulate preference learning by
% specifying a Dirichlet distribution over preferences (c). 
%--------------------------------------------------------------------------

% This will not be simulated here. However, this works by increasing the
% preference magnitude for an outcome each time that outcome is observed.
% The assumption here is that preferences naturally increase for entering
% situations that are more familiar.

% Allowable policies: U or V. 
%==========================================================================

%--------------------------------------------------------------------------
% Each policy is a sequence of actions over time that the agent can 
% consider. 
%--------------------------------------------------------------------------

% For our simulations, we will specify V, where rows correspond to time 
% points and should be length T-1 (here, 2 transitions, from time point 1
% to time point 2, and time point 2 to time point 3):

NumPolicies = 5; % Number of policies
NumFactors = 2; % Number of state factors

V         = ones(T-1,NumPolicies,NumFactors);

V(:,:,1) = [1 1 1 1 1;
            1 1 1 1 1]; % Context state is not controllable

V(:,:,2) = [1 2 2 3 4;
            1 3 4 1 1];
        
% For V(:,:,2), columns left to right indicate policies allowing: 
% 1. staying in the start state 
% 2. taking the hint then choosing the left machine
% 3. taking the hint then choosing the right machine
% 4. choosing the left machine right away (then returning to start state)
% 5. choosing the right machine right away (then returning to start state)


% Habits: E and e. 
%==========================================================================

%--------------------------------------------------------------------------
% Optional: a columns vector with one entry per policy, indicating the 
% prior probability of choosing that policy (i.e., independent of other 
% beliefs). 
%--------------------------------------------------------------------------

% We will not equip our agent with habits with any starting habits 
% (flat distribution over policies):

E = [1 1 1 1 1]';

% To incorporate habit learning, where policies become more likely after 
% each time they are chosen, we can also specify concentration parameters
% by specifying e:

 e = [1 1 1 1 1]';

% Additional optional parameters. 
%==========================================================================

% Eta: learning rate (0-1) controlling the magnitude of concentration parameter
% updates after each trial (if learning is enabled).

     eta = 1; % Default (maximum) learning rate
     
% Omega: forgetting rate (0-1) controlling the magnitude of reduction in concentration
% parameter values after each trial (if learning is enabled).

     omega = 1; % Default value indicating there is no forgetting (values < 1 indicate forgetting)

% Beta: Expected precision of expected free energy (G) over policies (a 
% positive value, with higher values indicating lower expected precision).
% Lower values increase the influence of habits (E) and otherwise make
% policy selection less deteriministic.

     beta = 1; % By default this is set to 1, but try increasing its value 
               % to lower precision and see how it affects model behavior

% Alpha: An 'inverse temperature' or 'action precision' parameter that 
% controls how much randomness there is when selecting actions (e.g., how 
% often the agent might choose not to take the hint, even if the model 
% assigned the highest probability to that action. This is a positive 
% number, where higher values indicate less randomness. Here we set this to 
% a fairly high value:

    alpha = 32; % fairly low randomness in action selection

%% Define POMDP Structure
%==========================================================================

mdp.T = T;                    % Number of time steps
mdp.V = V;                    % allowable (deep) policies

mdp.A = A;                    % state-outcome mapping
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % priors over initial states
mdp.d = d;                    % enable learning priors over initial states

if Gen_model == 1
    mdp.E = E;                % prior over policies
elseif Gen_model == 2
    mdp.a = a;                % enable learning state-outcome mappings
    mdp.e = e;                % enable learning of prior over policies
end 

mdp.eta = eta;                % learning rate
mdp.omega = omega;            % forgetting rate
mdp.alpha = alpha;            % action precision
mdp.beta = beta;              % expected free energy precision

%respecify for use in inversion script (specific to this tutorial example)
mdp.NumPolicies = NumPolicies; % Number of policies
mdp.NumFactors = NumFactors; % Number of state factors
    
   
% We can add labels to states, outcomes, and actions for subsequent plotting:

label.factor{1}   = 'contexts';   label.name{1}    = {'left-better','right-better'};
label.factor{2}   = 'choice states';     label.name{2}    = {'start','hint','choose left','choose right'};
label.modality{1} = 'hint';    label.outcome{1} = {'null','left hint','right hint'};
label.modality{2} = 'win/lose';  label.outcome{2} = {'null','lose','win'};
label.modality{3} = 'observed action';  label.outcome{3} = {'start','hint','choose left','choose right'};
label.action{2} = {'start','hint','left','right'};
mdp.label = label;

MDP = mdp;

end
