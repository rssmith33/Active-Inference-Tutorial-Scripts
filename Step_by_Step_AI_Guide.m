%% Step by step introduction to building and using active inference models

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% First, you need to add SPM12, the DEM toolbox of SPM12 and the
% folder with the example scripts to your path in Matlab.

clear all
close all      % These commands clear the workspace and close any figures

rng('shuffle') % This sets the random number generator to produce a different 
               % random sequence each time, which leads to variability in 
               % repeated simulation results (you can alse set to 'default'
               % to produce the same random sequence each time)

% Simulation options after model building below:

% If Sim = 1, simulate single trial. This will reproduce fig. 8. (Although
            % note that, for this and the following simulations, results 
            % will vary each time due to random sampling)

% If Sim = 2, simulate multiple trials where the left context is active 
            % (D{1} = [1 0]'). This will reproduce fig. 10.
             
% If Sim = 3, simulate reversal learning, where the left context is active 
            % (D{1} = [1 0]') in early trials and then reverses in later 
            % trials (D{1} = [0 1]'). This will reproduce fig. 11.
            
% If Sim = 4, run parameter estimation on simulated data with reversal
            % learning. This will reproduce the top panel of fig. 17.
            
% If Sim = 5, run parameter estimation on simulated data with reversal
            % learning from multiple participants under different models
            % (i.e., different parameter values) and perform model comparison. 
            % This will reproduce the bottom panel of fig. 17. This option
            % will also save two structures that include results of model
            % comparison, model fitting, parameter recoverability analyses,
            % and inputs needed for group (PEB) analyses.
            
rs1 = 4; % Risk-seeking parameter (set to the variable rs below) 
         % To reproduce fig. 8, use values of 4 or 8 (with Sim = 1)
         % To reproduce fig. 10, use values of 3 or 4 (with Sim = 2)
         % To reproduce fig. 11, use values of 3 or 4 (with Sim = 3)
         % This will have no effect on Sim = 4 or Sim = 5

Sim = 1;

% When Sim = 5, if PEB = 1 the script will run simulated group-level
% (Parametric Empirical Bayes) analyses.

PEB = 0; % Note: GCM_2 and GCM_3 (the inputs to PEB; see below) are saved 
         % after running Sim = 5 to avoid needing to re-run it each time 
         % you want to use PEB (i.e., because Sim = 5 takes a long time). 
         % After running Sim = 5 once, you can simply load GCM_2 and GCM_3 and 
         % run the PEB section separately if you want to come back to it later.

% You can also run the sections separately after building the model by
% simply clicking into that section and clicking 'Run Section' above

%% 1. Set up model structure

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

pWin = .8; % By default we set this to .8, but try changing its value to 
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
           
% We will not simulate, learning the 'a' matrix here.  
% However, similar to learning priors over initial states, this simply
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

% For example, to simulate learning the reward probabilities, we could specify:
    
    % a{1} = A{1}*200;
    % a{2} = A{2}*200;
    % a{3} = A{3}*200;
    % 
    % a{2}(:,:,3) =  [0  0;  % Null        
    %                .5 .5;  % Loss
    %                .5 .5]; % Win
    % 
    % 
    % a{2}(:,:,4) =  [0  0;  % Null        
    %                .5 .5;  % Loss
    %                .5 .5]; % Win

% As another example, to simulate learning the hint accuracy one
% might specify:

    % a{1} = A{1}*200;
    % a{2} = A{2}*200;
    % a{3} = A{3}*200;
     
    % a{1}(:,:,2) =  [0     0;     % No Hint
    %                .25   .25;    % Machine-Left Hint
    %                .25   .25];   % Machine-Right Hint
    

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

rs = rs1; % We set this value at the top of the script. 
          % By default we set it to 4, but try changing its value to 
          % see how it affects model behavior (higher values will promote
          % risk-seeking, as described in the main text)

C{2}(:,:) =    [0  0   0   ;  % Null
                0 -la -la  ;  % Loss
                0  rs  rs/2]; % win
            
% Note that, expanded out, this means that the other C-matrices will be:

% C{1} =      [0 0 0;     % No Hint
%              0 0 0;    % Machine-Left Hint
%              0 0 0];   % Machine-Right Hint
% 
% C{3} =      [0 0 0;  % Start State
%              0 0 0;  % Hint
%              0 0 0;  % Choose Left Machine
%              0 0 0]; % Choose Right Machine

            
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

% Policies can be specified as 'shallow' (looking only one step
% ahead), as specified by U. Or policies can be specified as 'deep' 
% (planning actions all the way to the end of the trial), as specified by
% V. Both U and V must be specified for each state factor as the third
% matrix dimension. This will simply be all 1s if that state is not
% controllable.

% For example, specifying U could simply be:

    % Np = 4; % Number of policies
    % Nf = 2; % Number of state factors
    % 
    % U         = ones(1,Np,Nf);
    % 
    % U(:,:,1) = [1 1 1 1]; % Context state is not controllable
    % U(:,:,2) = [1 2 3 4]; % All four actions in B{2} are allowed

% For our simulations, we will specify V, where rows correspond to time 
% points and should be length T-1 (here, 2 transitions, from time point 1
% to time point 2, and time point 2 to time point 3):

Np = 5; % Number of policies
Nf = 2; % Number of state factors

V         = ones(T-1,Np,Nf);

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

% We will not equip our agent with habits in our example simulations, 
% but this could be specified as a follows if one wanted to include a
% strong habit to choose the 4th policy:

% E = [.1 .1 .1 .6 .1]';

% To incorporate habit learning, where policies become more likely after 
% each time they are chosen, one can also specify concentration parameters
% by specifying e. For example:

% e = [1 1 1 1 1]';

% Additional optional parameters. 
%==========================================================================

% Eta: learning rate (0-1) controlling the magnitude of concentration parameter
% updates after each trial (if learning is enabled).

    eta = 0.5; % By default we here set this to 0.5, but try changing its value  
               % to see how it affects model behavior

% Omega: forgetting rate (0-1) controlling the reduction in concentration parameter
% magnitudes after each trial (if learning is enabled). This controls the
% degree to which newer experience can 'over-write' what has been learned
% from older experiences. It is adaptive in environments where the true
% parameters in the generative process (priors, likelihoods, etc.) can
% change over time. A low value for omega can be seen as a prior that the
% world is volatile and that contingencies change over time.

    omega = 1; % By default we here set this to 1 (indicating no forgetting, 
               % but try changing its value to see how it affects model behavior. 
               % Values below 1 indicate greater rates of forgetting.
               
% Beta: Expected precision of expected free energy (G) over policies (a 
% positive value, with higher values indicating lower expected precision).
% Lower values increase the influence of habits (E) and otherwise make
% policy selection less deteriministic. For our example simulations we will
% simply set this to its default value of 1:

     beta = 1; % By default this is set to 1, but try increasing its value 
               % to lower precision and see how it affects model behavior

% Alpha: An 'inverse temperature' or 'action precision' parameter that 
% controls how much randomness there is when selecting actions (e.g., how 
% often the agent might choose not to take the hint, even if the model 
% assigned the highest probability to that action. This is a positive 
% number, where higher values indicate less randomness. Here we set this to 
% a high value:

    alpha = 32;  % Any positive number. 1 is very low, 32 is fairly high; 
                 % an extremely high value can be used to specify
                 % deterministic action (e.g., 512)

% ERP: This parameter controls the degree of belief resetting at each 
% time point in a trial when simulating neural responses. A value of 1
% indicates no resetting, in which priors smoothly carry over. Higher
% values indicate degree of loss in prior confidence at each time step.

    erp = 1; % By default we here set this to 1, but try increasing its value  
             % to see how it affects simulated neural (and behavioral) responses
                          
% tau: Time constant for evidence accumulation. This parameter controls the
% magnitude of updates at each iteration of gradient descent. Larger values 
% of tau will lead to smaller updates and slower convergence time, 
% but will also promote greater stability in posterior beliefs. 

    tau = 12; % Here we set this to 12 to simulate smooth physiological responses,   
              % but try adjusting its value to see how it affects simulated
              % neural (and behavioral) responses
              
% Note: If these values are left unspecified, they are assigned default
% values when running simulations. These default values can be found within
% the spm_MDP_VB_X script (and in the spm_MDP_VB_X_tutorial script we
% provide in this tutorial).

% Other optional constants. 
%==========================================================================

% Chi: Occam's window parameter for the update threshold in deep temporal 
% models. In hierarchical models, this parameter controls how quickly
% convergence is 'cut off' during lower-level evidence accumulation. 
% specifically, it sets an uncertainty threshold, below which no additional 
% trial epochs are simulated. By default, this is set to 1/64. Smaller 
% numbers (e.g., 1/128) indicate lower uncertainty (greater confidence) is
% required before which the number of trial epochs are shortened.

% zeta: Occam's window for policies. This parameter controls the threshold
% at which a policy ceases to be considered if its free energy
% becomes too high (i.e., when it becomes too implausible to consider
% further relative to other policies). It is set to default at a value of 
% 3. Higher values indicate a higher threshold. For example, a value of 6
% would indicate that a greater difference between a given policy and the
% best policy before that policy was 'pruned' (i.e., ceased to be
% considered). Policies will therefore be removed more quickly with smaller
% zeta values.
         
% Note: The spm_MDP_VB_X function is also equipped with broader functionality
% allowing incorporation of mixed (discrete and continuous) models,
% plotting, simulating Bayesian model reduction during simulated
% rest/sleep, among others. We do not describe these in detail here, but
% are described in the documentation at the top of the function.

% True states and outcomes: s and o. 
%==========================================================================

%--------------------------------------------------------------------------
% Optionally, one can also specify true states and outcomes for some or all
% time points with s and o. If not specified, these will be 
% generated by the generative process. 
%--------------------------------------------------------------------------

% For example, this means the true states at time point 1 are left context 
% and start state:

    %      s = [1;
    %           1]; % the later time points (rows for each state factor) are 0s,
    %               % indicating not specified.
      

% And this means the observations at time point 1 are the No Hint, Null,
% and Start behavior observations.

    %      o = [1;
    %           1;
    %           1]; % the later time points (rows for each outcome modality) are 
    %               % 0s, indicating not specified
 
%% 2. Define MDP Structure
%==========================================================================
%==========================================================================

mdp.T = T;                    % Number of time steps
mdp.V = V;                    % allowable (deep) policies

    %mdp.U = U;                   % We could have instead used shallow 
                                  % policies (specifying U instead of V).

mdp.A = A;                    % state-outcome mapping
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % priors over initial states

mdp.d = d;                    % enable learning priors over initial states
    
mdp.eta = eta;                % learning rate
mdp.omega = omega;            % forgetting rate
mdp.alpha = alpha;            % action precision
mdp.beta = beta;              % expected precision of expected free energy over policies
mdp.erp = erp;                % degree of belief resetting at each timestep
mdp.tau = tau;                % time constant for evidence accumulation

% Note, here we are not including habits:

    % mdp.E = E;

% or learning other parameters:
    % mdp.a = a;                    
    % mdp.b = b;
    % mdp.c = c;
    % mdp.e = e;         

% or specifying true states or outcomes:

    % mdp.s = s;
    % mdp.o = o;
    
% or specifying other optional parameters (described above):

    % mdp.chi = chi;    % confidence threshold for ceasing evidence
                        % accumulation in lower levels of hierarchical models
    % mdp.zeta = zeta;  % occams window for ceasing to consider implausible
                        % policies
      
% We can add labels to states, outcomes, and actions for subsequent plotting:

label.factor{1}   = 'contexts';   label.name{1}    = {'left-better','right-better'};
label.factor{2}   = 'choice states';     label.name{2}    = {'start','hint','choose left','choose right'};
label.modality{1} = 'hint';    label.outcome{1} = {'null','left hint','right hint'};
label.modality{2} = 'win/lose';  label.outcome{2} = {'null','lose','win'};
label.modality{3} = 'observed action';  label.outcome{3} = {'start','hint','choose left','choose right'};
label.action{2} = {'start','hint','left','right'};
mdp.label = label;

clear beta
clear alpha
clear eta
clear omega
clear la
clear rs % We clear these so we can re-specify them in later simulations

%--------------------------------------------------------------------------
% Use a script to check if all matrix-dimensions are correct:
%--------------------------------------------------------------------------
mdp = spm_MDP_check(mdp);


if Sim ==1
%% 3. Single trial simulations
 
%--------------------------------------------------------------------------
% Now that the generative process and model have been generated, we can
% simulate a single trial using the spm_MDP_VB_X script. Here, we provide 
% a version specific to this tutorial - spm_MDP_VB_X_tutorial - that adds 
% the learning rate (eta) for initial state priors (d), and adds forgetting rate (omega), 
% which are not included in the current SPM version (as of 05/08/21).
%--------------------------------------------------------------------------

MDP = spm_MDP_VB_X_tutorial(mdp);

% We can then use standard plotting routines to visualize simulated neural 
% responses

spm_figure('GetWin','Figure 1'); clf    % display behavior
spm_MDP_VB_LFP(MDP); 

%  and to show posterior beliefs and behavior:

spm_figure('GetWin','Figure 2'); clf    % display behavior
spm_MDP_VB_trial(MDP); 

% Please see the main text for figure interpretations

elseif Sim == 2
%% 4. Multi-trial simulations

% Next, we can expand the mdp structure to include multiple trials

N = 30; % number of trials

MDP = mdp;

[MDP(1:N)] = deal(MDP);

MDP = spm_MDP_VB_X_tutorial(MDP);

% We can again visualize simulated neural responses

spm_figure('GetWin','Figure 3'); clf    % display behavior
spm_MDP_VB_game_tutorial(MDP); 

elseif Sim == 3
%% 5. Simulate reversal learning

N = 32; % number of trials (must be multiple of 8)

MDP = mdp;

[MDP(1:N)] = deal(MDP);

    for i = 1:N/8
        MDP(i).D{1}   = [1 0]'; % Start in the 'left-better' context for 
                                % early trials
    end

    for i = (N/8)+1:N
        MDP(i).D{1}   = [0 1]'; % Switch to 'right-better' context for 
                                % the remainder of the trials
    end
    
MDP = spm_MDP_VB_X_tutorial(MDP);

% We can again visualize simulated neural responses

spm_figure('GetWin','Figure 4'); clf    % display behavior
spm_MDP_VB_game_tutorial(MDP); 

elseif Sim == 4
%% 6. Model inversion to recover parameters (action precision and risk-seeking)
%==========================================================================
%==========================================================================

close all

% Generate simulated behavior under specific parameter values:
%==========================================================================

% We will again use the reversal learning version

N = 32; % number of trials

MDP = mdp;

[MDP(1:N)] = deal(MDP);

    for i = 1:N/8
        MDP(i).D{1}   = [1 0]'; % Start in the 'left-better' context for 
                                % early trials
    end

    for i = (N/8)+1:N
        MDP(i).D{1}   = [0 1]'; % Switch to 'right-better' context for 
                                % the remainder of the trials
    end
    
%==========================================================================
% true parameter values (to try to recover during estimation):
%==========================================================================

alpha = 4; % specify a lower action precision (4) than the prior value (16)
la = 1;    % keep loss aversion at a value of 1
rs = 6;    % specify higher risk-seeking (6) than the prior value (5)

C_fit = [0  0   0 ;    % Null
         0 -la -la  ;  % Loss
         0  rs  rs/2]; % Win

[MDP(1:N).alpha] = deal(alpha); 

for i = 1:N
    MDP(i).C{2} = C_fit; 
end
                           
                            
% If you wanted, you could also adjust the true value for other
% parameters in the same manner. For example:

    % beta = 5; % specify a lower expected policy precision (5) than the prior value (1)
    % [MDP(1:N).beta] = deal(beta); 

    % eta = .9; % specify a higher learning rate (.9) than the prior value (.5)
    % [MDP(1:N).eta] = deal(eta); 
    
%==========================================================================

MDP = spm_MDP_VB_X_tutorial(MDP);


% Invert model and try to recover original parameters:
%==========================================================================

%--------------------------------------------------------------------------
% This is where we do model inversion. Model inversion is based on variational
% Bayes. Here we will maximize (negative) variational free energy with
% respect to the free parameters (here: alpha and rs). This corresponds to 
% maximising the likelihood of the data under these parameters (i.e., maximizing
% accuracy) and at the same time penalizing for strong deviations from the
% priors over the parameters (i.e., minimizing complexity), which prevents
% overfitting.
% 
% You can specify the prior mean and variance of each parameter at the
% beginning of the Estimate_parameters script.
%--------------------------------------------------------------------------
mdp.la_true = la;   % Carries over true la value for use in estimation script
mdp.rs_true = rs;   % Carries over true rs value for use in estimation script

DCM.MDP   = mdp;                  % MDP model that will be estimated
DCM.field = {'alpha','rs'};       % parameter (field) names to optimise

% Note: If you wanted to fit other parameters, you can simply add their
% field names, such as:

 % DCM.field = {'alpha','rs','eta'}; % if you wanted to fit learning rate
 
% This requires that those parameters are also included in the possible
% parameters specified in the Estimate_parameters script.

% Next we add the true observations and actions of a (simulated)
% participant

DCM.U     = {MDP.o};              % include the observations made by (real 
                                  % or simulated) participants
                                  
DCM.Y     = {MDP.u};              % include the actions made by (real or 
                                  % simulated) participants
 
DCM       = Estimate_parameters(DCM); % Run the parameter estimation function
 
subplot(2,2,3)
xticklabels(DCM.field),xlabel('Parameter')
subplot(2,2,4)
xticklabels(DCM.field),xlabel('Parameter')
 
% Check deviation of prior and posterior means & posterior covariance
%==========================================================================

%--------------------------------------------------------------------------
% re-transform values and compare prior with posterior estimates
%--------------------------------------------------------------------------

field = fieldnames(DCM.M.pE);
for i = 1:length(field)
    if strcmp(field{i},'eta')
        prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM.Ep.(field{i}))); 
    elseif strcmp(field{i},'omega')
        prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    else
        prior(i) = exp(DCM.M.pE.(field{i}));
        posterior(i) = exp(DCM.Ep.(field{i}));
    end
end

figure, set(gcf,'color','white')
subplot(2,1,1),hold on
title('Means')
bar(prior,'FaceColor',[.5,.5,.5]),bar(posterior,0.5,'k')
xlim([0,length(prior)+1]),set(gca, 'XTick', 1:length(prior)),set(gca, 'XTickLabel', DCM.field)
legend({'Prior','Posterior'})
hold off
subplot(2,1,2)
imagesc(DCM.Cp),caxis([0 1]),colorbar
title('(Co-)variance')
set(gca, 'XTick', 1:length(prior)),set(gca, 'XTickLabel', DCM.field)
set(gca, 'YTick', 1:length(prior)),set(gca, 'YTickLabel', DCM.field)
 
% To show evidence of recoverability, you may want to estimate parameters
% from simulated data generated by a range of parameters, and then check
% the strengt of the correlation between the true parameters and estimated
% parameters to make sure there is a reasonably strong relationship. We try
% this at the end of section 7.

elseif Sim == 5
%% 7. Model comparison
%==========================================================================
%==========================================================================
 
% Now we will simulate data for 6 participants and fit them to two models:
% One which only fits action precision (alpha) and risk-seeking (rs), and 
% another that also fits learning rate (eta).

% Create vectors/matrices that will store results

F_2_params = [];
F_3_params = [];

avg_LL_2_params = [];
avg_prob_2_params = [];
avg_LL_3_params = [];
avg_prob_3_params = [];

GCM_2 = {};
GCM_3 = {};

Sim_params_2 = [];
true_params_2 = [];
Sim_params_3 = [];
true_params_3 = [];

% Set up reversal learning trials like before

N = 32; % number of trials

MDP = mdp;

[MDP(1:N)] = deal(MDP);

    for i = 1:N/8
        MDP(i).D{1}   = [1 0]'; % Start in the 'left-better' context for 
                                % early trials
    end

    for i = (N/8)+1:N
        MDP(i).D{1}   = [0 1]'; % Switch to 'right-better' context for 
                                % the remainder of the trials
    end

% Generate free energies for model fits for 2 parameter model (without eta)

rs_sequence = [4 6];   % specify different true risk-seeking values (prior = 5)
alpha_sequence = [4 16 24]; % specify different true action precisions (prior = 16)
     

for rs = rs_sequence  % specify different true risk-seeking values (prior = 5)
    for alpha = alpha_sequence   % specify different true action precisions (prior = 16) 
        
        
MDP_temp = MDP;
        
la = 1;   % keep loss aversion at a value of 1

C_fit = [0  0   0 ;    % Null
         0 -la -la  ;  % Loss
         0  rs  rs/2]; % Win

[MDP_temp(1:N).alpha] = deal(alpha); 

for i = 1:N
    MDP_temp(i).C{2} = C_fit; 
end

mdp.la_true = la;   % Carries over true la value for use in estimation script
mdp.rs_true = rs;   % Carries over true rs value for use in estimation script

MDP_temp = spm_MDP_VB_X_tutorial(MDP_temp);

spm_figure('GetWin','Figure 5'); clf    % display behavior to fit
spm_MDP_VB_game_tutorial(MDP_temp); 

DCM.MDP   = mdp;                  % MDP model that will be estimated
DCM.field = {'alpha','rs'};       % parameter (field) names to optimise

DCM.U     = {MDP_temp.o};              % include the observations made by (real 
                                  % or simulated) participants
                                  
DCM.Y     = {MDP_temp.u};              % include the actions made by (real or 
                                  % simulated) participants
 
DCM       = Estimate_parameters(DCM); % Run the parameter estimation function

% Convert parameters back out of log- or logit-space

field = fieldnames(DCM.M.pE);
for i = 1:length(field)
    if strcmp(field{i},'eta')
        DCM.prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        DCM.posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    elseif strcmp(field{i},'omega')
        DCM.prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        DCM.posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    else
        DCM.prior(i) = exp(DCM.M.pE.(field{i}));
        DCM.posterior(i) = exp(DCM.Ep.(field{i}));
    end
end

F_2_params = [F_2_params DCM.F];% Get free energies for each participant's model

GCM_2   = [GCM_2;{DCM}]; % Save DCM for each participant

% Get Log-likelihood and action probabilities for best-fit model

MDP_best = MDP;

[MDP_best(1:N).alpha] = deal(DCM.posterior(1)); 

C_fit_best = [0  0   0 ;                                % Null
              0 -la -la  ;                              % Loss
              0  DCM.posterior(2)  DCM.posterior(2)/2]; % Win

for i = 1:N
    MDP_best(i).C{2} = C_fit_best; 
end

for i = 1:N
    MDP_best(i).o = MDP_temp(i).o; 
end

for i = 1:N
    MDP_best(i).u = MDP_temp(i).u; 
end

MDP_best   = spm_MDP_VB_X_tutorial(MDP_best); % run model with best parameter values

% Get sum of log-likelihoods for each action across trials

L     = 0; % start (log) probability of actions given the model at 0
total_prob = 0;

for i = 1:numel(MDP_best) % Get probability of true actions for each trial
    for j = 1:numel(MDP_best(1).u(2,:)) % Only get probability of the second (controllable) state factor
        
        L = L + log(MDP_best(i).P(:,MDP_best(i).u(2,j),j)+ eps); % sum the (log) probabilities of each action
                                                                 % given a set of possible parameter values
        total_prob = total_prob + MDP_best(i).P(:,MDP_best(i).u(2,j),j); % sum the (log) probabilities of each action
                                                                     % given a set of possible parameter values

    end
end 

% Get the average log-likelihood for each participant and average action
% probability of each participant under best-fit parameters

avg_LL_2 = L/(size(MDP_best,2)*2);

avg_LL_2_params = [avg_LL_2_params; avg_LL_2];

avg_prob_2 = total_prob/(size(MDP_best,2)*2);

avg_prob_2_params = [avg_prob_2_params; avg_prob_2];

% Store true and estimated parameters to assess recoverability

Sim_params_2 = [Sim_params_2; DCM.posterior];% Get posteriors
true_params_2 = [true_params_2; [alpha rs]];% Get true params

clear DCM
clear MDP_temp
clear MDP_best

    end
end

% Separately store true and simulated parameters

True_alpha_2 = true_params_2(:,1);
Estimated_alpha_2 = Sim_params_2(:,1);  
True_rs_2 = true_params_2(:,2);
Estimated_rs_2 = Sim_params_2(:,2); 

% Generate free energies for model fits for 3 parameter model (with eta)

for rs = rs_sequence  % specify different true risk-seeking values (prior = 2)
    for alpha = alpha_sequence   % specify different true action precisions (prior = 16) 
        
MDP_temp = MDP;
        
la = 1;   % keep loss aversion at a value of 1

if rs == rs_sequence(1,1)
    eta = .2; % set lower value of eta than the estimation prior (.5) for 3 participants
elseif rs == rs_sequence(1,2)
    eta = .8; % set higher value of eta than the estimation prior (.5) for 3 participants
end


C_fit = [0  0   0 ;    % Null
         0 -la -la  ;  % Loss
         0  rs  rs/2]; % Win

[MDP_temp(1:N).alpha] = deal(alpha);
[MDP_temp(1:N).eta] = deal(eta);

for i = 1:N
    MDP_temp(i).C{2} = C_fit; 
end

mdp.la_true = la;   % Carries over true la value for use in estimation script
mdp.rs_true = rs;   % Carries over true rs value for use in estimation script

MDP_temp = spm_MDP_VB_X_tutorial(MDP_temp);

spm_figure('GetWin','Figure 6'); clf    % display behavior to fit
spm_MDP_VB_game_tutorial(MDP_temp); 

DCM.MDP   = mdp;                  % MDP model that will be estimated
DCM.field = {'alpha','rs','eta'}; % parameter (field) names to optimise

DCM.U     = {MDP_temp.o};              % include the observations made by (real 
                                  % or simulated) participants
                                  
DCM.Y     = {MDP_temp.u};              % include the actions made by (real or 
                                  % simulated) participants
 
DCM       = Estimate_parameters(DCM); % Run the parameter estimation function

% Convert parameters back out of log- or logit-space

field = fieldnames(DCM.M.pE);
for i = 1:length(field)
    if strcmp(field{i},'eta')
        DCM.prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        DCM.posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    elseif strcmp(field{i},'omega')
        DCM.prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        DCM.posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    else
        DCM.prior(i) = exp(DCM.M.pE.(field{i}));
        DCM.posterior(i) = exp(DCM.Ep.(field{i}));
    end
end


F_3_params = [F_3_params DCM.F]; % Get free energies for each participant's model

GCM_3   = [GCM_3;{DCM}]; % Save DCM for each participant

% Get Log-likelihood and action probabilities for best-fit model

MDP_best = MDP;

[MDP_best(1:N).alpha] = deal(DCM.posterior(1)); 

C_fit_best = [0  0   0 ;                                % Null
              0 -la -la  ;                              % Loss
              0  DCM.posterior(2)  DCM.posterior(2)/2]; % Win

for i = 1:N
    MDP_best(i).C{2} = C_fit_best; 
end

if rs == rs_sequence(1,1)
    eta = .2; % set lower value of eta than the estimation prior (.5) for 3 participants
elseif rs == rs_sequence(1,2)
    eta = .8; % set higher value of eta than the estimation prior (.5) for 3 participants
end

[MDP_best(1:N).eta] = deal(eta);


for i = 1:N
    MDP_best(i).o = MDP_temp(i).o; 
end

for i = 1:N
    MDP_best(i).u = MDP_temp(i).u; 
end

MDP_best   = spm_MDP_VB_X_tutorial(MDP_best); % run model with best parameter values

% Get sum of log-likelihoods for each action across trials

L     = 0; % start (log) probability of actions given the model at 0
total_prob = 0;

for i = 1:numel(MDP_best) % Get probability of true actions for each trial
    for j = 1:numel(MDP_best(1).u(2,:)) % Only get probability of the second (controllable) state factor
        
        L = L + log(MDP_best(i).P(:,MDP_best(i).u(2,j),j)+ eps); % sum the (log) probabilities of each action
                                                                 % given a set of possible parameter values
        total_prob = total_prob + MDP_best(i).P(:,MDP_best(i).u(2,j),j); % sum the (log) probabilities of each action
                                                                     % given a set of possible parameter values

    end
end 

% Get the average log-likelihood for each participant and average action
% probability of each participant under best-fit parameters

avg_LL_3 = L/(size(MDP_best,2)*2);

avg_LL_3_params = [avg_LL_3_params; avg_LL_3];

avg_prob_3 = total_prob/(size(MDP_best,2)*2);

avg_prob_3_params = [avg_prob_3_params; avg_prob_3];

% Store true and estimated parameters to assess recoverability

Sim_params_3 = [Sim_params_3; DCM.posterior];% Get posteriors
true_params_3 = [true_params_3; [alpha rs eta]];% Get true params

clear DCM
clear MDP_temp
clear MDP_best

    end
end

% Separately store true and simulated parameters

True_alpha_3 = true_params_3(:,1);
Estimated_alpha_3 = Sim_params_3(:,1);  
True_rs_3 = true_params_3(:,2);
Estimated_rs_3 = Sim_params_3(:,2); 
True_eta_3 = true_params_3(:,3);
Estimated_eta_3 = Sim_params_3(:,3); 

clear alpha

% Random Effects Bayesian Model Comparison (of Free Energies of best-fit 
% models per participant):

F_2_params = F_2_params';
F_3_params = F_3_params'; % Convert free energies to column vectors

[alpha,exp_r,xp,pxp,bor] = spm_BMS([F_2_params F_3_params]);

disp(' ');
disp(' ');
disp('Protected exceedance probability (pxp):');
disp(pxp);
disp(' ');

% The pxp value is the protected exceedance probability (pxp), which will 
% provide a probability of each model being the best-fit model. For example, 
% pxp = [.37 .63] would indicate a higher probability of the 3-parameter model 

%--------------------------------------------------------------------------

% We can also calculate the average probability and log-likelihood (LL) of the 
% actions under the 2- and 3-parameter models:

average_LL_2p = mean(avg_LL_2_params);
average_action_probability_2p = mean(avg_prob_2_params);
average_LL_3p = mean(avg_LL_3_params);
average_action_probability_3p = mean(avg_prob_3_params);

disp(' ');
fprintf('Average log-likelihood under the 2-parameter model: %.2g\n',average_LL_2p);
fprintf('Average action probability under the 2-parameter model: %.2g\n',average_action_probability_2p);
disp(' ');
fprintf('Average log-likelihood under the 3-parameter model: %.2g\n',average_LL_2p);
fprintf('Average action probability under the 3-parameter model: %.2g\n',average_action_probability_2p);
disp(' ');

%% Brief continuation of section 6 on recoverability
%==========================================================================
% Here we can also compute the strength of the relationship between true
% and estimated parameters to check recoverability. 
%==========================================================================

% Assemble matrices for correlation (2-parameter model)
recover_check_alpha_2 = [True_alpha_2 Estimated_alpha_2];
recover_check_rs_2 = [True_rs_2 Estimated_rs_2];

% Get correlations and significance
[Correlations_alpha_2, Significance_alpha_2] = corrcoef(recover_check_alpha_2);
[Correlations_rs_2, Significance_rs_2] = corrcoef(recover_check_rs_2);

% In this case, the correlations appear quite high for rs, and moderate for
% alpha

disp(' ');
disp('2-parameter model:');
disp(' ');
fprintf('Alpha recoverability: r = %.2g\n',Correlations_alpha_2(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_alpha_2(1,2));
disp(' ');
fprintf('Risk-seeking recoverability: r = %.2g\n',Correlations_rs_2(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_rs_2(1,2));
disp(' ');

% Assemble matrices for correlation (3-parameter model)
recover_check_alpha_3 = [True_alpha_3 Estimated_alpha_3];
recover_check_rs_3 = [True_rs_3 Estimated_rs_3];
recover_check_eta_3 = [True_eta_3 Estimated_eta_3];

% Get correlations and significance
[Correlations_alpha_3, Significance_alpha_3] = corrcoef(recover_check_alpha_3);
[Correlations_rs_3, Significance_rs_3] = corrcoef(recover_check_rs_3);
[Correlations_eta_3, Significance_eta_3] = corrcoef(recover_check_eta_3);

% In this case, the correlations appear high for rs and alpha, and moderate for
% learning rate. Note, however, that a wider range of values should be simulated to
% confirm recoverability in actual studies (and with a larger number of subjects).

disp(' ');
disp('3-parameter model:');
disp(' ');
fprintf('Alpha recoverability: r = %.2g\n',Correlations_alpha_3(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_alpha_3(1,2));
disp(' ');
fprintf('Risk-seeking recoverability: r = %.2g\n',Correlations_rs_3(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_rs_3(1,2));
disp(' ');
fprintf('Learning rate recoverability: r = %.2g\n',Correlations_eta_3(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_eta_3(1,2));
disp(' ');
%% Organize and save results

two_parameter_model_estimates.alpha_true = recover_check_alpha_2(:,1);
two_parameter_model_estimates.alpha_estimated = recover_check_alpha_2(:,2);
two_parameter_model_estimates.risk_seeking_true = recover_check_rs_2(:,1);
two_parameter_model_estimates.risk_seeking_estimated = recover_check_rs_2(:,2);
two_parameter_model_estimates.final_log_likelihoods = avg_LL_2_params;
two_parameter_model_estimates.final_action_probabilities = avg_prob_2_params;
two_parameter_model_estimates.protected_exceedance_probability = pxp;

three_parameter_model_estimates.alpha_true = recover_check_alpha_3(:,1);
three_parameter_model_estimates.alpha_estimated = recover_check_alpha_3(:,2);
three_parameter_model_estimates.risk_seeking_true = recover_check_rs_3(:,1);
three_parameter_model_estimates.risk_seeking_estimated = recover_check_rs_3(:,2);
three_parameter_model_estimates.learning_rate_true = recover_check_eta_3(:,1);
three_parameter_model_estimates.learning_rate_estimated = recover_check_eta_3(:,2);
three_parameter_model_estimates.final_log_likelihoods = avg_LL_3_params;
three_parameter_model_estimates.final_action_probabilities = avg_prob_3_params;
three_parameter_model_estimates.protected_exceedance_probability = pxp;

save('Two_parameter_model_estimates','two_parameter_model_estimates');
save('Three_parameter_model_estimates','three_parameter_model_estimates');
save('GCM_2','GCM_2');
save('GCM_3','GCM_3');

figure
scatter(two_parameter_model_estimates.alpha_true,two_parameter_model_estimates.alpha_estimated,'filled')
lsline
title('Recoverability: Alpha (two-parameter model)')
xlabel('True (Generative) Alpha') 
ylabel('Estimated Alpha') 
[Corr_alpha_2, Sig_alpha_2] = corrcoef(two_parameter_model_estimates.alpha_true,two_parameter_model_estimates.alpha_estimated);
text(1, 23, ['r = ' num2str(Corr_alpha_2(1,2))])
text(1, 22, ['p = ' num2str(Sig_alpha_2(1,2))])

figure
scatter(two_parameter_model_estimates.risk_seeking_true,two_parameter_model_estimates.risk_seeking_estimated,'filled')
lsline
title('Recoverability: Risk-Seeking (two-parameter model)')
xlabel('True (Generative) Risk-Seeking') 
ylabel('Estimated Risk-Seeking') 
[Corr_rs_2, Sig_rs_2] = corrcoef(two_parameter_model_estimates.risk_seeking_true,two_parameter_model_estimates.risk_seeking_estimated);
text(4.1, 6.75, ['r = ' num2str(Corr_rs_2(1,2))])
text(4.1, 6.5, ['p = ' num2str(Sig_rs_2(1,2))])

figure
scatter(three_parameter_model_estimates.alpha_true,three_parameter_model_estimates.alpha_estimated,'filled')
lsline
title('Recoverability: Alpha (three-parameter model)')
xlabel('True (Generative) Alpha') 
ylabel('Estimated Alpha') 
[Corr_alpha_3, Sig_alpha_3] = corrcoef(three_parameter_model_estimates.alpha_true,three_parameter_model_estimates.alpha_estimated);
text(1, 29, ['r = ' num2str(Corr_alpha_3(1,2))])
text(1, 27, ['p = ' num2str(Sig_alpha_3(1,2))])

figure
scatter(three_parameter_model_estimates.risk_seeking_true,three_parameter_model_estimates.risk_seeking_estimated,'filled')
lsline
title('Recoverability: Risk-Seeking (three-parameter model)')
xlabel('True (Generative) Risk-Seeking') 
ylabel('Estimated Risk-Seeking') 
[Corr_rs_3, Sig_rs_3] = corrcoef(three_parameter_model_estimates.risk_seeking_true,three_parameter_model_estimates.risk_seeking_estimated);
text(4.1, 6.75, ['r = ' num2str(Corr_rs_3(1,2))])
text(4.1, 6.5, ['p = ' num2str(Sig_rs_3(1,2))])

figure
scatter(three_parameter_model_estimates.learning_rate_true,three_parameter_model_estimates.learning_rate_estimated,'filled')
lsline
title('Recoverability: Learning Rate (three-parameter model)')
xlabel('True (Generative) Learning Rate') 
ylabel('Estimated Learning Rate') 
[Corr_lr_3, Sig_lr_3] = corrcoef(three_parameter_model_estimates.learning_rate_true,three_parameter_model_estimates.learning_rate_estimated);
text(.25, .53, ['r = ' num2str(Corr_lr_3(1,2))])
text(.25, .52, ['p = ' num2str(Sig_lr_3(1,2))])


if PEB == 1
%% 10. Hierarchical Bayes (between-subjects)
%==========================================================================
% clear and re-load saved GCMs for second-level analyses

% This will allow you to reload the GCM data later to use PEB without 
% needing to re-run the 'Sim = 5' option.

clear GCM_2 
clear GCM_3
load('GCM_2.mat')
load('GCM_3.mat')
%==========================================================================

%--------------------------------------------------------------------------
% Using PEB, you can test  the evidence for a 'full' model that assumes a
% group difference in all parameters and simpler models that assume no
% differences in one or more parameters.
% 
% This allows testing evidence for a difference (or for no difference)
% in estimated parameters. PEB uses a general linear (random effects) model,  
% which also allows testing evidence for individual difference effects 
% (e.g., age, symptom severity).

% See relevant literature on these routines, e.g., 
% Friston, Litvak, Oswal, Razi, Stephan, van Wijk, Ziegler, & Zeidman, 2016
% Zeidman, P., Jafarian, A., Seghier, M. L., Litvak, V., et al., 2019
%--------------------------------------------------------------------------

% Second-level model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First, specify whether you want to use the 2- or 3-parameter model:

GCM_PEB = GCM_3; % either GCM_2 or GCM_3
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Specify default PEB parameters and between-subjects model (M)

M       = struct();
M.alpha = 1;        % prior PEB parameter variance = 1/alpha
M.beta  = 16;       % prior expectation of between-subject variability (random effects precision) = 1/(prior DCM parameter variance/M.beta)
M.hE    = 0;        % default
M.hC    = 1/16;     % default
M.Q     = 'all';    % covariance components: {'single','fields','all','none'}
M.X     = [];       % design matrix for general linear model

M.X = ones(length(GCM_PEB),1); % first column in general linear model is the mean of all participants

for i = 1:length(GCM_PEB)
    if i < (length(GCM_PEB)/2)+1 % in this simulation group 1 is the first half of the simulated sample
        M.X(i,2) = 1;            % and group 2 is the second
    else
        M.X(i,2) = -1;
    end
end

M.X(:,3) = 30 + 5.*randn(size(M.X,1),1); % Simulate a range of ages (mean = 30, SD = 5)
    
M.X(:,2) = detrend(M.X(:,2),'constant'); % Center group values around 0
M.X(:,3) = detrend(M.X(:,3),'constant'); % Center age values around 0

PEB_model  = spm_dcm_peb(GCM_PEB,M); % Specify PEB model
PEB_model.Xnames = {'Mean','Group','Age'}; % Specify covariate names

[BMA,BMR] = spm_dcm_peb_bmc(PEB_model); % Estimate PEB model

spm_dcm_peb_review(BMA,GCM_PEB); % Review results

% If you select the 'Second-level effect - Group' you can see that rs is
% significantly different between groups

% Please see main text for further information about how to interpet 
% results figures
    
end

end

%==========================================================================
% This completes the tutorial script. By adapting these scripts you can 
% now build a generative model of a task, run simulations, assess parameter
% recoverability, do bayesian model comparison, and do hierarchical
% bayesian group analyses. See the main text for further explanation of
% other aspects of these steps.
