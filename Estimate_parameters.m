function [DCM] = Estimate_parameters(DCM)

% MDP inversion using Variational Bayes
% FORMAT [DCM] = spm_dcm_mdp(DCM)
%
% Expects:
%--------------------------------------------------------------------------
% DCM.MDP   % MDP structure specifying a generative model
% DCM.field % parameter (field) names to optimise
% DCM.U     % cell array of outcomes (stimuli)
% DCM.Y     % cell array of responses (action)
%
% Returns:
%--------------------------------------------------------------------------
% DCM.M     % generative model (DCM)
% DCM.Ep    % Conditional means (structure)
% DCM.Cp    % Conditional covariances
% DCM.F     % (negative) Free-energy bound on log evidence
% 
% This routine inverts (cell arrays of) trials specified in terms of the
% stimuli or outcomes and subsequent choices or responses. It first
% computes the prior expectations (and covariances) of the free parameters
% specified by DCM.field. These parameters are log scaling parameters that
% are applied to the fields of DCM.MDP. 
%
% If there is no learning implicit in multi-trial games, only unique trials
% (as specified by the stimuli), are used to generate (subjective)
% posteriors over choice or action. Otherwise, all trials are used in the
% order specified. The ensuing posterior probabilities over choices are
% used with the specified choices or actions to evaluate their log
% probability. This is used to optimise the MDP (hyper) parameters in
% DCM.field using variational Laplace (with numerical evaluation of the
% curvature).
%
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_mdp.m 7120 2017-06-20 11:30:30Z spm $

% OPTIONS
%--------------------------------------------------------------------------
ALL = false;

% Here we specify prior expectations (for parameter means and variances)
%--------------------------------------------------------------------------
prior_variance = 1/4; % smaller values will lead to a greater complexity 
                      % penalty (posteriors will remain closer to priors)

for i = 1:length(DCM.field)
    field = DCM.field{i};
    try
        param = DCM.MDP.(field);
        param = double(~~param);
    catch
        param = 1;
    end
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        if strcmp(field,'alpha')
            pE.(field) = log(16);          % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'beta')
            pE.(field) = log(1);           % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'la')
            pE.(field) = log(1);           % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'rs')
            pE.(field) = log(5);           % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'eta')
            pE.(field) = log(0.5/(1-0.5)); % in logit-space - bounded between 0 and 1
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'omega')
            pE.(field) = log(0.5/(1-0.5)); % in logit-space - bounded between 0 and 1
            pC{i,i}    = prior_variance;
        else
            pE.(field) = 0;                % if it can take any negative or positive value
            pC{i,i}    = prior_variance;
        end
    end
end

pC      = spm_cat(pC);

% model specification
%--------------------------------------------------------------------------
M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
M.pE    = pE;                            % prior means (parameters)
M.pC    = pC;                            % prior variance (parameters)
M.mdp   = DCM.MDP;                       % MDP structure

% Variational Laplace
%--------------------------------------------------------------------------
[Ep,Cp,F] = spm_nlsi_Newton(M,DCM.U,DCM.Y); % This is the actual fitting routine

% Store posterior distributions and log evidence (free energy)
%--------------------------------------------------------------------------
DCM.M   = M;  % Generative model
DCM.Ep  = Ep; % Posterior parameter estimates
DCM.Cp  = Cp; % Posterior variances and covariances
DCM.F   = F;  % Free energy of model fit

return

function L = spm_mdp_L(P,M,U,Y)
% log-likelihood function
% FORMAT L = spm_mdp_L(P,M,U,Y)
% P    - parameter structure
% M    - generative model
% U    - inputs
% Y    - observed repsonses
%
% This function runs the generative model with a given set of parameter
% values, after adding in the observations and actions on each trial
% from (real or simulated) participant data. It then sums the
% (log-)probabilities (log-likelihood) of the participant's actions under the model when it
% includes that set of parameter values. The variational Bayes fitting
% routine above uses this function to find the set of parameter values that maximize
% the probability of the participant's actions under the model (while also
% penalizing models with parameter values that move farther away from prior
% values).
%__________________________________________________________________________

if ~isstruct(P); P = spm_unvec(P,M.pE); end

% Here we re-transform parameter values out of log- or logit-space when 
% inserting them into the model to compute the log-likelihood
%--------------------------------------------------------------------------
mdp   = M.mdp;
field = fieldnames(M.pE);
for i = 1:length(field)
    if strcmp(field{i},'alpha')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'beta')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'la')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'rs')
        mdp.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'eta')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    elseif strcmp(field{i},'omega')
        mdp.(field{i}) = 1/(1+exp(-P.(field{i})));
    else
        mdp.(field{i}) = exp(P.(field{i}));
    end
end

% place MDP in trial structure
%--------------------------------------------------------------------------
la = mdp.la_true;  % true level of loss aversion
rs = mdp.rs_true;  % true preference magnitude for winning (higher = more risk-seeking)

if isfield(M.pE,'la')&&isfield(M.pE,'rs')
    mdp.C{2} = [0  0       0   ;      % Null
                0 -mdp.la -mdp.la  ;  % Loss
                0  mdp.rs  mdp.rs/2]; % win
elseif isfield(M.pE,'la')
    mdp.C{2} = [0  0       0   ;      % Null
                0 -mdp.la -mdp.la  ;  % Loss
                0  rs      rs/2];     % win
elseif isfield(M.pE,'rs')
    mdp.C{2} = [0  0       0   ;      % Null
                0 -la     -la  ;      % Loss
                0  mdp.rs  mdp.rs/2]; % win
else
    mdp.C{2} = [0  0   0   ;  % Null
                0 -la -la  ;  % Loss
                0  rs  rs/2]; % win
end

j = 1:numel(U); % observations for each trial
n = numel(j);   % number of trials

[MDP(1:n)] = deal(mdp);  % Create MDP with number of specified trials
[MDP.o]    = deal(U{j}); % Add observations in each trial

% solve MDP and accumulate log-likelihood
%--------------------------------------------------------------------------
MDP   = spm_MDP_VB_X_tutorial(MDP); % run model with possible parameter values

L     = 0; % start (log) probability of actions given the model at 0

for i = 1:numel(Y) % Get probability of true actions for each trial
    for j = 1:numel(Y{1}(:,2)) % Only get probability of the second (controllable) state factor
        
        L = L + log(MDP(i).P(:,Y{i}(2,j),j)+ eps); % sum the (log) probabilities of each action
                                                   % given a set of possible parameter values
    end
end 

clear('MDP')

fprintf('LL: %f \n',L)
