# Active-Inference-Tutorial-Scripts

Supplementary scripts for Step-by-step active inference modelling tutorial

By Ryan Smith and Christopher Whyte

Step_by_Step_AI_Guide.m: 

This is the main tutorial script. It illustrates how to build a partially observable Markov decision process (POMDP) model within the active inference framework, using  a simple explore-exploit task as an example. It shows how to run single-trial and multi-trial simulations including perception, decision-making, and learning. It also shows how to generate simulated neuronal responses. It further illustrates how to fit task models to empirical data for behavioral studies and do subsequent Bayesian group analyses.

Step_by_Step_Hierarchical_Model:

Separate script illustrating how to build a hierarchical (deep temporal) model, using a commonly used oddball task paradigm as an example. This also shows how to simulate predicted neuronal responses (event-related potentials) observed using this task in empirical studies.

EFE_Precision_Updating:

Separate script that allows the reader to simulate updates in the expected free energy precision (gamma) through updates in its prior (beta). At the top of the script you can choose values for the prior over policies, expected free energy over policies, and variational free energy over policies after a new observation, as well as the initial prior on expected precision. The script will then simulate 16 iterative updates and plot the resulting changes in gamma. By changing the initial values of the priors and free energies, you can get more of an intuition about the dynamics of these updates and how they depend on the relationship between the initial values that are chosen.

VFE_calculation_example:

Separate script that allows the reader to calculate variational free energy for approximate posterior beliefs given a new observation. The reader can specify a generative model (priors and likelihood matrix) and an observation, and then experiment with how variational free energy is reduced as approximate posterior beliefs approach the true posteriors.

Prediction_error_example:

Separate script that allows the reader to calculate state and outcome prediction errors. These minimize variational and expected free energy, respectively. Minimizing state prediction errors maintains accurate beliefs (while also changing beliefs as little as possible). Minimizing outcome prediction errors maximizes reward and information gain.

Message_passing_example:

Separate script that allows the reader to perform (marginal) message passing. In the first example, the code follows the message passing steps described in the main text (section 2) one by one. In the second example, this is extended to also calculate firing rates and ERPs associated with message passing in the neural process theory associated with active inference.

EFE_learning_novelty_term:

Separate script that allows the reader to calculate the novelty term that is added to the expected free energy when learning the Dirichlet concentration parameters (a) for the likelihood matrix (A). Small concentration parameters lead to a larger value for the novelty term, which is subtracted from the total EFE value for a policy. Therefore, less confidence in beliefs about state-outcome mappings in the A matrix lead the agent to select policies that will increase confidence in those beliefs ('parameter exploration').

Pencil_and_paper_exercise_solutions:

Solutions to 'pencil and paper' exercises provided in the tutorial paper. These are provided to aid the reader in developing intuitions for the equations used in active inference.

spm_MDP_VB_X_tutorial:

Tutorial version of the standard routine for running active inference (POMDP) models.

Simplified_simulation_script:

Simplified and heavily commented version of the spm_MDB_VB_X_tutorial script. This is provided to make it easier for the reader to understand how the standard simulation routines work.

Estimate_parameters: 

Script called by the main tutorial script for estimating parameters on (simulated) behavioral data.

Note: Additional scripts are secondary functions called by the main scripts for plotting simulation outputs.
