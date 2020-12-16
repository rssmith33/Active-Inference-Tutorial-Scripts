# Active-Inference-Tutorial-Scripts

Supplementary scripts for Step-by-step active inference modelling tutorial

By Ryan Smith and Chris Whyte

Step_by_Step_AI_Guide.m: 

This is the main tutorial script. It illustrates how to build a partially observable Markov decision process (POMDP) model within the active inference framework, using  a simple explore-exploit task as an example. It shows how to run single trial and multi-trial simulations including perception, decision-making, and learning. It also shows how to generate simulated neuronal responses. It further illustrates how to fit task models to empirical data for behavioral studies and do subsequent Bayesian group analyses.

Step_by_Step_Hierarchical_Model:

Separate script illustrating how to build a hierarchical (deep temporal) model, using a common oddball task paradigm as an example. This also shows how to simulate predicted neuronal responses (event-related potentials) observed using this task in empirical studies.

Pencil_and_paper_exercise_solutions:

Solutions to 'pencil and paper' exercises for solving equations used in active inference.

spm_MDP_VB_X_tutorial:

Tutorial version of the standard routine for running active inference POMDP models.

Simplified_simulation_script:

Simplified and heavily commented version of the spm_MDB_VB_X_tutorial script. This is provided to make it easier to understand how the standard simulation routines work.

Estimate_parameters: 

Script called by the main tutorial script for estimating parameters on (simulated) behavioral data.

All other scripts are just secondary functions called by the main scripts for plotting simulation outputs.
