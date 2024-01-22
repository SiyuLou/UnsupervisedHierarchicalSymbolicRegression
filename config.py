
## define features for the Rf governing equation
solvent = ['H', 'EA', 'DCM', 'MeOH', 'Et2O']
#solute
FG_dist = ['Benzene', 'FG_Position_correct','Dipole Moment (a.u.)']
functional_O = ['phenol', 'OH','aldehyde','CO2H','RCO2R','R2C=O','ROR']
functional_N = ['C#N','RNH2']
functional_O_N = ['NO2','Amides']
functional_halogen = ['F', 'Cl', 'Br', 'I']
functional_others = ['methyl']
functional_all = functional_O + functional_N + functional_O_N + functional_halogen + functional_others

feature_names = solvent + FG_dist + functional_all
feature_nums = [len(solvent),len(FG_dist+functional_all)]

#########-------------------------------------#################
## define features for solute polarity index \xi governing equation
feature_names_solute = FG_dist + functional_all
feature_nums_solute = [len(FG_dist), len(functional_all)]

#########-------------------------------------#################
## define features for FG polarity index \beta governing equation
functional_1 =['Amides','CO2H']
functional_2 = ['RNH2','OH','phenol']
functional_3 = ['NO2','RCO2R']
functional_4 = ['aldehyde','R2C=O','ROR','C#N','F']
functional_5 = ['Cl','Br','I','methyl']
feature_names_functional = functional_1 + functional_2+functional_3+ functional_4+functional_5
feature_nums_functional = [len(functional_1), len(functional_2), len(functional_3), len(functional_4), len(functional_5)]
