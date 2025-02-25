# ANOMIX
A PyTorch implementation of "[ANOMIX: A Simple yet Effective Graph Mixing for GAD]"


# Dependencies
  Dependencies for all experiments of ANOMIX are as follows:
  
   • Python == 3.7.8
   
   • PyTorch == 1.6.0
   
   • NetworkX == 2.6.3
   
   • Scikit-learn == 0.23.2
   
   • NumPy == 1.18.5
   
   • SciPy == 1.7.3

   • DGL == 0.4.1

# Usage
All of datasets used in this paper are put in ./dataset folder and graph information (e.g., adjacency, attribute, and label) is included in each dataset file (.mat). 

    python run.py --dataset cora

