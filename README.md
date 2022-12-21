# project2_PINN_ACJ

This repository contains all the codes and files related to project 2 of CS-433 Machine Learning. The aim of this project was to implement a Physic-informed neural network (PINN) in order to first predict the forcing acting on a system based on discrete masses and springs. After that, the last step was to invert the predicted force in order to obtain the stifness matrix and therefore check if our PINN learned the physics. 

# Project structure 

## Dataset description
For this project, two data sets were used. Both have the same structure. The first data set $S_{1}$ can be found in the file FILE_NAME and is represented by a 4000x23 matrix. This data set contains the position, velocity and acceleration of a system of 5 discrete masses linked by springs. The masses at the end are fixed and therefore don't move. The simulation length is 40 seconds and the time step is 0.01 s. Therefore, there is 4000 observations. The i-th row represent the i-th observation. The first column of this matrix represent the time period. The two next represent the x and y position of the first fixed mass. The next ones represent  the positions $x$ and $y$, the velocities $\dot{x}$ and $\dot{y}$, and the accelerations $\ddot{x}$ and $\ddot{y}$ of each movable masses. This is about 18 columns (for each of the 3 movable masses : 2 cols for position, 2 cols velocity and 2 cols for acceleration). Finally, the last two columns represent the position of the other fixed mass. The same analogy can be applied to the second data set $S_{2}$ which contains 8 masses. Therefore, the data can be represented by a 4000x41 matrix. 

The data set $S_{1}$ was used to train our PINN and the second one $S_{2}$ to test it. 


## File structure 
- simulator : this folder contains all the files needed to solve the PDEs of a system composed of masses and springs. The 'SpringMassSystem.py' file contains some helper functions. The notebook 'SpringMassSimulation_datagen.ipynb' allows to choose the following parameters : 
     - the number of masses $N$ 
     - the rest length $L_{0}$ 
     - the spring stiffness $k$ 
     - the mass of all the masses $M$.
     - the simulation length $T_{d}$
     - the step time $T_{s}$
    
Having those parameters, the notebook allows to compute the positions $x$ and $y$, the velocities $\dot{x}$ and $\dot{y}$, and the accelerations $\ddot{x}$ and $\ddot{y}$ of each masses. Those results are then stored in a csv file and then plotted. 

- derivations : this folder don't contain any code. It only contains docs where we put our ideas and different reasoning from the start to the very end. It may be useful for anybody who wants to provide further work related to the project. In particularly, the file 'PINN constraints for a spring-mass system.docx' contains the detailled derivations of our losses and NN module.

- hyperparameters_tests : this folder contains all the files related to the hyperparameters which goal is to find the combination of hidden layers and neurons that results to the lowest error between the truth trajectories and the ones predicted by our PINN. This folder contains a notebook 'final_hyperparamters_tests.ipynb' that has been run cell by cell to compute our results, a data file 'k=50_L=7_N=5_M=1.csv' that contains the data set used for the hyperparameters tests and 4 npy files that contains our results. 

- src : this folder contains some helpers py files used to compute the different losses as well as the trajectories, the velocities and the acceleration of each movable masses. 

- varing_num_masses : This folder contains a notebook 'varing_masses_experiment.ipynb' where we trained and tested our PINN with different set-up by changing the nb of masses. For each set-up, we computed the predited forces as well as the predicted trajectories. 


# Team member 
This project was done in the scope of the course CS-433 Machine Learning by : 
*   `Jonathan Chuah`
*   `Tz-Ching Yu`
*   `Albias Havolli `
