# project2_PINN_ACJ

This repository contains all the codes and files related to project 2 of CS-433 Machine Learning. The aim of this project is to implement a Physic-informed neural network (PINN) in order to first predict the forcing acting on a system based on discrete masses and springs. After that, the last step is to invert the predicted force in order to obtain the stifness matrix and therefore check if our PINN learned the physics. 

# Project structure 

## Dataset description
For this project, two data sets were used. Both have the same structure. The first data set $S_{1}$ can be found in the file FILE_NAME and is represented by a 4000x23 matrix. This data set contains the position, velocity and acceleration of a system of 5 discrete masses linked by springs. The masses at the end are fixed and therefore don't move. The simulation length is 40 seconds and the time step is 0.01 s. Therefore, there is 4000 observations. The i-th row represent the i-th observation. The first column of this matrix represent the time period. The two next represent the x and y position of the first fixed mass. The next ones represent the position, velocity and acceleration of the movable masses in x and y direction. This is about 18 columns (for each of the 3 movable masses : 2 cols for position, 2 cols velocity and 2 cols for acceleration). Finally, the last two columns represent the position of the other fixed mass. 


## File structure 
add .... 


# Team member 
This project was done in the scope of the CS-433 course by : 
*   `Jonathan Chuah`
*   `Tz-Ching Yu`
*   `Albias Havolli `
