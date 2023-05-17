# Unconflow-PINN

# What is this repository for?
Unconflow-PINN is a software for solving the unconfined aquifer transient flow problem by using PINN. 

# Highlights
PINNs accurately compute the time-varying phreatic surface and piezometric heads; PINNs have proven to be very effective in data-scarce environments; PINNs are a promising alternative to classical numerical methods in hydrogeology.

# How to use the code
The two available folders "SC1" and "SC2" contain the code to solve the case of a homogeneous and isotropic, and heterogeneous and anisotropic unconfined aquifer, respectively.

From a programming language perspective, the two cases are solved in the same way, and therefore, instructions on how to use scripts, functions, and data will be provided only in reference to one of the two aforementioned folders.

Let's start by downloading the SC1 folder, inside of which we will find 3 scripts, 7 functions, 1 .mat data file, and 1 .xlsx file.

The main script is "Unconfined_homogeneous_isotropic.m". Inside, you will find the main body of the code used to develop the PINN. It is the first script that needs to be run to start training the neural network.

Within this script, 7 functions will be used. The first two, "initializeHe.m" and "initializeZeros.m", are used to initialize the weights and biases of the neural networks.

In this work, two structurally identical neural networks, with the only difference being their input and output layers, are used. The first network (ANN1) is trained to compute the piezometric head value (output) using the point coordinates (𝑥, 𝑧) and the time (𝑡) as inputs. The second network (ANN2), which takes the 𝑥 coordinate and the time as input values, returns the 𝑧 coordinate value (output) that indicates the position of the free surface at a specific time. Although both networks could be trained simultaneously from the beginning using a single loss function, we found that it is more efficient if there is a preliminary iteration in which ANN1 is trained first, and then ANN2 is trained next (with ANN1 fixed). The weights and biases found in this preliminary iteration are used as the starting values for the joint training of the two networks.

Clarifying the concept of using 2 neural networks with different objectives, we proceed by saying that the function "model.m" represents ANN1, while "model_2.m" defines ANN2.

The function "modelGradients_parameters.m" is used to evaluate the gradient during the training of ANN1, while "modelGradients_parameters2.m" is used to evaluate the gradient during the training of ANN2. Finally, "modelGradients.m" is the function that evaluates the gradient during the joint training of the two ANNs.

This concludes the PINN training phase. Next, you can save the result in a MAT file, which in our case was named "results_lower_observations(90)_200epochs_25000CP.mat", because we removed 90% of the observations (MODFLOW results) that you will find in the "Observations_homogeneous_isotropic.xlsx" file.

After training the network, what will actually need to be saved in the MAT file are the variables "parameters" and "parameters_2," which contain the weight values of the network at the end of the training process.

After training, the networks undergo validation by comparing their predictions with the results obtained from MODFLOW. The root mean square error (RMSE) is calculated for the piezometric heads at the center points of the discretization grid and the elevation of the phreatic surface at specific time intervals (0.01, 0.25, 0.5, and 1). The neural networks generate predictions by inputting coordinates (𝑥, 𝑧, 𝑡) into ANN1 and (𝑥, 𝑡) into ANN2. 

The script "ANN1_evaluation.m" offers both quantitative and visual comparisons of the piezometric head between the network predictions and the MODFLOW predictions calculated at the center points of the active discretization cells in the numerical model. Contour maps, along with the discrepancies, are presented.

Once the proper functioning of ANN1 has been verified, we need to validate the ability of the two networks to work together and accurately estimate the position of the phreatic surface and the piezometric head below it. The script "PINN_vs_NumericalModel" generates piezometric head maps at the specified times using both the MODFLOW simulation and the PINN prediction. It is important to note that the MODFLOW maps are generated using a pixel-based discretization method, while the PINN maps are constructed with a denser discretization, that the user can define (in this script discr=200).

# Do you have any question?
Feel free to contact us at this email address: daniele.secci@unipr.it
