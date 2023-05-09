# Unconflow-PINN
The unconfined aquifer transient flow problem is solved by using PINN; PINNs accurately compute the time-varying phreatic surface and piezometric heads; PINNs have proven to be very effective in data-scarce environments; PINNs are a promising alternative to classical numerical methods in hydrogeology.

The two available folders "SC1" and "SC2" contain the code to solve the case of a homogeneous and isotropic, and heterogeneous and anisotropic unconfined aquifer, respectively.

From a programming language perspective, the two cases are solved in the same way, and therefore, instructions on how to use scripts, functions, and data will be provided only in reference to one of the two aforementioned folders.

Let's start by downloading the SC1 folder, inside of which we will find 2 scripts, 7 functions, 1 .mat data file, and 1 .xlsx file.

The main script is "Unconfined_homogeneous_isotropic.m". Inside, you will find the main body of the code used to develop the PINN. It is the first script that needs to be run to start training the neural network.

Within this script, 7 functions will be used. The first two, "initializeHe.m" and "initializeZeros.m", are used to initialize the weights and biases of the neural networks.

In this work, two structurally identical neural networks, with the only difference being their input and output layers, are used. The first network (ANN1) is trained to compute the piezometric head value (output) using the point coordinates (𝑥, 𝑧) and the time (𝑡) as inputs. The second network (ANN2), which takes the 𝑥 coordinate and the time as input values, returns the 𝑧 coordinate value (output) that indicates the position of the free surface at a specific time. Although both networks could be trained simultaneously from the beginning using a single loss function, we found that it is more efficient if there is a preliminary iteration in which ANN1 is trained first, and then ANN2 is trained next (with ANN1 fixed). The weights and biases found in this preliminary iteration are used as the starting values for the joint training of the two networks.

Clarifying the concept of using 2 neural networks with different objectives, we proceed by saying that the function "model.m" represents ANN1, while "model_2.m" defines ANN2.

The function "modelGradients_parameters.m" is used to evaluate the gradient during the training of ANN1, while "modelGradients_parameters2.m" is used to evaluate the gradient during the training of ANN2. Finally, "modelGradients.m" is the function that evaluates the gradient during the joint training of the two ANNs.

This concludes the PINN training phase. Next, you can save the result in a MAT file, which in our case was named "results_lower_observations(90)_200epochs_25000CP.mat", because we removed 90% of the observations (MODFLOW results) that you will find in the "Observations_homogeneous_isotropic.xlsx" file.
