# Intelligent Systems Project 2 -- Time Series Forecasting on X-Ray Flux Data

## Description

- In this project, I trained several different RNN's to forecast x-ray flux from the sun. I experimented with several architectures, all of which are in the `models.py` file. 
- X-ray flux data is important because it is how we detect solar flares. Major solar flares can cause radio blackouts. 
- I obtained the data from https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes15/xrsf-l2-flsum_science/sci_xrsf-l2-flsum_g15_s20100331_e20200304_v1-0-0.nc
- All of my models normalize the data, so the models output is also normalized. For real application, this data will need to be de-normalized.
- For bulldozing, I used the NVidia GeForce 1660 Ti GPU on my laptop

## Running the Code

### Loading and Evaluating a Pre-Trained Network

In this repository I have included an ensemble of pre-trained networks, all of which are contained in the `models/` directory. If you would like to evaluate the loss and accuracy of one of these pre-trained and persisted networks, follow the instructions below:

1. Navigate to the `src/` directory by running the command `$ cd src` from the parent directory of this repository
2. Choose a model from the `models/` directory to evaluate. For this example I will use the `triple_rnn` model
3. Run the command `$ python fullEvaluation.py triple_rnn`
   - You may substitute a different model name than `triple_rnn` to the program. Be sure that the model name matches the name of one of the models in the `models/` directory.
   - Do not supply the path the to model, and do not include the `.keras` in the argument.
4. This program will evaluate the model on the **entire** dataset (training, validation, and testing data). It will also generate a plot of it predictions against the ground truth in `images/fullEvaluation/{model_name}.png`. 

### Training a Network From Scratch

If you would like to train a new network from scratch and evaluate that network, follow the instructions below:

1. Navigate to the `src/` directory by running the command `cd src` from the parent directory of this repository
2. Choose an architecture from `models.py` to train. To select which model to use, simply change the `fcn` argument of the `getModel` function to be the function which returns the architecture of your choosing. 
3. Run the command `python trainRNN.py {modelName}`, where `{modelName}` is the name of the model that you want to persist.
   - Do not supply a path to the program as `{modelName}`. Do not include a file extension with `{modelName}`. The program will automatically add the `.keras` extension. 
4. This program will train the network, evaluate it on the test data, and then run `fullEvaluation.py`. At the end of its execution, there will be a persisted Keras model in `models/{modelName}.keras`. There will also be a plot of the predictions of the model against the ground truth on the **test** data saved in `images/training/{modelName}.png`. After the execution of `fullEvaluation.py` completes, there will be a plot of the predictions of the model against the ground truth on the **entire** dataset saved in `images/fullEvaluation/{modelName}.png`.

## Results

- For this problem, using mean-absolute-error yields far better results than using mean-squared-error. MSE yielded high losses (~0.9) and low accuracies (< 0.2)
- For the architectures which use an exponentially decaying learning rate, an initial learning rate of 1e-4 offered better results than using an initial learning rate of 1e-2 or 1e-3
- Plots of network performance on the test data are given in the `images/training` directory. Plots of network performance on the entire dataset are given in the `images/fullEvaluation` directory.
- Below is a table which contains each architecture I trained, and its corresponding Test Loss and Test Accuracy

| Architecture           | Test Loss (MAE) | Test Accuracy |
|------------------------|-----------------|---------------|
| Simple RNN             | 0.095           | 0.248         |
| Simple RNN w/ LR Decay | 0.088           | 0.397         |
| Double RNN             | 0.092           | 0.291         |
| Double RNN w/ LR Decay | 0.088           | 0.489         |
| Short Deep Net         | 0.092           | 0.451         |
| Short Deep w/ LR Decay | 0.086           | 0.482         |
| Triple RNN             | 0.098           | 0.359         |
| Triple RNN w/ LR Decay | 0.087           | 0.510         |

- Short Deep w/ LR Decay, Triple RNN w/ LR Decay, and Double RNN w/ LR Decay performed best, with low loss and high accuracy. I predict that this is because I ran them with an initial learning rate of 1e-4 instead of 1e-3 (which is the initial learning rate I used when I ran the other networks with LR Decay). 


## Summary

One of the things that could greatly improve the effectiveness of the Neural Networks for making predictions about X-Ray Flux data is altering the data set to be more well behaved. The data used for this project has non-uniform time stamps, and some outliers. The data could be altered slightly to make the times between data points uniform. Additionally, outliers could be discarded. This would likely lead to increases in the accuracy of the networks. 

Given more time, a different selection of dataset might be advantageous. Specifically, the data in the file `data/sci_xrsf-l2-avg1m_g15_s20100331_e20200304_v1-0-0.nc` would be a good starting point, since it is more complete. In that file, there is much more data, but it may be difficult to get in a format that works well with the neural networks. If that data can be wrangled, however, I am confident that it would produce more robust predictions. See https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes15/ for other data that may be used for predictions about x-ray flux.

Another thing that could be tampered with is the learning rate, activation functions, and loss function. I found that using mean absolute error for a loss function resulted in far better results than using mean squared error, but more experimentation with loss functions might bring to light a different loss function which is even more effective. Tampering with the initial learning rate and rate of decay for the learning rates could help to get the network to train to higher accuracies as well. 

All in all, I was able to produce several networks that predicted the trends in x-ray flux from the sun fairly well. There are certainly more steps that can be taken in order to increase the effectiveness of my networks, but given time and resource constraints, I feel that my networks perform well.