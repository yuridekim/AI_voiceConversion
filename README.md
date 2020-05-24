# Many-to-Many Voice Conversion Using CycleVAE
Deep generative model experiments
1. Define your translation task of interest.
2. Prepare training, validation, and test data.
3. Build a deep generative model (e.g, GAN, VAE, …)
4. Train the neural network and find the best hyper-parameters using training and validation data.
5. Test the neural network using the test data.

# Description of the translation task
I used Cycle VAE to implement a many-to-many voice conversion.   
I changed A(female), B and C(male) to my voice(female).   
I maintained the speed, volume or the conversation language info but changed the voice only.   
A,B,C was used as Source, and my voice as the Target data.   


# Training/Validation/Test data
For A, B, C(Source), I used training & evaluation data that was provided [here](https://datashare.is.ed.ac.uk/handle/10283/3061)   
I assigned VCC2SF1 to A and VCC2SM1, VCC2SM2 for B and C each.   
For my voice as the target, I recorded it myself with audacity.   
There were #81 train datas and #35 test datas.   

To evaluated the model, I used a part of train data as my validation data using split in python(8:2 proportion)

# Model
Used model from [here](https://github.com/positivewon/AI_Homework_VC)

# Description of the compiling, training, and testing procedures
For learning the model, I used Google Colaboratory   
First, we process our data for use with preprocess-eval, preprocess-train   
For learning our models, we use train.py.   
In CycleVAE, we use multiple decoders for better quality along with an encoder.   
The encoder transforms the input into a latent vector which includes our language information. Then, it goes into the decoder along with the speaker identity so that it has the features of the target speaker.   
To implement this, we utilize loss.py to calculate and minimize reconstruction loss, Kullback-Leibler divergence.     
Right after half the epoch, we start cycle training to calculate and minimze the cycle loss and updates our weights.   
The epoch with the lowest validation set error will be chosen.

Finally we convert and save the voices into individual files using convert.py

# Description of the hyper-paramters tuning experiments
To find the optimal model, I adjusted the hyper parameters(learning rate, batch size, epoch)   
After using the default parameters, I tried to minimze the batch size, maximize epochs, and adjusted learning rate around 0.001.   
I deviated from these parameters depending on the previous results so that it would not be overfit/underfit.   
Here are the 9 adjustments so that I could win the smallest validation error(reconstruction loss+cycle loss).   

1. learning rate: 0.0005, batch size:4, epoch: 1000(default)
Validation의 Error: 2.2953
2. learning rate: 0.0005, batch size:4, epoch: 500
Validation의 Error: 2.3770
* I set epoch to 500 temporarily while I could reach the result of adjusting learning rate and batch size faster
3. learning rate: 0.0005, batch size:2, epoch: 500
Validation의 Error: 2.2832
4. learning rate: 0.0005, batch size:1, epoch: 500
Validation의 Error: 2.3285
* When I set batch size to 4, the error was quite large so I decreased to 3,2,1. 2 seemd to be the optimal value
5. learning rate: 0.001, batch size:2, epoch: 500
Validation의 Error: 2.2676
6. learning rate: 0.002, batch size:2, epoch: 500
Validation의: 2.3276
* I tried increasing the learning rate, but found 0.001 to be optimal.
7. learning rate: 0.001, batch size:2, epoch: 1000
Validation의 Error: 2.1714
8. learning rate: 0.001, batch size:2, epoch: 1500
Validation의 Error: 2.1643
9. learning rate: 0.001, batch size:2, epoch: 2000
Validation의 Error: 2.1501
* I found that the bigger the epoch, the better the feature. But due to time restrictions, I only experimented until 2000.
 
∴ learning rate=0.001, batch size=2, epoch=2000

# Result: The converted test data(input, output, and target)
Source(A, B, C): each VCC2SF1, VCC2SM1, VCC2SM2
Target(My Voice): VCC2SF2
Output:
*	From A to my voice: SF1_to_SF2
*	From B to my voice: SM1_to_SF2
*	From C to my voice: SM2_to_SF2
