# A Deep Convolution Neural Network, AI Based Approach - To Detect COVID19 From CT Scan Images

1. Deep learning has dramatically increased the state of the art in Speech, Vision and many other areas.  
2. COVID-19 is a severe global problem that is haunting India with its ever since infection and death rate, and AI can play a significant role in preventing losses by monitoring      and detecting infected persons in early-stage.                           
3. The aim of our project comprises of the following aspect                             
4. High speed, accurate and fully automated method to detect COVID-19 from the patient's CT scan images.   
 
# Working
1. It is based on Supervised Machine Learning, where the model is trained with the pre-collected standard dataset. So, Dataset preparation paves a major contribution to the            model accuracy along with the implemented architecture of the model.                       
2. After the Dataset collection, Deep Convolution Neural Network architecture will be implemented to classify the Covid-19 and Non-Covid-19 CT scan images.             
3. CNN architecture is invariant to rotation, scale and contrast of the image, which may reduce the accuracy of test data. Data Augmentation technique will solve the problem by        generating variant images of the training data.                         
4. This CNN primarily extracts the features of the image by adjusting the filter weights through the Back-propagation technique.                         
5. Then, this dataset will be applied to a fully connected neural network layer which gets fully trained through Categorical-Cross entropy loss function, Adam optimizer, Softmax      activation function and Back-propagation technique.                         
6. All the proposed working procedure will be conducted on Google Collaboratory Python Environment using Machine Learning Framework Keras and Tensorflow developed by Google.   


# Dataset for the training is collected from the Kaggle, Machine Learning and Data Science Community.     
1. Dataset is available at ----> https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset         
2. These data have been collected from real patients in hospitals from Sao Paulo, Brazil. The aim of this dataset is to encourage the research and development of artificial             intelligent methods which are able to identify if a person is infected by SARS-CoV-2 through the analysis of his/her CT scans.                         
## Please cite:                                                              
1. Soares, Eduardo, Angelov, Plamen, Biaso, Sarah, Higa Froes, Michele, and Kanda Abe, Daniel. "SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-       CoV-2 identification." medRxiv (2020). doi: https://doi.org/10.1101/2020.04.24.20078584.                                                                                     
2. Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks, 130, 185-194.                                                                    

# Dataset Preparation
1. After downloading the dataset from the above source or platform                                         
2. Dataset folder has 2 subfolder, one folder containing Covid-19 CT Scan images which represents one of the class in our classification and the another folder contains Non           Covid CT Scan images which is the another class in our classification.                                                                 
3. Dataset has 1252 CT scans that are positive for  infection (COVID-19) and 1230 CT scans for patients non-infected by SARS-CoV-2, 2482 CT scans in total.                       
4. Dataset will be constructed into two sections.                                                        
5. The first section includes a training dataset, exclusively used for training purpose and the second section is validating or test dataset, to evaluate the performance of the       model.  

# Training the Model
1. Model is trained using CNN and transfer learning models sucha as ResNet, VGG, DenseNet, Exception, MobileNet and Inception.   
2. Refer my collab notebooks for the code.                 
3. All my trained models and datasets are availabe at ----->  https://drive.google.com/drive/folders/1EmdLVEU3vDUmK9CQwBKHeEN2Zv1Ewil1?usp=sharing
 
# Performance Analysis 
1. After training the model, accuracy of each model is compared and DenseNet Deep-Neural Network model accuracy is quite high compared to the other models with respect to this 
   dataset and problem classification.                    
2. A test dataset is created by collecting the ct scan images from other platforms, which was not used in training.                        
3. VGG model is picked and used to predict the test images from test dataset.                      
4. Grad Cam Algorithm is applied to the model                 
5. Grad-CAM is a popular technique for visualizing where a convolutional neural network model is looking.                 
6. Grad-CAM is class-specific, meaning it can produce a separate visualization for every class present in the image.
7. Refer my Performance Analysis collab notebook (.pynb file) for the code.             
 
 


