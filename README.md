# LOCATE (Latent Of miCrobiome And meTabolites rElations)
This code is attached to the paper "Microbiome-metabolome interactions predict host phenotype". 
We propuse a machine learning (ML) tool based on latent representation which predicts the log normalized metabolites composition 
based on the log normalized microbiome composition. 
LOCATE has a higher overall accuracy than all current state-of-the-art predictors in both 16S and shotgun sequencing. 
We propose an intermediate representation between the microbiome and the metabolite 
concentrations and show that this representation can be used to predict the host phenotype better than either the microbiome or the metabolome.

## How to apply LOCATE
LOCATE's code is avaliable at this github as well as a [pypi](https://pages.github.com/).

### LOCATE's GitHub
#### LOCATE_model
This code consists of LOCATE's model class with the following functions:
1. **init** - with all model parametes as will be further explained.

2. **find_transformer** - finds the approximated A* to relate the intermediate representation of the microbiome to the training metabolites 
   (for more details see paper's Methods).
   
3. **forward** - with LOCATE forward function.

4. **configure_optimizers** - controls the model's optimizers.

5. **loss_g** - representation network loss function.

6. **training_step** - with LOCATE training step.

7. **validation_step** - with LOCATE validation step.

8. **backward** - with the special backwards of LOCATE.

#### LOCATE_functions
Consists of two functions:

1. **LOCATE_training**

    **Input**

    1. **X_train:** Log normalized and with column z-score dataframe (for more details about normalization, see paper Methods)
       of training microbiome features (dataframe).
       
    2. **Y_train:** Log normalized and with column z-score dataframe (for more details about normalization, see paper Methods)
       of training metabolites fearures (datframe).
       
    3. **X_val:** Log normalized and with column z-score dataframe (for more details about normalization, see paper Methods) 
        of validation microbiome features (dataframe).
        
    4. **Y_val:** Log normalized and with column z-score dataframe (for more details about normalization, see paper Methods)  
       of validation metabolites fearures (datframe) - for early stopping.
       
    5. **representation_size:** Dimension of the intermediate representation (int).
    
    6. **weight_decay_rep:** L2 regularization coefficient of the representation network (float).
    
    7. **weight_decay_dis:** L2 regularization of the optional discriminator, is not used in the paper (float).
    
    8. **lr_rep:** Leaning rate of the representation network (float).
    
    9. **lr_dis:** Learning rate of the optional discriminator network, is not used in the paper (float).
    
    10. **rep_coef:** Weight of the loss upgrades of the representation network, is set to 1, when no discriminator is used (float).
    
    11. **dis_coef:** Weight of the loss upgrades of the discriminator network, is set to 0, when no discriminator is used (float).
    
    12. **activation_rep:** Activation function of the representation network, one of: {relu,elu,tanh}.
    
    13. **activation_dis:** Activation function of the discriminator network, one of: {relu,elu,tanh}.
    
    14. **neurons:** Number of neurons in the first layer of the representation network (int).
    
    15. **neurons2:** Number of neurons in the second layer of the representation network (int).
    
    16. **dropout:** Dropout parameter (float).
    
    **Output**
    
    Returns a trained model.
    
    
    2. **LOCATE_predict**
    
    **Input**
    
      1. **model:** Trained model (the output of LOCATE_training).
      2. **X_val:** Log normalized and with column z-score dataframe (for more details about normalization, see paper Methods)
          of validation microbiome features (dataframe).
      3. **metab_names:** List of metabolites names.
      
      **Output**
     
      Returns Z_val = intermediate representation, metabolites predictions dataframe.
      
      
    3. **usage_example**
    
     Example of using the code on randomized data with its defaltive parameters:
     
     ```
     model = LOCATE_training(X_train, Y_train, X_val, Y_val)
    Z_val, n_pred = LOCATE_predict(model, X_val, Y_val.columns)
    ```
    
    
### LOCATE's pypi

This package contains 2 different elements:

  1. LOCATE training 
  
  2. LOCATE predict
  
  #### Installing LOCATE
  
  ```pip install LOCATE-model```
  
  #### Using LOCATE
  ```
  import LOCATE
  
  model = LOCATE.LOCATE_training(X_train, Y_train, X_val, Y_val)
  Z_val, n_pred = LOCATE.LOCATE_predict(model, X_val, Y_val.columns)
  ```
  
  
## Contributors

Oshrit Shtossel


## Contact

If you want to contact me you can reach me at oshritvig@gmail.com
  

     
      
      
    
    
