# The Image Separation Problem
CIFAR-10 is a widely used dataset in machine learning and computer vision. It consists of 60,000 color images with a resolution of 32x32 pixels, 
divided into 10 different classes, with 6,000 images per class. The classes represent common objects such as airplanes, cars, birds, cats, deer, 
dogs, frogs, horses, ships, and trucks.

The dataset is pre-split into a training set (50,000 images) and a test set (10,000 images). It is commonly used to train and evaluate image 
classification algorithms and serves as a standard benchmark for developing and comparing deep learning models.

## Definition of the Predictive Model
For the proposed problem, a **CNN** (convolutional neural network) was chosen.
A CNN model was selected because their use is typical for image recognition problems (Image Classification & Detection).

The creation of the model was inspired by the architecture of one of the most famous CNNs, _AlexNet_ (use of _convolutional layers_ 
followed by _normalizations_ and _pooling_, _dropout_ after _dense layers_, etc.), while adding features like _residual computation_.

Interesting aspects implemented in the model:
- Various **convolutional blocks** with **periodic normalizations** (to counteract overfitting);
- Use of the **residual computation technique** (in this version, compared to a previous model without it, there is an increase in accuracy
  and a reduction in loss of at least 10%);
- Use of **GlobalAveragePooling()** instead of **Flatten()** for the transition from a multidimensional to a one-dimensional vector;
- **Dropout Layers**: a regularization technique used in neural networks to reduce overfitting (preventing the model from over-adapting to
  training data and losing the ability to generalize to new data);
- Use of the **Adam optimizer**.

## Final Considerations and Possible Improvements
As observed from the training results, the model tends to achieve slightly higher _accuracy_ on the second output compared to the first 
(*output_1_accuracy*: 0.8439, *output_2_accuracy*: 0.9033) with an overall _loss_ of approximately 0.72. Certainly, further training phases 
could yield a **slight improvement** of a few percentage points.

In the later training stages, the improvement in loss becomes increasingly marginal.
Consequently, it is evident that a **significant performance increase** beyond the testing phase (e.g., by 10%, thus moving from about 77% to 
approximately 90% _mean accuracy_ like the best models for the proposed problem) can only be achieved by **modifying the model itself** and 
adopting some data-related enhancements:

- **Data Augmentation**: Increase dataset variety by introducing transformations such as rotations, flips, and scaling;
- **Adding additional layers**: Build a deeper architecture;
- **Model tuning**: Experiment with other optimizers, optimize batch size, or adjust dropout rates;
- **Better regularization**: Apply additional regularization techniques such as L1/L2 or increase Dropout;
- **Validation set**: Use a validation set to monitor overfitting during training (e.g., reserving 20% of the training data).

Regarding the _Standard Deviation_, it is relatively low, indicating that the model is **stable** and **consistent**.
