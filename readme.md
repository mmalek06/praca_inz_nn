<h3>The attack plan is as follows:</h3>

1. Train a NN on the base, non-augmented images to obtain a baseline score
2. Augment images as described in the augmentation notebook. For debugging purposes it's easier to do it like this than on the fly - during training.
3. See how previously tried out approaches do with the augmented images.
4. Train a denoising autoencoder to remove artifacts from the images. This step is actually very involved:
    - first train a separate DAE for each noise category
    - then train a classifier capable of recognizing noise category
    - create a decision function that would be able to choose a fitting DAE
5. Use the above in ML pipeline of the models trained in the first two steps.
6. Build the final processing pipeline containing the best performing parts.