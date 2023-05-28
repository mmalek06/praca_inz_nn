<h3>Notebooks in the order they should be read</h3>

1. eda - basic EDA code to get a grasp of the tabular data contents
2. resizing - the name says it all
3. categorize_images - code for transforming the original downloaded dataset into something tf-feedable
4. box_lesions - an algorithm that given the mask images is capable of generating the bounding box images; this will be
   helpful for the ROI detection net
5. inception_resnet_v2_self_trained_on_200x150 - first attempts to train a preconfigured NN on suboptimal input
6. inception_resnet_v2_self_trained_on_200x150_tabular - more input - added tabular data input
7. inception_resnet_v2_self_trained_on_200x150_augmentation - training a network on an augmented dataset
8. roi_detection - a demonstration of ROI (region of interest) detection network used as a helper network that will 
   be able to detect regions actually containing lesions; its result will be used in the next networks
9. inception_resnet_v2_self_trained_on_299x299 - first attempts to train a preconfigured NN on optimal input
10. (unknown point number at this time, I've given it an 8 for now) - augmentation of the images with various artifacts - shadows, dirty lens effects, gaussian noise; training
    denoising autoencoders for noise removal, then a noise classifier network that will be able to pick the correct
    DAE; that's all for pre-processing; the results will be passed to the main cancer classifier