<h3>Notebooks in the order they should be read</h3>

1. eda - basic EDA code to get a grasp of the tabular data contents
2. categorize_images - code for transforming the original downloaded dataset into something tf-feedable
3. inception_resnet_v2_self_trained_on_200x150 - first attempts to train a preconfigured NN on suboptimal input
4. inception_resnet_v2_self_trained_on_200x150_tabular - better input - added tabular data input
5. roi_detection - a demonstration of ROI (region of interest) detection network used as a helper network that will 
   be able to detect regions actually containing lesions; it's result will be used in the next networks
6. inception_resnet_v2_self_trained_on_299x299 - first attempts to train a preconfigured NN on optimal input
7. inception_resnet_v2_self_trained_on_299x299_tabular - better input and tabular data usage
8. (unknown point number at this time, I've given it an 8 for now) - augmentation of the images with various artifacts - shadows, dirty lens effects, gaussian noise; training
   denoising autoencoders for noise removal, then a noise classifier network that will be able to pick the correct
   DAE; that's all for pre-processing; the results will be passed to the main cancer classifier