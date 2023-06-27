<h3>Table of contents</h3>
<br />
Different phases of work require different code, so I've splitted them into folders:

<h4>Image manipulation</h4>

1. image_manipulation/resizing - used for resizing the images coming from the HAM10000 dataset to a 150x200 size;
2. image_manipulation/resize_extended_dataset - same as the above but for the extended dataset;
3. image_manipulation/augmentation - I thought some augmentation upfront could be useful, but during the 
   experimentation it turned out what keras can do on the fly is enough. I'm leaving this notebook in case
   further work reveals that it's actually required;
4. image_manipulation/box_lesions + box_augmented_lesions - both notebooks are used to draw boxes around lesions - 
   only for debugging purposes;
5. image_manipulation/move_and_split - this one is for moving images around and splitting them into
   training and validation sets. It's also only for the augmented images, so not really used;
6. image_manipulation/categorize_images + categorize_extended_images - since the original datasets were not splitted
   into categories, using the csv files attached to the dataset allowed me to properly label the images, so that
   they are easily feedable to keras machinery.

<h4>ROI</h4>
