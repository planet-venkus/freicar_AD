#Training
For training complete the file train_segmentation.py. In order to successfully train the network you need to implement the training loop and implement the getItem function in the dataloader (dataset_helper/freicar_segreg_dataloader.py).

As soon you run the dataloader the first time it will start to download the training/evaluation data automatically.

Similar to the first exercise we provide a file named "run_freicar_dataloader.py" that you can leverage to test your dataloader.

If you like to monitor your training with an additional visualization framework we ask you to add the necessary code for visdom/wandb/tensorboard yourself.

#Segmentation/Regression Visualization
To ease the visualization of segmentation results we provide you the file "dataset_helper/color_coder.py" that provides a class, which converts network outputs to a color coded RGB image.
For lane regression visualization see the function "visJetColorCoding" in the training script.

#Evaluation
As stated in the exercise-pdf you need to implement the IoU calculation and evaluate the IoU scores on the evaluation and training set respectively every N epochs during training.

#Bird's Eye View
The file "birdsEyeT.py" provides the class "birdseyeTransformer" that can transform a image to the bird's eye view using inverse perspective mapping. An example of how to use it is provided with the file "birdseye_demo.py".
Note that the input image size must be 640x360. If you modify, pad or shift the image this function won't work.

#The below information is also given in the submitted pdf report.

##### Converting Model to Torchscript #######

➢ Run the command `train_segmentation.py --convert_torch <path_to_model_weights>`
This will create scripted_model.pt file in the current folder. We will load this file in our
cpp file.Copy this scripted_model.pt inside segmentation_publisher/src/ folder. This is
because we have hardcoded the path to this torchscript inside Imageconverter.cpp

##### Running the Ros node for publishing lane regression and segmentation ######

➢ We have given the absolute path for freicar_homography.yaml inside birdeye.cpp.
➢ Please place the libtorch folder (We downloaded and tested it for libtorch 1.7.1
CUDA 11) inside segmentation_publisher/ folder. The absolute path for libtorch is
given in CMakeLists and the build will fail if libtorch is not found.
➢ We are subscribing to the topic `/freicar_1/sim/camera/rgb/front/image` inside ros
node, passing it to our model and publishing the results for segmentation and
regression of normal image to `/image_converter/seg_output` and
`image_converter/lreg_output` respectively. We are publishing the birds eye
regression to `/visualization_msgs/MarkerArray`
➢ Run catkin build to build the ros node.
➢ Run rosrun segmentation_publisher image_converter_node to run the ros node.
