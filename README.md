# **Behavioral Cloning**
---
[//]: # (Image References)

[image1]: ./md_imgs/cameras.png "Model Visualization"
[image2]: ./md_imgs/cameras_flip.png "Grayscaling"
[image3]: ./md_imgs/cameras_noSteerCrct.png "Recovery Image"
[image4]: ./md_imgs/crop.png "Recovery Image"
[image5]: ./md_imgs/hist1.png "Recovery Image"
[image6]: ./md_imgs/hist2.png "Normal Image"
[image7]: ./md_imgs/recovery.png "Flipped Image"
[image8]: ./md_imgs/yuv.png "Flipped Image"



The following report discusses my solution to project 3 of the the Udacity Self
Driving Car Nanodegree. The goals of the project were to:
* Use a provided simulator to capture data of appropriate/desired driving.
  * The simulator contains a "Training Mode" to collect data and an "Autonomous Mode" to test developed models.
  * There are 2 tracks - one simple and one difficult. Developing a model to traverse the simple track is required.
  * The data consists of screenshots captured at three different points on the car - center-mounted, left-mounted, and right-mounted - as well as the steering angle at the time each image was captured.
* Implement and train a convolution neural network to predict steering angle from collected data.
* Test that the model successfully drives around the test track(s) without leaving the road.


I will be touching on the important aspects of the [rubric](https://review.udacity.com/#!/rubrics/432/view) in this writeup; other aspects (such as code quality) don't warrant discussion here.

---
## 1. Setup

Instructions regarding various files and for running this project can be found [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3).

The simulator used in this project can be downloaded [here for Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip) or [here for Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

My submission for this project includes the following files:
* README.md -> this document
* model.py -> script to create and train the model
* drive.py -> script to drive the car in autonomous mode
* model.h5 -> trained convolution neural network model
* track1.mp4 -> video of the car successfully traversing track 1
* track2.mp4 -> video of the car successfully traversing track 2

Note: I didn't include any data in my submission, and there is a lot of it and it was not required/requested.

**NOTE: Dealing with Quirky Issues Running the Simulation:**
* I developed the model across Mac (2013 MB Pro w/ integrated graphics) and Windows PCs (Windows 7 w/ an NVIDIA K2100M graphics card - 2GB RAM) and had some issues running the model across platforms. Below are some notes per platform and a recommendation for simulation parameters when attempting to run my model.
  * Windows PC:
    * The simulation simply would not run on Fantastic mode, even at 640x480. It would be very choppy, the car would veer off the road (because the choppiness of the simulation would wind up the integrator of the vehicle speed PI controller and speed would fluctuate drastically), python would freeze, and I'd get an OOM error.
    * I had to reduce the gains to Kp=0.08 and Ki=0.0002 to get the simulation to run.
    * I believe I had issues with the simulator if it was running a monitor that wasn't directly connected to the graphics card (i.e.: was connected through through a dock)
  * Mac:
    * My Mac generally struggled to run the simulation, but could run a (choppy) 640x480 version on the Beautiful setting at 10 mph and successfully complete both runs. Increasing speed led to some issues, as the simulation would lag significantly, which I'm assuming led to reduced performance of the NN.
    * The same gain settings required on the Windows PC wouldn't work on my Mac, for some reason. The car would struggle to climb hills, and the lower integral gain would cause wind up on steep hills that would cause the car to surge forward and go off the road after it crested these hills. I successfully ran the simulator with gains of Kp=0.1 and Ki=0.002 (interesting that the integral gain was an order of magnitude different...)
  * On both:
    * The cars often wouldn't move the first time running the simulator, so I'd have to press 'Esc' and run them a second time.

As a result of these issues, I suggest the following settings:
* My defaults (set for Track 1 completion on both of my PCs):
  * Speed = 15 mph (if car is unstable, reduce speed)
  * Kp=0.08, Ki=0.0002
  * Beautiful setting at 640x480
* Tweaks, if required:
  * No higher than Beautiful setting, at 640x480 (you may be able to run different settings if you have a more capable graphics card that doesn't cause the simulation to lag; if your computer is less capable, try the Good setting).
  * One of the following 2 gain settings (set in the drive.py file):
    * Kp=0.1, Ki=0.002 (required for Track 2 on the Mac); or Kp=0.08, Ki=0.0002. (I wish I had more time to diagnose this and provide better direction, but unfortunately, I don't)
  * Max speeds:
    * Track 1: 15 mph
    * Track 2: 10 mph (I was able to complete at 15 mph on my Windows PC, but not the Mac.)


## 2. Network Architecture

This portion of the project was relatively straight-forward, as the Udacity classroom sessions referenced a proven model architecture developed and tested by [NVIDIA](https://arxiv.org/pdf/1604.07316v1.pdf). While some specific details of the architecture were omitted from the information provided by NVIDIA (ex: dropout rates), I chose to use the NVIDIA architecture from the beginning, and felt confident that the network architecture would not be an issue.

Below is a description of my architecture:

* Cropping - 160x320x3 to 66x260x3
  * The output of this cropping layer (and thus the input to the model) differed from that used by NVIDIA, in that the images I used were wider -> 260 vs 200. I could have further cropped the images, or resized them, but based on the amount of data I was using and the size of the network, I didn't feel it was necessary. With a larger training set, or had it taken longer to train this model, I may have opted to reduce the size of the images.
* Normalization - 66x20x3 to 66x20x3
* Convolution - 5x5 convolution; 2x2 stride; depth of 24; relu activation; output 31x128x24
* Dropout - selectable dropout percentage
* Convolution - 5x5 convolution; 2x2 stride; depth of 36; relu activation; output 14x62x36
* Dropout - selectable dropout percentage
* Convolution - 5x5 convolution; 2x2 stride;  depth of 48; relu activation; output 5x29x48
* Dropout - selectable dropout percentage
* Convolution - 3x3 convolution; 1x1 stride; depth of 64; relu activation; output 3x27x64
* Dropout - selectable dropout percentage
* Convolution - 3x3 convolution; 1x1 stride; depth of 64; relu activation; output 1x25x64
* Dropout - selectable dropout percentage
* Flatten - output 1x1600
* Fully connected - depth 1000
* Dropout - selectable dropout percentage
* Fully connected - depth 100
* Dropout - selectable dropout percentage
* Fully connected - depth 50
* Dropout - selectable dropout percentage
* Fully connected - depth 1 -> output layer

I used mean square error as the cost function, with an adam optimizer, and trained for 3 epochs with a batch size of 512. The use of the adam optimizer meant that I really didn't have to do much in the way of parameter tuning.

Training was performed on an AWS GPU instance (g3.4xlarge - NVIDIA Tesla M60 w/ 8GB of RAM at time of writing).

### 2.1 Notes re: Dropout
To prevent overfitting, I added dropout layers after each convolution/connected layer in the architecture, with the ability to select from a pre-determined set of dropout rates (0%, 30%, or 50%). In general, the larger the number of parameters in a given layer, the higher the selected dropout rate. For smaller layers, such as the last convolutional or connected layers, I didn't perform any dropout.


## 3. Data Collection and Processing

Data collection played by far the largest role in the success (or failure) of the model. Whereas in previous projects thus far in the nanodegree data collection was somewhat taken for granted (i.e.: all data was provided for us!) this project made it readily apparent that it really is all about the data when it comes to machine learning.

The following subsections summarize general approaches taken and findings gleaned as I worked on this project, with chronology not considered. The subsequent section (**Data Collection/Testing Walkthrough**) provides more of a step by step summary of model performance with data collection over time.

###3.1 Camera Views
From the beginning, I chose to use all 3 camera views in all of my training. I started with a steering correction factor of +/- 0.2, as suggested in the Udacity classroom courses, but over time - based on some trig, a review of captured images with the angle plotted, and (mostly) experimentation in the simulator - I ended up using a steering correction of 0.25. Regarding the performance of the car in sharp turns, I found that with lower steering corrections, the car wouldn't properly make the sharpest turns on Track 2, but would hit the barriers. Corrections higher than 0.25 appeared to reduce performance and make the car "jittery".

Below are images of a set of camera views (left, center, and right cameras) before and after steering correction. The blue line is an approximation of the steering angle for each image.

Before steering correction:
![alt text][image3]

After steering correction (in the left and right images only, obviously):
![alt text][image1]

###3.2 Image Flipping
Each of the center, left, and right images were flipped horizontally to supplement the number of training samples, and to ensure there was no left/right bias in the training set (as suggested in the classroom sessions). This was done in the training pipeline for the sake of not generating tens of thousands of additional images to be stored/transferred. If I was looking to optimize the speed of training, I would have considered flipping the images and storing them separately to remove this process from the pipeline.

The image below is a flipped version of first set of images in the subsection above:
![alt text][image2]

###3.3 Cropping
While I started off with the cropping suggested in the classroom session (50 pixels off the top, 20 pixels off the bottom), based on experimentation, I ended up cropping off a larger portion of the top of the image -> 74 pixels off the top, 20 pixels off the bottom. This seemed to improve performance overall, and, conveniently, also served to make my images match the height of the images used in NVIDIA's original pipeline. As mentioned above, my images have a width of 260 pixels (ie: 30 pixels cropped off of each side); if I were looking to optimize training time, I would have chosen to resize the image down to a width of 200, as used by NVIDIA.

Below is an example of a cropped version of the example image set shown thus far:
![alt text][image4]

This cropping provides fairly limited field of view for, and while this helped improve performance, I imagine such aggressive cropping would not be acceptable at high speeds. Note though, that both lines are still visible in the center camera image (left side). In the future, it might be worthwhile spending more time to determine an optimal cropping value that balances high-speed performance.

###3.4 Removing Straight Driving Bias
The authors of the NVIDIA paper mentioned that they skewed their data towards turns, rather than straight lines. A histogram of the steering angles for collected data, shown below, indicates that straight driving was heavily over-represented in the datasets.

![alt text][image5]

The three largest bins in the histogram are associated with straight driving, since each capture of the vehicle going straight had center left and right camera views.

I took two approaches to removing "straight driving bias":
  1. I set up a filter to keep on a proportion of all images within a given steering angle threshold (I settled on +/- 0.04).
  2. I eventually began selectively recording only portions of the track in which the car was turning, leaving out the straight sections. I chose to do this because I felt that there may be some cases where straight driving was appropriate, and it would be incorrect to remove such cases from the data. For example, if the car were close to the left edge of the road during a left turn, it may be appropriate to drive straight in order to have the car return to the center of the road.

In the end, I used some data that wasn't collected with the method mentioned in point2 above, and thus, I still used a filter.

Below is an example of the distribution after filtering. While all distributions will be similar, I did use a random number generator to determine which images to filter out, so the results will be slightly different each time.

![alt text][image6]

###3.5 YUV Conversion
All images were converted to YUV before being passed to the network, as suggested by the NVIDIA authors. This was done within the pipeline itself. Once again, had pipeline speed been an issue, I would have considered doing this outside of the pipeline and saving off the results.

The image below illustrates a YUV conversion, though it is a YUV image plotted in RGB space, which may not be a true representation, but does look cool.

![alt text][image8]


###3.6 Generator Implementation
Eventually, I acquired so much data that it couldn't all be loaded into memory at one time (even with 32GB of RAM). At this point, I implemented a generator; since each image is being flipped in the training pipeline, care was taken to ensure that the batch size being generated matched that being passed to the generator function (see code comments).


##4. Data Collection/Testing Walkthrough
Note that data was not included in the submission, nor was all data I collected used in the final model. This section simply summarizes my progression through all data collection and training.

Note that this is a general and broad summary of my progress. I definitely experimented with some of the items noted in the section above as I progressed through these steps, but it's difficult to summarize all of this coherently in one place.

I set up my code to allow for the selection of any number of individual datasets, and built up a library of different data sets, with the intent of mixing and matching various sets to get the best combination. The following points summarize the data that I collected; in general, I started with a small amount of data, and built up the library, trying to assess the effects of the different data. Datasets with a strikethrough did not end up getting used to generate the final model (rationale is provided below).


1. **General Driving Data**
  1. Train1 - Track 1. Centered in the lane.
      * **Summary:** The car didn't run great with this data (which was expected, since it was a small dataset); it often turned off the road. I next recorded some recoveries.
2. **Recovery Data**
      1. ~~Train2~~ - Track 1; forwards only. Recoveries from the sides of the road to the center of the road, starting parallel to the side edges and recovering to the center.
      2. ~~Train3~~ - Track 1; forwards only. Recoveries starting with the car angled towards the edge of the road, and recovering to the center of the road and parallel to the edges
      * **Summary:** The thought behind this recovery data was that, should the car end up too close to the side of the road, or veering towards the side of the road (which it was doing often at this point), this data would help the network correct back to the center. It failed miserably, and would either drive alongside the side of the road, or veer off. The Train3 dataset seemed to induce the worst behavior, and I didn't use it again.

        The image below shows an example recovery image:

![alt text][image7]

3. **Generalization**
    1. Train 4 - Track 2; forwards only. Data was recorded on Track 2 to improve model generalization.
    * **Summary:** At this point, the model was driving decently, but still struggled in some ares; it veered off the road in some places, or would drive close to the side of the road and not correct.
4. **Problem Areas + Additional Data**
    1. ~~Train5~~ - Track 1. Data was collected in several problem areas to try to improve performance (yes, I realized that this would probably result in overfitting on specific track areas, but I was still experimenting at this stage).
    2. Train6 - Track 1; forwards; turns only. Considering I was having issues with the car driving along the sides of the road, I collected more data with the car in the center of the road.
    * **Summary:** This data didn't seem to help all that much. The model was performing well, but would still get "stuck" along the side of the road in some places. At this point, I went through some of my data image by image (with steering angle plotted on the image) and realized that my Train2 dataset was flawed!! In that dataset, I had started recording with the car on the side of the road *and* with my wheels parallel to the edge of the road, after which I would turn the wheels to move the car back to the center of the road. This meant that I had a fair number of images in this dataset that had the car along the side of the road with the wheels pointed straight. This was the source of my issues.
  5. **Show Me The Data**
      * At this point, I felt like I'd worked through a few issues and discovered a flaw in my data. I felt like the next step was to collect a significant amount of really solid data.
      1. Train7 - Track 1; better recoveries.
      2. Train8 - Track 2; forwards, only turns.
      3. Train9 - Track 2; backwards, only turns.
      4. ~~Train10~~ - Track 2; recoveries. Not a lot of places to do this on Track 2...
      5. Train11 - Track 1; forwards, only turns.
      6. Train12 - Track 1; backwards, only turns.
      * **Summary:** I took extreme care when collecting this data to ensure that ***every frame*** of data would point the network in the right direction. After training the model with this data, plus the good datasets I had collected previously, the model performed very well. It navigated the first track easily, and almost completed the challenge track. When I removed the track 2 recovery data (Train10), the model navigated both tracks. SUCCESS!!

Having now succeeded, I retraced some of my steps and tried a few different variations of datasets, along with some variations of steering angle correction and straight bias filtering, but settled on the values/parameters noted above using the following datasets: Train1, Train4, Train6, Train7, Train8, Train9, Train11, Train12. When using all three cameras, flipping each image, and after filtering out a portion of straight driving, there were around 88,000 images.


## 5. Results
The vehicle was able to navigate both tracks. The track1.mp4 and track2.mp4 videos the vehicle completing Tracks 1 and 2  (albeit with some "wobbliness" on straight sections of Track 1).

One thing to note is that on Track 1, as speed increased, vehicle stability decreased. By this, I mean that, at the start of the course, the vehicle would start to wobble; if the speed was low (ex: less than 20 mph), the oscillations would damp out, but as speed increased (say 25 mph), it would fail and go off the track. I believe this may be because the recovery data that I collected was fairly aggressive; it was recorded at a very low rate of speed, and with relatively high steering angles to bring the vehicle back to the center of the road. Perhaps better recovery data - where recoveries were not so severe - would improve performance. Alas, it's time to move on to other projects!


## 6. Thoughts
It was incredibly satisfying to complete this project. Training a neural network to classify signs is cool, but training one to drive a car is awesome. More than anything else, this project illustrated that a network is only as good as the data that you feed it. I went into the project feeling quite confident about the outcome knowing that I could use a tried-and-tested architecture - the NVIDIA architecture referenced throughout - but only achieved success when I became very meticulous in my data collection. In retrospect, this seems incredibly obvious, but having been handed complete datasets in the previous project I can't say that I truly appreciated this fact until I was forced to collect data myself.
