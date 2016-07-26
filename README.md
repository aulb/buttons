# Short
*I wanted to count and document my personal collections of buttons. However, manually taking individual button, taking its picture, and cropping them would be a very menial work. What better time to apply the knowledge I learned from school to help me with this project.

# Dependencies
I decided to learn and use opencv for most of the image related operation. Numpy and scikit are used whenever else opencv is not applicable. 

# Process
Before counting, I made some simplification to the project:
- Picture dimension is assumed to be 3120 by 4160 pixels
- Containing only circular buttons
- Buttons roughly sorted by size
These simplifications allowed me to obtain better results faster. In my case the pictures were taken from a 1+ phone camera setup at 30cm above the table. 

I decided to use opencv's hough circles implementation to preliminary find the circles. Button images undergoes a simple preprocessing using a median filter to get rid of speckle noises and blurring filter for smoothness. Multiple parameters for hough circles (namely param1 and param2) are also used to detect circles. For parameters concerning sizes such as minimum and maximum circle radius, it is determined based on observation and quick empirical runs. These parameter sets detect different circles. Lastly, the detected circles are grouped according to location and pruned. 

The results were great, with less than 15 out of 350 being bad crops. Although in the smallest circle group, some of the buttons were not detected at all. Here are the results: http://imgur.com/a/ttbBT/all

# Mistakes
Cannot transform the image to binary because Hough circles already performs canny detection which changes the grayscale image to binary. From learningopencv: "The cvHoughCircles() function will internally (automatically) call cvSobel()^* for you so you can provide a more general grayscale image". I made the mistake of providing binary image from the start thus ruining the performance of the circle detection.

# Folder Directory
- Buttons
	- saved			: contains all the results
	- sorted		: contains all of the source image 
- main.py 			: main script running the cropping process
- ImageCrop.py 		: main class handling all of the cropping process
- ImageCrop.json 	: config file, handles parameters and the like

# Future Considerations
Here are some of the lists for the future:
- Rectangular button or any shaped button detection. This would probably mean having to dabble in contour detection.
- Button statistics to generate and visualize button data. This would required some kind of feature extraction, dimensionality reduction methods.
- Lighting consideration. Better photo quality to produce sharper looking buttons. Due to the lack of space and equipment, the lighting condition are not optimal. 
- Better documentations. No explanation needed here.
- Exploring different color spaces such as HSV to look for information.
- More modular image dimension and automatic detections.

# Button Statistics
These are some of the features that I considered that are worthy of extracting:
- Average RGB color, monotonicity of color
- Gradient scorings
- Presence of text
- Manual labels*

Buttons are often a way of representing a cause or displaying pride. Manual intervention would be needed in this case to label the data. The buttons would need to be divided into categories based on what type of button it is. Example categories could be educational, school related, slightly offensive, graphic/cartoon.

Another way is to simply resize the button and learn the features.

# License
I don't think I'll ever need this. Just incase
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.