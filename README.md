# MitoA analyser Documentation

---

### The MitoA analyser is an easy-to-use GUI-based morphological quantification tool that enables biologists to analyse and compare different samples of segmented images. 

To download the MitoA tool for Windows please visit https://zenodo.org/record/3557720#.XeeS4NV7lhG 

---

The MitoA tool is intended to be used after successful segmentation with the MitoS tool but can also be applied to any dataset consisting of 8-bit raw images and their masks. 

## Running the MitoA analyser

Do not close the terminal while running the MitoA tool. 

## Get Measurements 

Select this option to generate Excel tables containing summarized shape descriptor measurements of all images. Once the tables have been generated you may proceed with Analyse

* Enter name of measurements table 
* Select directory in which table should be saved
* Select directory containing 8-bit raw images
* Select directory containing segmented images 

The Excel table consists of two sheets, one named ShapeDescriptor, the other called BranchAnalysis. The values below are measured for each object and the average, median, standard deviation, standard error, minimum, 
maximum and number of objects per image are collected. 

### Shape Descriptors 

* Area: area of region
* Minor Axis Length: minor axis of the ellipse that is fitted around the region
* Major Axis Length: major axis of the ellipse that is fitted around the region
* Eccentricity: ratio of focal distance over the major axis length (0 is perfect circle) 
* Perimeter: perimeter of region
* Solidity: ratio of pixels in region to pixels in the convex hull 
* Mean Intensity: average intensity in region
* Max Intensity: maximum intensity in region
* Min Intensity: minimum intensity in region

### Branch Analysis

* Number of branches: number of branches in tubular mitochondria 
* Branch length: length of each branch found per region
* Total branch length: total length of all branches per region
* Curvature index: ratio between branch length and the euclidean distance of the start and end point of branch


## Analyse 

To gain further insights from the tables generated with Get Measurements, Analyse allows to create tables with statistical information.

### 2 samples

If you want to compare the morphology of two samples then use this section. 

#### Morphological comparison - Generate table 

Works only with samples that contain at least 8 images or more. 

Create a table containing the following information:

* The normality test p-value for each descriptor and sample 
* The Hypothesis test used to compare the descriptors of sample 1 and 2
* The p-value of the hypothesis test 
* The effect size 
* Single values for each descriptor are assigned a separate Excel sheet 

##### Functionality

* Select table 1 
* Enter name of sample 1
* Select table 2
* Enter name of sample 2
* Select statistical value to analyse

#### Morphological comparison - Generate plots

Creates boxplots with indication of statistically significant difference (only if number of images per sample greater or equal to 8). 

* Select table 1 
* Enter name of sample 1
* Select table 2
* Enter name of sample 2
* Select shape descriptor to display
* Select statistical value to display


#### Correlation analysis 

Check correlation between up to 4 descriptors. 

* Select table 1 
* Enter name of sample 1
* Select table 2
* Enter name of sample 2
* Select shape descriptor to display
* Select statistical value to display

### More than 2 samples

If you want to compare the morphology of multiple samples use this section. Please be aware that the number of images in each sample should be equal. 

#### Morphological comparison - Generate table 

* Select directory in which all tables are located (make sure that only the tables generated with Get Measurements are located in the folder selected) 
* Select statistical value to analyse

#### Morphological comparison - Generate plots

* Select directory in which all tables are located (make sure that only the tables generated with Get Measurements are located in the folder selected) 
* Select shape descriptor to display
* Select statistical value to analyse

## License

BSD 3-Clause License

Copyright 2020 Christian Fischer 

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
