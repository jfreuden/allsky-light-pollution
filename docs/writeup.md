**_tl;dr:_** The AllSky camera at the Allegheny Observatory was never intended for scientific use. Accordingly, the data were not saved at FITS files, rather a raw JPG format, with the observation time and timestamps burned into the image, without calibration information. We were able to use an OCR tool to extract text from the images, and able to analyze the images themselves, but could not make rigorous conclusions about the light pollution in our study period.

# Timeline

1. Acquired initial set of images from the Allegheny Observatory's web stream to begin determining what we can do
2. Initially, we believed we would have FITS files to work with, so we prepared code to read the headers and extract the exposure time and gain information.
3. We gained access to a subset of imagery, pre-2021, but they were only stored in JPG format, with the observation time and timestamps burned into the image, but no gain information. We had 1.6 million 640x480 JPG images taken approximately every minute, with interruptions.
4. We attempted to use Google's off-the-shelf Tesseract OCR to read the image text. However, the accuracy was unsatisfactory, due to low contrast and low resolution.
   1. Manipulated image data to improve Tesseract accuracy on our data
   2. Improved runtime performance and parallelization of Tesseract runs through `pyocr`
   3. With small subset runs completed, we noticed that the accuracy of Tesseract OCR on our image text was simply unacceptable. We lost too much data to low parse confidence and couldn't trust the accuracy of what successfully parsed.
5. We wrote a simple text classifier using the original font to parse pre-2016 data in an aliased font. This classifier failed to parse the antialiased text in later images.
6. We adapted the classifier to a new antialiased atlas that could be run on the later data
7. We dropped any row where the confidence of any character match was less than 85%, or where the parsed text didn't fit the format (e.g. no March 74th)
8. We ran ephemera simulations for each image timestamp to collect moon and sun positions from the perspective of the Allegheny Observatory
   1. We dropped any row where the sun was above -18 degrees horizon
   2. We dropped any row where the moon was above -18 degrees horizon, except for new moons (very small subset)
14. We gained access to the remaining imagery. We now had a timespan of August 2010 through March 2026, with some gaps. Full imagery is around 2.5 million images.
15. Re-ran the filtering to create an `ideal_df` "Ideal Dataframe". 581,805 images taken
16. Broad analysis showed that no conclusive statements could be made over the entire time period.

- Attempting to use the exposure time failed due to changes in the exposure settings, and no gain information in the images that could help us make like-for-like comparisons.
- Image brightnesses showed no conclusive statements due to too many reconfigurations or changes


17. We cut the study down to only 112,777 images on or after 21 March 2024, split into two groups:
- 53,626 images between 21 March 2024 and 21 March 2025.
- 59,151 images between 21 March 2025 and 25 March 2026.

18. Seeing no discernable trendline in the noise, and no conclusive results, We hand-filtered all 112,777 images in this period to only the 24,911 where it wasn't cloudy or anomalous in any way.
19. We saw a much-improved error in our plots, showing that the assumption we could integrate over error induced by weather and other factors was disproven.
20. With the much-improved plot (see [notebooks/conclusions/acquisitions-method-unsuitable.ipynb](notebooks/conclusions/acquisitions-method-unsuitable.ipynb)), we could see a startling discontinuous decrease in image brightness in the months BEFORE the streetlight modernization project began.
23. Investigating the imagery around the discontinuity, we found that between 2 March 2025 and 21 June 2025, the camera was capturing faulty images and when it returned to normal operation, the images were noticeably darker for the same sky brightness
	1. The average pixel brightness for clear-night images between 1 January 2025 and 2 March 2025 were 138.6 +/- 5.4.
	2. The average pixel brightness for clear-night images between 22 June 2025 and 4 August 2025 were 115.1 +/- 6.7.
24. When we examine the change in brightness and change in brightness per second of exposure time for 21 June 2025 onward, we see an upward trend implying a 7.2% gain in image brightness over the period, but the fit itself is ultimately unconvincing, and we cannot differentiate an increase in image brightness due to brighter skies from a change to the camera settings or other factors.
25. Particularly, we cannot trust the nonlinearity and unclear behavior in how the camera converts the raw image data to JPG values. As such, we cannot make strong conclusions about the effects of the streetlight modernization on Pittsburgh light pollution, although the data are _indicative_ of an increase in sky brightness.
