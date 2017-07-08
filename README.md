# rusty-filters
A rust library built to design filters in the continuous time domain, convert them to the discrete time domain using the bilinear transform, and visualize them with gnuplot.

I built this library to experiment with and understand protoyping various filters in the discrete domain so that I wouldn't have to waste a bunch of time converting them to discrete filters by hand.  

## example output

### "moog low pass filter" with various cutoff frequencies and quality factors
![frequency response of moog low pass filter](/images/moog4.png "frequency response of moog low pass filter")
### butterworth low pass filter with various cutoff frequencies
![frequency response of butterworth low pass filter](/images/butter4.png "frequency response of butterworth low pass filter")
### butterworth high pass filter with various cutoff frequencies
![frequency response of butterworth high pass filter](/images/butter4-hpf.png "frequency response of butterworth high pass filter")
