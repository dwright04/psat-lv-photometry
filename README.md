# psat-lv-photometry

psat-lv-photometry is a repository for scripts used to measure photometry of transients discovered during a search for an optical counterpart to the first gravitational wave source discovered by LIGO (GW150914).  Imaging for the search was conducted primarily by Pan-STARRS1 with additional images from the NTT and LSQ.

The key difference between this pipeline and others is the application of machine learning to the selection of reference stars for the zero point calculation.  Typically reference stars are manually selected to ensure quality measurements.  In this case we replace the need for human intervention by employing a Convolutional Neural Network to select only the highest quality stars in the image.

## Papers
[Pan-STARRS and PESSTO search for an optical
counterpart to the LIGO gravitational wave source
GW150914 - Smartt et al. 2016](https://arxiv.org/pdf/1602.04156.pdf)

[Observation of Gravitational Waves from a Binary Black Hole Merger - Abbott et al. 2016](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.116.061102)

## Installation
Clone the Github repository and pip install from requirements.txt.

```
 $ git clone https://github.com/dwright04/psat-lv-photometry.git
 $ pip install -r requirements.txt
```

