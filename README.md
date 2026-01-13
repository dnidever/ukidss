# UKIDSS PSF photometry

Software to create a source catalog of all UKIDSS GPS imaging data.

# Processing Steps

## 1) Measurement (loop over exposure)

    - ukidss_measure.py: core program, runs on one exposure

## 2) Calibration & QA metrics (loop over exposure)

    - ukidss_calibrate.py: core program, runs on one exposure

## 3) Enforce QA cuts, cross-matching, averaging (loop over healpix)

    - ukidss_combine.py: core program, runs on one Healpix pixel

## 4) Load database (loop over exposure and healpix)