# WSRD-DNSR

This is the repository of the WSRD dataset, used as benchmark for the **NTIRE2023 Challenge on Image Shadow Removal**. 

# Training data
The collection of 1000 roughly aligned image pairs can be found using the following URLS: 
[train input link](https://drive.google.com/file/d/1sElA4ztV44wiSewyWgpacjFI42MgpOhe/view?usp=sharing) | [train gt link](https://drive.google.com/file/d/1OfcSirKj27CTJvG8B2jZgTB1Vb9PjEXd/view?usp=sharing).

# Validation data
We provide a database of 100 images, that were used for evaluation on the provided Codalab server: 
[valid input link](https://drive.google.com/file/d/1Sjbuf4GtGW-kG_GdpM3wRRgAEEnTF2fg/view?usp=sharing) | [valid gt link](https://drive.google.com/file/d/1jWdSrUATvA1MrrYW2Mgrh6PScF6egkLF/view?usp=sharing).

# Testing data
The input images used for testing in the Final Phase of the challenge are available [here](https://drive.google.com/file/d/1CdDh9XdkITmzHn08mCchoIQQojoJ8rvl/view?usp=sharing).
Since we are planning for a second edition of the challenge, the test ground-truth images will remain private, for the moment. 
Follow the current repository for more updates.  

# DNSR
Along with the provided data, we also provide results for DNSR, a baseline optimized for reconstruction fidelity. 

 * Checkpoints for DNSR are available [here](https://drive.google.com/drive/folders/1E6dsgKl6tOKFBEu__6fovPQHvCDrxryG?usp=drive_link).
 * Results of the DNSR over the WSRD benchmark are available [here](https://drive.google.com/drive/folders/16QT_99F9puhcmUI_iljS7vLSLksg1o4B?usp=sharing). 
 * Additional results for the ISTD/ISTD+ benchmarks are available [here](https://drive.google.com/drive/folders/1H2YyWB2_XxK7GEAksdLitRUB6jbCMOmQ?usp=sharing).

Please follow the current repository for more updates. 

---

---
# NTIRE 2024 and WSRD+

A version of the WSRD Dataset will be used as a benchmark for the **NTIRE24 Challenge on Image Shadow Removal**. 
The challenge has a [fidelity track](https://codalab.lisn.upsaclay.fr/competitions/17539) and a [perceptual track](https://codalab.lisn.upsaclay.fr/competitions/17546).

# Training data
This new version proposes improved pixel-alignment through  homography estimation. 
[train input link](https://drive.google.com/file/d/1n9l3UyQw6HjCXqycvHAfl4T-jsJpPHeJ/view?usp=drive_link) | [train gt link](https://drive.google.com/file/d/1DZEMIJ8PIxmZww8iAqlcvlKWyfssNQRO/view?usp=sharing).

# Validation data
The validation split is used in the Development Phase of the challenge. 
Here, you can download the [input images](https://drive.google.com/file/d/1l2aertz2qKVLUkP-egwiCBcyf_5GWnav/view?usp=sharing) | [ground truth images](https://drive.google.com/file/d/1a8JVs_zVQSdmxeDYJnqeEyynd9wV6n5D/view?usp=sharing). 



# Testing data
The test split will be used in the final Test Phase. Since we are aiming for proper evaluation on unseen data, these images will stay, for the moment, private.

# Online validation system
To test your model on both validation and testing splits, you can use the [Codalab competition](https://codalab.lisn.upsaclay.fr/competitions/17539) which will remain open.
[Results](https://codalab.lisn.upsaclay.fr/competitions/17539#results) comparing the teams participating in the 2024 challenge are also available. 

# Further Requests
For access and other requests feel free to drop us an [email](mailto:florin-alexandru.vasluianu@uni-wuerzburg.de). 