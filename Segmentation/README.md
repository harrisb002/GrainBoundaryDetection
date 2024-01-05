## Grain Boundary Segmentation

---

This section covers various segmentation techniques use to segment grain boundaries isolating specific chemical regions, most notably silica (quartz).

The techniques tested are:
* Watershed
* SLIC (superpixel)
* Felzenszwalb

During the tests, it is observed that Felzenszwalb provided the most accurate of the three and is a good candidate for further testing.

The tests can be found with examples in the `Boundary Segmentation.ipynb` notebook with step-by-step documentation for each.

The first cells in the notebook explores Watershed, SLIC, and Felzenszwalb segmentations. After these examples, the next section labeled, `THE GOOD STUFF`, defines a number of functions used in the full implementation of the segmentation algorithm, and then the algorithm itself, `segment_and_overlay_chemistry`. The algorithm uses Felzenszwalb segmentation and postprocessing includes the labeling and drawing of the boundaries. There are also a number of additional options available to fine-tune Felzenszwalb as well as options for labels, colors, and overlays. In the algorithm's function body is a complete list and description of each parameter of the function and a description of each helper function.

In the next cell is the most important function for obtaining the boundaries within the chemistry mask. `segment_and_overlay_chemistry` takes in the band contrast image, the chemistry mask (that is, a black and white image containing only the target area in white) and the preferred opacity and color of the labels which will be projected onto the band contrast image, and returns the image.  

The final cell is the execution of the function `segment_boundaries` -- the workhorse of `segment_and_overlay_chemistry`-- which uses images of grain chemistry and overlays their labeled and segmented boundaries onto the band contrast image. There are step-by-step explanations for each code block as well as additional information. The function call is simple and makes the whole process very easy to use. Following the algorithm execution, the resulting images saved and the segmented images generated are plotted next to their masks for visual comparison.

#### Note: Running this cell can take several minutes to complete. There are print statements that update every iteration to notify status.
