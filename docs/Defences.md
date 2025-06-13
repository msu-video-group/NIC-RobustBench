## Defence algorithms

| Attack       | Implementation directories                                  | Description  | Original Paper |Original Implementation |
| -------------|-------------------------------------------------------------| ------------ |------------ |------------ |
| Flip         | `reversible_flip`                      | Reflects the image horizontally or vertically | |
| Random roll       | `reversible_random_roll`                                                     | Rolls the image by a random number of pixels | | 
| Random rotate         | `reversible_random_rotate`  | Rotates the image on the selected angle | 
| Random color reorder          | `reversible_random_color_order`                                                 | Swaps color channels of the image tensor | 
| Random ensemble         | `reversible_ensemble`                               | Combination of 10 actions from Roll, Rotate, and Color ||
| Geometric self-ensemble         | `self_ensemble`                                                      | Generates 8 image candidates with flipping and rotation and chooses the least distorted after preprocess-NIC-postprocess pipeline | [link](https://ieeexplore.ieee.org/abstract/document/10124732/?casa_token=ItkISg0qW0sAAAAA:JOZNCXeP-ugNmp0SkRRvb04fVYYRtKgA9l3vEFfjsMjLrNDKSQ0bMhkuVSQCmoE1E8FKON3t0lj8)|
| DiffPure | `diffpure`                                              | Applies DiffPure purification without postprocess step |[link](https://arxiv.org/abs/2205.07460)|
