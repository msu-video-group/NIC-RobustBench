## Attack algorithms

| Attack       | Implementation directories                                  | Description  | Original Paper |Original Implementation |
| -------------|-------------------------------------------------------------| ------------ |------------ |------------ |
| FTDA         | `ftda`, `ftda-linf`, `ftda-randn-init`                      | NIC attack to increase distance between decoded images | [link](https://ieeexplore.ieee.org/abstract/document/10124732)| [link](https://github.com/tongxyh/ImageCompression_Adversarial)
| I-FGSM       | `ifgsm`                                                     | Iterative sign gradient descent | [link](https://arxiv.org/abs/1607.02533) | [link](https://github.com/1Konny/FGSM)
| MADC         | `madc`, `madc-linf`, `madc-norm`, `madc-randn-init`,`mad-mix`  | Proj. grad. on a proxy metric (MSE) | [link](https://jov.arvojournals.org/article.aspx?articleid=2193102)
| PGD          | `pgd-ifgsm`                                                 | I-FGSM with random initialization | [link](https://arxiv.org/abs/1706.06083)
| SSAH         | `ssah`,`ssah-randn-init`                                    | Grad. desc. in high freq. domain | [link](http://openaccess.thecvf.com/content/CVPR2022/html/Luo_Frequency-Driven_Imperceptible_Adversarial_Attack_on_Semantic_Similarity_CVPR_2022_paper.html)| [link](https://github.com/LinQinLiang/SSAH-adversarial-attack)
| CAdv         | `cadv`                                                      | Gradient descent with color filter | [link](https://arxiv.org/abs/1904.06347)| [link](https://github.com/AI-secure/Big-but-Invisible-Adversarial-Attack)
| Random noise | `random-noise`                                              | Gaussian noise with $\sigma \in [\frac{5}{255}; \frac{14}{255}]$ ||
| Korhonen et al. | `korhonen-et-al`                                         | Sobel-filter-masked gradient descent | [link](https://dl.acm.org/doi/abs/10.1145/3552469.3555715)|