## Harnessing Input-Adaptive Inference for Efficient VLN [[ICCV 2025]](https://iccv.thecvf.com/virtual/2025/poster/429)

This repository contains the code for reproducing the results of our paper:

- Harnessing Input-adaptive Inference for Efficient VLN
- **Dongwoo Kang**, [Akhil Perincherry](https://www.akhilperincherry.com), [Zachary Coalson](https://www.zachcoalson.com), Aiden Gabriel, [Stefan Lee](https://web.engr.oregonstate.edu/~leestef), [Sanghyun Hong](https://sanghyun-hong.com).

&nbsp;

### TL;DR

We present a novel test-time optimization method to improve efficiency in VLN by 1.7-2.6× while retaining 77–89% of task success.

&nbsp;

### Abstract

Abstract: An emerging paradigm in vision-and-language navigation (VLN) is the use of history-aware multi-modal transformer models. Given a language instruction, these models process observation and navigation history to predict the most appropriate action for an agent. While they have significantly improved performance, the scale of these models can be a bottleneck in practical settings with limited computational resources. In this work, we propose a novel input adaptive navigation method to enhance VLN model efficiency. We first show that existing input-adaptive mechanisms fail to reduce computations without substantial performance degradation. To address this, we introduce three adaptive algorithms, each deployed at a different level: (1) To improve spatial efficiency, we selectively process panoramic views at each observation of an agent. (2) To improve intra-model efficiency, we propose importance-based adaptive thresholding for the early-exit methods. (3) To improve temporal efficiency, we implement a caching mechanism that prevents reprocessing of views previously seen by the agent. In evaluations on seven VLN benchmarks, we demonstrate over a 2× reduction in computation across three off-the-shelf agents in both standard and continuous environments. 

&nbsp;

---

## Repository Structure

- [`Standard-VLN`](./Standard-VLN/): Contains code for reproducing our results in the standard VLN setting (Sections 4.1, 4.3, and 4.4).
- [`Continuous-VLN`](./Continuous-VLN/): Contains code for reproducing our results in the continuous VLN setting (Section 4.2).

&nbsp;

---

## Cite Our Work

Please cite our work if you find this source code helpful.

```
@inproceedings{kang2025harnessing,
  title={Harnessing Input-Adaptive Inference for Efficient VLN},
  author={Kang, Dongwoo and Perincherry, Akhil and Coalson, Zachary and Gabriel, Aiden and Lee, Stefan and Hong, Sanghyun},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2025},
}    
```

&nbsp;

---

Please contact Dongwoo Kang (kangdo@oregonstate.edu) for any questions and recommendations.