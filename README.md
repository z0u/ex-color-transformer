# DevInterp experiments with color embeddings

This is a series of experiments in which we attempt to impose structure on (latent) embeddings. Ultimately, the goal is to develop a capability to structure the latent spaces in complex models like LLMs.

## Background

Recent experiments have provided some evidence that "bad" behaviors in LLMS cluster together (such as _writing malicious code_ and _being racist_). Although surprising, it makes some intuitive sense: perhaps such behaviors cluster together because it's just the most efficient way to compress knowledge. However, _intervening_ on model behavior remains a tremendous challenge — partly because we don't know which directions in latent space correspond to undesirable traits, and we don't know how tangled up they might be with benign concepts. Indeed, attempts to align models to display "good" behavior often comes at the cost of reduced performance overall.

We hope that this research will reveal more precise and robust ways to constrain the capabilities of LLMs. In contrast to mech interp — which attempts to discover model characteristics _after_ training — we anticipate that anchoring core concepts to known directions will make alignment efforts more robust, through two mechanisms:

1. The relevant directions would be known _even before training_, so you don't need to look for them. This could improve the prospect of both measuring model alignment throughout training, and intervening on misaligned behavior after training.
2. Directions of interest should act as attractors for similar concepts, reducing the chance that unrelated (benign) concepts become entangled with them.

## M1. Preliminary experiments with color

We begin with some experiments with color, because color spaces are well defined and highly intuitive for visualization.

![Visualization of colorful latent embeddings in experiment 1.7, showing three large scatter plots from the end of training, and ten small thumbnails from earlier training steps. The thumbnails show how latent space evolves from a small cluster of dots, to a color cube, though various contortions, until finally it forms a smooth, regular sphere.](docs/m1-color-mlp/large-assets/ex-1.7-color-phase-history.png)

1. [Color data](docs/m1-color-mlp/ex-1.1-color-data.ipynb): Exploration of ways to construct and visualize color spaces such as RGB and HSV.
2. [MLP bottleneck](docs/m1-color-mlp/ex-1.2-color-mlp-bottleneck.ipynb): A 2-layer MLP autoencoder (extremely simple network) that squeezes bright, saturated RGB data through a 2D embedding layer. The network successfully discovers the color wheel — although it needs some help, in the form of explicit normalization.
3. [Curriculum learning](docs/m1-color-mlp/ex-1.3-color-mlp-curriculum.ipynb): The same MLP, but with a 3D embedding layer. Curriculum learning and regularization are used to encourage the model to discover the color wheel without explicit normalization. The hues are embedded into the first two dimensions (as before); later phases in the curriculum add varying tones (values), which naturally balloon out into the third dimension.
4. [Parameter transitions](docs/m1-color-mlp/ex-1.4-parameter-transitions.ipynb): Exploration of ways to cause hyperparameters to vary smoothly over time, both to a schedule, and in reaction to measurements during training.
5. [Smooth curriculum](docs/m1-color-mlp/ex-1.5-color-mlp-anchoring.ipynb): Like experiment 1.3, but with 4D embeddings, and hyperparameters smoothly varying across curriculum phases. For example, the extents of the color space of the training data (HSV) are gradually increased instead of extending it in large discrete steps.
6. [Smooth vs. stepped curricula](docs/m1-color-mlp/ex-1.6-curriculum-comparison.ipynb): A direct comparison of training stability and latent space evolution when using smooth hyperparameter transitions versus traditional stepped phase changes. This experiment had a negative result: it seems the smooth transitions don't help with training dynamics (although they do make curriculum specification easier).
7. [Sparse labels for regularization](docs/m1-color-mlp/ex-1.7-sparse-labels.ipynb): We do away with the data phases, training on the full dataset from the start but with targeted (but noisy) regularization. We achieve similar results to the earlier experiments, but with a more realistic training dataset: previously, the curriculum phases were "clean" in a way that is probably hard to replicate in LLM corpora.
8. [Regularizer combinations](docs/m1-color-mlp/ex-1.8-regularizer-combinations.ipynb): Systematic study to see the effects of each regularizer by itself, and all combinations of the regularizers. In each run, the regularizer weight schedules are kept the same, but select regularizers are not applied at all. We observe that they are all needed to produce a latent space with the desired characteristics.

MLP experiment summary:

| Ex  | Phases                   | Embeddings | Regularization terms                | Hyperparameters  |
| --- | ------------------------ | ---------- | ----------------------------------- | ---------------- |
| 1.2 | 1: Hues only             | 2D         | None (explicit normalization)       | Constant         |
| 1.3 | 5: 6 colors ~ all values | 3D         | Unitarity, planarity                | Stepped          |
| 1.5 | 4: 6 colors ~ all colors | 4D         | Unit, planar, repulsion (Euclidean) | Smooth           |
| 1.6 | 5: 6 colors ~ all colors | 4D         | Unit, planar, repulsion (Euclidean) | Smooth & stepped |
| 1.7 | 1: All colors            | 4D         | Unit, planar, repulsion (cosine)    | Smooth           |
| 1.8 | 1: All colors            | 4D         | All combinations                    | Smooth           |

Publications:

- [Selective regularization for alignment-focused representation engineering - LessWrong](https://www.lesswrong.com/posts/HFcriD29cw3E5QLCR/selective-regularization-for-alignment-focused)

## M2. Practical control and intervention (TO DO)

> Okay, you can structure latent spaces... but can you actually use that structure?

1. Selective concept suppression (temporary, inference-time). Demonstrate intervention at inference time, showing that some colors can be reliably muted without affecting those that are not "close". For example, cause the network to fail to reconstruct _red_ and colors close to red, but allow _orange_.
2. Permanent concept deletion (weight ablation). Demonstrate that the latent space can be further manipulated to completely remove a representation. For example, pressure the network to reconfigure the space so that _only_ red colors are on one particular embedding dimension, and then _delete_ that dimension from the network. Hopefully, this would make it difficult to fine-tune the model later to restore the deleted capability.

## M3. Structured color transformer (TO DO)

Proof-of-concept transformer network with similar latent space structure. It could be a very small transfomer that can perform simple color operations, such as mixing colors.

1. Simple transformer doing color operations (mixing, complementary colors, etc.)
2. Successful transfer of anchoring techniques to the residual stream or QK space (attention mechanism), with validation that structure persists through transformer training dynamics

## M4. Language model application (TO DO)

Impose structure on the latent representations of a transformer language model.

1. Weak labeling pipeline for internet text (identifying "harmful," "deceptive," etc.)
2. Application to actual language model training
3. Evaluation of structured representations in the residual stream or QK space

---

## References

This project relies on several open-source libraries.

- **Matplotlib:** Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. _Computing in Science & Engineering, 9_(3), 90-95.

- **NumPy:** Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020). Array programming with NumPy. _Nature, 585_, 357–362.

- **PyTorch:** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In _Advances in Neural Information Processing Systems_ (pp. 8026-8037).

- **scikit-image:** van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., ... & Yu, T. (2014). scikit-image: image processing in Python. _PeerJ, 2_, e453.

- **scikit-learn:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. _Journal of Machine Learning Research, 12_, 2825-2830.

- **scikit-learn API:** Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., ... & Varoquaux, G. (2013). API design for machine learning software: experiences from the scikit-learn project. _ECML PKDD Workshop: Languages for Data Mining and Machine Learning_, 108-122.
