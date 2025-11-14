# cgan-wgan-gp-audio-gen

# Conditional WGAN-GP Audio Generation Project

A PyTorch-based **Conditional WGAN-GP** for generating audio from mel‑spectrograms using advanced GAN stabilization techniques, spectral normalization, group normalization, TTUR, L1 reconstruction loss, and HiFi‑GAN vocoding. This project is built for generating class‑conditioned audio for a multi-category sound dataset.

This repository contains:

* A **class‑conditioned generator** for mel‑spectrogram synthesis
* A **spectral-norm + group-norm discriminator** for stable adversarial training
* Gradient penalty (WGAN‑GP)
* TTUR (different learning rates for G & D)
* L1 loss regularization
* Mel→Waveform conversion using **SpeechBrain HiFi‑GAN**
* Real-vs-fake comparison plots
* Automatic spectrogram plotting and audio saving

---

# Project Overview

This project builds a **Conditional WGAN-GP** model that learns to generate mel-spectrograms conditioned on audio class labels (e.g. drilling, dog_bark, siren). The generated mel-spectrogram is then converted into a waveform using a pretrained **HiFi‑GAN vocoder**.

GAN training is notoriously difficult—especially on audio. To stabilize training, this project integrates:

* **WGAN with Gradient Penalty**
* **Spectral Norm + GroupNorm in Discriminator**
* **TTUR** (different LR for G vs D)
* **L1 Mel Reconstruction Loss**
* **Fixed‑latent seed evaluation per epoch**
* **One-real-sample caching per class** for comparison


---

#  Key Features & Techniques Used

## **1. Conditional WGAN‑GP Generator**

* Input: `z` latent vector + one-hot class embedding
* 4-layer ConvTranspose2D upsampling
* Final output: **1 × 80 × 352 mel-spectrogram**
* Output range clipped to `[−15, 1]` to match log-mel dynamic range

**Reason:** Provides stable mel‑spectrogram outputs that align with mel-energy distributions.

---

## **2. Discriminator with Spectral Norm + GroupNorm**

Spectral Normalization is applied to **every** convolutional layer.

Why combine **SpectralNorm + GroupNorm**?

* SpectralNorm stabilizes discriminator Lipschitz continuity
* BatchNorm conflicts with SpectralNorm when batch size is small
* Batch size here = **16**, so **GroupNorm** is ideal (based on research paper below)

Reference research paper:

* *Group Normalization* — [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)

Additional references supporting SpectralNorm:

* [https://www.codegenes.net/blog/pytorch-spectral-norm/](https://www.codegenes.net/blog/pytorch-spectral-norm/)
* Kaggle implementation: [https://www.kaggle.com/code/drlexus/wgan-spectral-normalization-gp-upsampling](https://www.kaggle.com/code/drlexus/wgan-spectral-normalization-gp-upsampling)

**Reason:** Reduces discriminator oscillation and provides much smoother GAN dynamics.

---

## **3. Gradient Penalty (WGAN‑GP)**

Implements the classic WGAN‑GP penalty:

* Improves Lipschitz constraint
* Prevents mode collapse
* Provides smooth discriminator gradients

References:

* [https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/blob/master/Computer_Vision/WGAN_with_Gradient_Penalty_from_Scratch_PyTorch/utils.py](https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/blob/master/Computer_Vision/WGAN_with_Gradient_Penalty_from_Scratch_PyTorch/utils.py)
* [https://medium.com/@m.naufalrizqullah17/exploring-conditional-gans-with-wgan-4f13e91d30eb](https://medium.com/@m.naufalrizqullah17/exploring-conditional-gans-with-wgan-4f13e91d30eb)

---

## **4. TTUR — Two Time-Scale Update Rule**

Used separate learning rates for Generator and Discriminator:

* **LR_G = 1e-4**
* **LR_D = 2e-5** (slower)

Research papers:

* [https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html](https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html)
* TTUR Paper: [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

**Reason:** GANs converge faster and more stably when G and D learn at different speeds.

---

## **5. L1 Reconstruction Loss**

Added alongside adversarial loss:

* Helps generator match real mel structure
* Improves high-frequency stability

Reference:

* [https://www.codegenes.net/blog/l1loss-pytorch/](https://www.codegenes.net/blog/l1loss-pytorch/)

---

## **6. Mel-spectrogram Processing (SpeechBrain)**

Used SpeechBrain's mel-spectrogram pipeline:

* Accurate mel normalization
* Slaney mel-scale
* Better compatibility with HiFi‑GAN

Reference:

* [https://www.codegenes.net/blog/melspectrogram-pytorch/](https://www.codegenes.net/blog/melspectrogram-pytorch/)

---

## **7. Log‑Mel → Waveform Conversion (HiFi‑GAN)**

The model uses pretrained **HiFi‑GAN** from SpeechBrain:

* [https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz](https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz)

**Reason:** Converts generated log-mels into high-quality waveforms.

---

## **8. Reproducibility Enhancements**

Fixed seeds using PyTorch's official reproducibility guide:

* [https://docs.pytorch.org/docs/stable/notes/randomness.html](https://docs.pytorch.org/docs/stable/notes/randomness.html)

**Reason:** Ensures consistent training behavior.

---

## **9. Activation Functions**

Used LeakyReLU throughout.

Activation reference:

* [https://medium.com/@sushmita2310/12-types-of-activation-functions-in-neural-networks-a-comprehensive-guide-a441ecefb439]

---

## **10. Base Code Inspiration**

The initial GAN structure (mainly dataloader + plotting logic) was inspired by sample code given in github repo:

* [https://github.com/ankush-10010/Decibal-Duel](https://github.com/ankush-10010/Decibal-Duel)

---


#  Training Details

* **Epochs:** 300
* **Batch Size:** 16
* **Latent Dim:** 100
* **Optimizer:** Adam
* **Betas:** (0.0, 0.99)
* **Loss:** WGAN-GP + L1

---

#  Output

During training, the system automatically saves:

* Generated mel-spectrograms every 10 epochs
* Generated waveform audio
* Real-vs-Fake comparison plots
* Training logs

---

#  References List

(resources used in model design)

### GAN & Stability

* [https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/](https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/)
* TTUR: [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)
* GAN convergence stability: [https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html](https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html)

### SpectralNorm / GroupNorm / BatchNorm

* [https://www.codegenes.net/blog/pytorch-spectral-norm/](https://www.codegenes.net/blog/pytorch-spectral-norm/)
* [https://www.kaggle.com/code/drlexus/wgan-spectral-normalization-gp-upsampling](https://www.kaggle.com/code/drlexus/wgan-spectral-normalization-gp-upsampling)
* GroupNorm: [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)

### Mel processing / Vocoding

* [https://www.codegenes.net/blog/melspectrogram-pytorch/](https://www.codegenes.net/blog/melspectrogram-pytorch/)
* [https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz](https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz)

### Conditional GANs

* [https://medium.com/@m.naufalrizqullah17/exploring-conditional-gans-with-wgan-4f13e91d30eb](https://medium.com/@m.naufalrizqullah17/exploring-conditional-gans-with-wgan-4f13e91d30eb)

### L1 Loss

* [https://www.codegenes.net/blog/l1loss-pytorch/](https://www.codegenes.net/blog/l1loss-pytorch/)

### Reproducibility

* [https://docs.pytorch.org/docs/stable/notes/randomness.html](https://docs.pytorch.org/docs/stable/notes/randomness.html)

### Base Inspiration

* [https://github.com/ankush-10010/Decibal-Duel](https://github.com/ankush-10010/Decibal-Duel)

---

#  Final Notes

This model is currently being trained only till 210 epochs due to limited time . Although audio quality may still improve with more training or further tuning, the architecture provides a strong baseline for future experimentation.
