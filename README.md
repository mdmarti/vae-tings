# vae-tings

Here is a place for me (miles) to store notes and keep track of what I've tried so far with all these different VAEs. tmp branch has whatever I'm working on currently, main will have files that are finished, functional, nicely commented, etc. etc. etc. Really just templates for whatever work I may need to do in the future with VAEs, neural networks, or other things like that. So far models here are:

- vanilla VAE
- \beta - VAE: adds a constant scale to KL-divergence, but with good theory! Tough to find good \beta to balance reconstruction quality with disentanglement. Also - penalizes mutual information?? see [Locatello et al.](https://arxiv.org/abs/1811.12359)
- VQ-VAE: vector-quantization VAE: learns quantization points so that you can use fancy PixelCNN for sampling without getting posterior collapse. Kinda funky. feels tough to get working. Works a little differently than paper description would have you think.
- QP-VAE: a lil sum'n sum'n if you want latent space consistency
