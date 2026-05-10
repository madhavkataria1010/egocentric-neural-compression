# Related Work and Positioning

*Last updated: 2026-05-09. This is a literature scan that grounds the positioning of `ego-codec`. Anything in this repo's README or pitch should be supportable from one of the references below or explicitly flagged as our contribution.*

---

## TL;DR — what we are not, and what we are

We are **not** proposing a new neural video codec architecture. State-of-the-art learned codecs (DCVC-RT, DCMVC, DCVC-FM) are already past H.266/VVC at competitive speed. Beating them on generic video would take a small army.

We **are** proposing that the input distribution learned codecs are tuned for — generic exocentric video — is the wrong one for the dataset that egocentric data-aggregators (Build AI, Aria, Quest, Vision Pro, smart glasses) actually have. Specifically, we exploit **synchronized IMU as conditional side information** for inter-frame motion compensation. To our knowledge no published learned codec uses IMU this way; the closest prior work uses gyro signals for *stabilization* (a perceptual-quality task), not *compression* (a rate-distortion task).

The contribution is therefore **methodological / data-design**, not architectural: a clean experimental answer to "if your data already has an IMU, how many bits is it worth?"

---

## 1. Learned image and video compression

### 1.1 Image codec genealogy

Modern learned image compression begins with **Ballé et al. 2017** (factorized prior) and **Ballé et al. 2018** (scale hyperprior), followed by **Minnen, Ballé & Toderici, NeurIPS 2018** — the *Mean-Scale Hyperprior with autoregressive context* that became the de facto baseline for the field. Our I-frame codec is a Mean-Scale Hyperprior with the autoregressive context omitted (we found, like much subsequent work, that the autoregressive part trades a small RD gain for a large decode-latency penalty).

- **Reference baseline (this repo)**: Mean-Scale Hyperprior, ~2.5M params, no autoregressive context.

### 1.2 Video codec genealogy

End-to-end learned video compression starts with **Lu et al. (DVC), CVPR 2019** — the first model to optimize motion estimation, motion compression, and residual compression jointly in a single learned pipeline. DVC uses an explicit optical-flow encoder (PWC-Net) and codes both flow and residual.

Subsequent improvements:

- **Hu et al., FVC** — feature-space video coding (motion estimation in latent space).
- **Li, Li & Lu, DCVC, NeurIPS 2021** — replaces explicit residual coding with *contextual* coding: the previous reconstruction conditions the entropy model rather than being subtracted out. ([paper](https://papers.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf))
- **DCVC-DC, DCVC-HEM** — diversified context, hybrid entropy model.
- **Li, Li & Lu, DCVC-FM, CVPR 2024** — feature modulation across rate; 25.5% over VTM (H.266) at intra-period −1. ([paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Neural_Video_Compression_with_Feature_Modulation_CVPR_2024_paper.pdf))
- **Jia et al., DCVC-RT, CVPR 2025** — practical real-time variant. 100+ FPS at 1080p, 4K real-time, 21% over H.266. Identifies memory I/O and function-call overhead (not MACs) as the real bottleneck and removes explicit motion modules in favor of implicit temporal modeling. ([paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.pdf), [project](https://dcvccodec.github.io/))
- **Tang et al., Context Modulation, CVPR 2025** — temporal context conditioning. ([paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Neural_Video_Compression_with_Context_Modulation_CVPR_2025_paper.pdf))
- **DCMVC** — 22.7% over H.266, 10.1% over DCVC-FM (latest as of search).
- **Khan et al., Perceptual Video Compression with Neural Wrapping, CVPR 2025** — pre/post-processing networks bolted onto traditional codecs.
- **Regensky et al., ICCV 2025** — Beyond Perspective: 360° neural video compression. ([paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Regensky_Beyond_Perspective_Neural_360-Degree_Video_Compression_ICCV_2025_paper.pdf))

### 1.3 Comparison vs traditional codecs (what we are competing against)

Recent benchmarks ([Conventional vs Learned, arXiv 2408.05042](https://arxiv.org/abs/2408.05042); [4K eval, arXiv 2511.00969](https://arxiv.org/html/2511.00969)) show:

| Codec | BD-rate vs VTM (PSNR) | Status |
| --- | --- | --- |
| ECM (latest traditional) | −18.3% | SOTA traditional |
| AVM (AV1 successor) | −16.1% | |
| DCVC-FM | −7.3% | CVPR 2024 |
| DCVC-RT | −21% (vs H.266) at real-time speed | CVPR 2025 |
| DCMVC | −22.7% (vs H.266) | latest neural |

For perceptual metrics (VMAF), neural codecs match or beat ECM with > 8% gain. The headline: **neural codecs have caught up to traditional codecs on PSNR and pulled ahead on perceptual quality** as of 2025.

The implication for this project: head-to-head architecture papers are crowded. The interesting frontier is *what we condition on*, not the architecture of the entropy model.

---

## 2. Side information and conditional coding

### 2.1 Information-theoretic foundation

The right framing for "video + IMU" coding is **conditional source coding with two-sided side information**: both encoder and decoder have the IMU, so we want to minimize H(video | IMU), the conditional entropy. This is the *easy* corner of the side-information taxonomy.

The harder problem — **Wyner-Ziv coding** (1976) — is when side info is only at the decoder. Recent work has put learned models on that problem:

- **Özyılkan et al., 2023, "Learned Wyner-Ziv Compressors Recover Binning"** — neural networks rediscover the optimal binning structure under quadratic-Gaussian assumptions. ([arXiv 2305.04380](https://arxiv.org/pdf/2305.04380))
- **Mital et al., "Learned layered coding for Successive Refinement in the Wyner-Ziv Problem"** ([arXiv 2311.03061](https://arxiv.org/abs/2311.03061))
- **Survey: Distributed Compression in the Era of ML** ([arXiv 2402.07997](https://arxiv.org/html/2402.07997v1))

We treat IMU as two-sided side info, so we don't need WZ machinery — but the theoretical upper bound on bit savings is bounded by the mutual information I(video; IMU). Empirically that mutual information is large for ego-video because IMU directly explains the dominant motion component.

### 2.2 Multi-modal learned compression

The closest published practice to ours is multi-modality compression where the *side modality* is itself an image:

- **Lu et al., CVPR 2022 — Learning Based Multi-Modality Image and Video Compression** — Transformer-based spatial alignment for joint RGB+IR compression on FLIR/KAIST. ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Learning_Based_Multi-Modality_Image_and_Video_Compression_CVPR_2022_paper.pdf))
- **Multi-Modality Deep Network for Extreme Learned Image Compression**, 2023. ([arXiv 2304.13583](https://arxiv.org/abs/2304.13583))

Recent work goes further into *generative* conditioning:

- **M3-CVC: Controllable Video Compression with Multimodal Generative Models**, 2024 — text prompts as side info, diffusion decoders. ([arXiv 2411.15798](https://arxiv.org/html/2411.15798))

The dominant pattern is text/audio/depth as conditioning. **No published codec, to our knowledge, uses IMU.** This is the gap we step into.

### 2.3 Distributed video coding (older)

Classical Distributed Video Coding ([Wikipedia](https://en.wikipedia.org/wiki/Distributed_source_coding)) used Wyner-Ziv constructions to push complexity from encoder to decoder for low-power capture devices. The encoder sends syndromes; the decoder reconstructs using side info from temporally adjacent frames. The intuition that *something cheap to send can replace something expensive to compute* is the same intuition we're applying with IMU.

---

## 3. Egocentric video data

### 3.1 Project Aria (Meta)

The Aria platform ([Engel et al. 2023, arXiv 2308.13561](https://arxiv.org/abs/2308.13561)) is a research-grade ego-glasses with synchronized:
- High-res RGB + 2× monochrome SLAM cameras + eye-tracking cameras
- 2× IMUs at 1 kHz (accel @ 8g, gyro)
- Magnetometer, barometer, multi-mic spatial audio

Datasets:
- **Aria Everyday Activities (AEA)** — Lv et al. 2024 ([arXiv 2402.13349](https://arxiv.org/abs/2402.13349)). 143 sequences, 5 indoor locations, 4 wearers × 26 hours total. Comes with high-frequency 3D trajectories, scene point clouds, per-frame eye gaze, time-aligned speech transcripts. **Why we use it:** synced video+IMU, license is open, scale is right for a 1-week artifact.
- **Nymeria** — Ma et al. ECCV 2024 ([arXiv 2406.09905](https://arxiv.org/abs/2406.09905)). 300 hours, 264 participants, 1200 sequences, **with full-body XSens mocap**. miniAria wristbands provide additional IMU. The richest egocentric+motion dataset extant. *If the IMU-conditioning story works on AEA, Nymeria is the obvious next dataset.*
- **Aria Everyday Objects, Aria Synthetic Environments, Aria Pilot Dataset** — additional Aria flavors for object-centric and SLAM tasks.

### 3.2 Build AI / Eddy Xu datasets

The strategic context of this artifact. Build AI's data scaling has been the fastest in the field:

| Release | Hours | Workers | Resolution | Codec | Format |
| --- | ---: | ---: | --- | --- | --- |
| Egocentric-10K (Nov 2025) | 10,000 | 2,153 | 1080p @ 30 fps | (raw / lossless?) | 16.4 TB total |
| Egocentric-100K (Dec 2025) | 100,000 | 14,228 | **456 × 256** (fisheye, Kannala-Brandt) | **H.265** | WebDataset, MP4 in TAR shards |
| Egocentric-1M (Apr 2026) | ~1,000,000 | — | — | — | "Largest egocentric dataset in the world" |

Sources: [Egocentric-10K HF](https://huggingface.co/datasets/builddotai/Egocentric-10K), [Egocentric-100K HF](https://huggingface.co/datasets/builddotai/Egocentric-100K), [Eddy Xu on X](https://x.com/eddybuild/status/2041751488817774968).

**Key observations for our pitch:**
1. Build AI is **already storage-cost sensitive** — they downresed to 456×256 between Gen-10K and Gen-100K and adopted H.265. They will care about a 20% storage win.
2. Egocentric-100K's "Build AI Gen 1" hardware is **monocular video only** per public info — no IMU stream is documented. This means:
   - For Gen-1 archive data, our IMU conditioning doesn't directly apply.
   - For **Build AI Gen 2 onward**, adding an IMU is a near-zero-cost hardware decision (every smartphone already ships one). If a Gen-2 with IMU were used to capture future data, IMU-conditioned compression could be applied directly — and the per-hour storage cost compounds across millions of hours.
3. Eddy explicitly lists [egocentric-native neural compression and 3D head position via VIO](https://www.eddy.build/) among his research interests. He has the data, the cost pressure, and the stated interest. The fit is direct.

### 3.3 Other egocentric datasets

- **Ego4D** (Grauman et al. CVPR 2022) — 3,670 hours. Mostly RGB+audio; IMU sparse.
- **EPIC-KITCHENS** — kitchen-centric, has gyro/accel.
- **Ego-Exo4D** (Grauman et al. 2024) — paired ego and exo.
- **HoloAssist** — task-oriented ego with IMU.

---

## 4. IMU + computer vision (the gyro-stabilization lineage)

This is the most mature body of work that directly uses IMU at the pixel level — but **for stabilization, not compression**.

- **Karpenko, Jacobs, Baek, Levoy. *Digital Video Stabilization and Rolling Shutter Correction using Gyroscopes.* Stanford CSTR 2011-03.** ([PDF](https://graphics.stanford.edu/papers/stabilization/karpenko_gyro.pdf)) — the canonical reference. Real-time on iPhone 4. Gyroscope → frame-to-frame rotation → homography correction. *Our `IMUWarpPredictor` is a learned generalization of Karpenko's gyro-to-homography mapping, applied as a motion-compensation prior rather than a stabilization warp.*
- **Bell et al. (Springer 2014)** — Non-linear filter for gyroscope-based stabilization.
- **Hong et al. (Sensors 2018)** — Hybrid IMU + KLT motion estimation.
- **Gyroflow** (open source) — IMU-driven stabilization for action cameras / drones.
- **IMU-aided motion deblurring** (PMC 11622971) — gyro → ego-motion PSF for blur kernel.
- **MotionTrace** ([arXiv 2408.01850](https://arxiv.org/html/2408.01850v1)) — IMU-based field-of-view prediction for AR.
- **Solin et al. 2017** ([arXiv 1703.00154](https://arxiv.org/pdf/1703.00154)) — inertial odometry on smartphones.

The pattern: gyro/IMU is consistently shown to recover global camera motion cheaply and accurately. **Our claim is that this same global-motion estimate is exactly the side info a learned video codec needs to skip flow-encoding bits.**

---

## 5. Foveation and gaze-conditioned compression

A parallel line of work uses **gaze**, not IMU, as the relevant ego signal. Worth noting because Aria streams eye-gaze, so a follow-up could combine.

- **Kaplanyan et al. SIGGRAPH 2019 — DeepFovea** — neural reconstruction for foveated rendering and video compression using natural-video statistics. ([ACM 3355089.3356557](https://dl.acm.org/doi/10.1145/3355089.3356557))
- **FovOptix, MMSys 2024** — foveated rendering + foveated encoding for VR streaming. ([paper](https://openreview.net/pdf?id=YsN5c3xidK))
- **Ye et al., Computer Animation and Virtual Worlds 2024** — neural foveated super-resolution for VR. ([Wiley](https://onlinelibrary.wiley.com/doi/10.1002/cav.2287))
- **Schwarz et al. 2025 — Foveated Compression for Telepresence Visualization** ([PDF](https://www.ais.uni-bonn.de/papers/Telepresence_2025_Schwarz.pdf))

Difference vs our work: foveation buys bits via *perceptual* relaxation (peripheral pixels need less quality). IMU conditioning buys bits via *predictive* compression (the residual is smaller). They compose.

---

## 6. Robotics-data infrastructure (the bar to beat)

If the pitch is "compress Build AI's archive better," we need to know what the field already does:

- **Robo-DM** — Berkeley AutoLab, ICRA 2025. ([paper](https://autolab.berkeley.edu/assets/publications/media/2025_ICRA_DataLoader_final_v3.pdf), [GitHub](https://github.com/BerkeleyAutomation/robodm)) Uses **AV1 @ CRF=30**. Achieves up to 70× lossy compression vs raw, 50× faster decode than LeRobot.
- **LeRobot** (Hugging Face) — also AV1 @ CRF=30. The de-facto pipeline.
- **NVRC** ([arXiv 2409.07414](https://arxiv.org/html/2409.07414v1)) — INR-based video compression.
- **PNVC** — practical INR codec. ([arXiv 2409.00953](https://arxiv.org/html/2409.00953v1))

**Implication for benchmarks:** `pframe-imu` should be evaluated against `x265` (preferably `libsvtav1`) at matched bitrates as the *industry baseline*, in addition to our `pframe-noimu` ablation. Without that comparison the result reads as "neural beats neural", which is uninteresting.

---

## 7. The gap we step into

After this review, the gap is sharp:

| Capability | Karpenko-line | Learned codecs | This repo |
| --- | :-: | :-: | :-: |
| Use IMU for global motion | ✅ | ❌ | ✅ |
| End-to-end RD optimization | ❌ | ✅ | ✅ |
| Egocentric-specific eval | ❌ | ❌ | ✅ |
| Treat IMU as zero-cost side info | (n/a) | ❌ | ✅ |

To our knowledge no published learned codec uses IMU as conditioning. The combination is straightforward to argue (IMU dominates pixel motion for ego-video; it's already streamed alongside; learned codecs have a clean "context" slot to plug it into) and we believe yields a measurable RD win on Aria — testable in a week.

---

## 8. Implications for this artifact

Concrete decisions baked from this review:

1. **Train at 256×256** patches (not 1080p) — Build AI's actual operating resolution is 456×256.
2. **Compare against `x265` (and `libsvtav1` if time allows)**, not just our own ablations. AV1@CRF=30 is the realistic baseline.
3. **Use Aria Everyday Activities** for IMU-paired training data; report Nymeria as future work.
4. **Frame the cost story explicitly** in the README's "what I'd do at scale" — at 1M hours of 256p H.265 (Eddy's roadmap), a 20% bit saving is not a paper figure, it is millions of dollars of S3 amortized over years.
5. **Cite Karpenko 2011 as the conceptual ancestor** — this is honest and signals research literacy.
6. **Be explicit that we are not proposing a new codec architecture.** The contribution is the conditioning + the application + the evidence; honesty here is more credible than overclaiming.
7. **Report mutual-information lower-bound, not just BD-rate** — even a one-line estimate of I(residual; IMU-flow) makes the claim concrete and testable.

## 9. Reading priority before training kicks off

If we have time before the A40 sweep starts, in priority order:

1. Karpenko 2011 (8 pages) — geometry of gyro→homography.
2. Lv et al. 2024 (Aria Everyday Activities) — exact IMU/timestamp conventions, gotchas.
3. Lu et al. CVPR 2022 (multi-modality compression) — closest published methodology.
4. Jia et al. CVPR 2025 (DCVC-RT) — what "real-time" actually demands; informs §8 of the README.
5. Ma et al. ECCV 2024 (Nymeria) — for the "what I'd do next" section.

---

## 10. References (consolidated)

- A. Karpenko, D. Jacobs, J. Baek, M. Levoy. *Digital Video Stabilization and Rolling Shutter Correction using Gyroscopes.* Stanford CSTR 2011-03. <https://graphics.stanford.edu/papers/stabilization/karpenko_gyro.pdf>
- D. Minnen, J. Ballé, G. Toderici. *Joint Autoregressive and Hierarchical Priors for Learned Image Compression.* NeurIPS 2018.
- G. Lu et al. *DVC: An End-to-end Deep Video Compression Framework.* CVPR 2019. <https://arxiv.org/abs/1812.00101>
- J. Li, B. Li, Y. Lu. *Deep Contextual Video Compression.* NeurIPS 2021. <https://papers.neurips.cc/paper/2021/file/96b250a90d3cf0868c83f8c965142d2a-Paper.pdf>
- J. Li, B. Li, Y. Lu. *Neural Video Compression with Feature Modulation* (DCVC-FM). CVPR 2024. <https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Neural_Video_Compression_with_Feature_Modulation_CVPR_2024_paper.pdf>
- Z. Jia et al. *Towards Practical Real-Time Neural Video Compression* (DCVC-RT). CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.pdf>
- C. Tang et al. *Neural Video Compression with Context Modulation.* CVPR 2025.
- G. Lu et al. *Learning Based Multi-Modality Image and Video Compression.* CVPR 2022. <https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_Learning_Based_Multi-Modality_Image_and_Video_Compression_CVPR_2022_paper.pdf>
- E. Özyılkan et al. *Learned Wyner-Ziv Compressors Recover Binning.* 2023. <https://arxiv.org/pdf/2305.04380>
- J. Engel et al. *Project Aria: A New Tool for Egocentric Multi-Modal AI Research.* 2023. <https://arxiv.org/abs/2308.13561>
- Z. Lv et al. *Aria Everyday Activities Dataset.* 2024. <https://arxiv.org/abs/2402.13349>
- L. Ma et al. *Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild.* ECCV 2024. <https://arxiv.org/abs/2406.09905>
- A. Kaplanyan et al. *DeepFovea.* SIGGRAPH 2019.
- T. Chen et al. *Robo-DM: Data Management For Large Robot Datasets.* ICRA 2025. <https://arxiv.org/abs/2505.15558>
- Build AI / Eddy Xu — Egocentric-10K, -100K, -1M datasets. <https://huggingface.co/datasets/builddotai/Egocentric-100K>
