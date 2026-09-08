# We still don't know if the models want to learn

We recieved **15,602 accepted uploads from 206 GitHub accounts** to One Layer Deeper.

## The Challenge

The public problem is repeated modular squaring:

```python
z = x % N
for _ in range(T):
    z = z * z % N
```

Our [launch post](https://blog.tilderesearch.com/blog/one-layer-deeper) asked participants to design architectures, optimizers, and losses that learn this computation. Loops were allowed; supplying an arithmetic solver or generating additional training targets with one was prohibited.

Throughout the competiton the leaderboard showed held out test set accuracy for the Hard tier which allowed one H100, one hour of training, 30 minutes of evaluation, and 500 million model-state elements. Certification required every example correct at consecutive rungs: T=1, 2, 4, 8, 16, 32, 64.
Thanks to @az, in the beta phase we very quickly discovered we needed to vary the Hard problem slightly to stop submissions which hardcode the algorithm.
We also extend thanks to @apaz who made us write the rules for the main in a very different way. At a high level their solution was RL over basic python operations like multiply and addition, this was a very nice solution but a little outside of end to end learning.

Only **five** accounts achieved certification at T>1. Two of these were mutli day attacks which streamed training examples out of the sandbox through the loss logging and then hard used the hard coded alrogithm.

## What the three leaders did

**Hard's hidden twist was reversing the decimal digits of N in the prompt.** A true modulus of `253` appeared as `352`; x, T, and the answer kept their normal digit order. The recurrence was still forward squaring. Participants had to recover the input convention as well as compute the answer.

All three passed through T=64. We discuss each participant’s latest submission with an exact 100% stored Hard score: CodeReclaimers’ `58311e4a` (29 August), Yash Kant’s `86e9acf2` (26 August), and Sahil Verma’s `9fa00118` (29 August). These are the versions used in our final source reviews and verification runs.

Yash and Sahil included both digit orders among their input interpretations. CodeReclaimers' modulus-mapping search also included digit reversal. Their generalization therefore included identifying the encoding, **not** learning a reversed arithmetic recurrence.

![Three illustrated mechanisms: CodeReclaimers’ computed residue transitions, Yash’s learned digit cells, and Sahil’s selection among supplied programs.](assets/submission-ideas.svg)

**CodeReclaimers selects an arithmetic law and computes its residue transitions.** Learned agreement identifies a candidate law and useful modulus channels. Once the law is selected, the code constructs its transition maps directly, composes them for the requested depth, and combines the resulting remainder constraints into an answer. Before law selection, evaluation uses learned slot maps.

The one-step map is computed with this exact source expression:

```python
(r + D * r * (r - 1) // 2) % m
```

For the selected value `D=2`, this reduces to `r*r % m`. The transition values are therefore calculated from the selected formula; the model does not have to learn a table approximating every squaring result.

For a toy example with `N=35=5×7, x=3`, and `T=2`, the post-selection computation is equivalent to:

```python
D = 2  # the candidate selected through agreement with learned predictions

def transition(r, m):
    return (r + D * r * (r - 1) // 2) % m

r5, r7 = 3, 3
for _ in range(2):
    r5, r7 = transition(r5, 5), transition(r7, 7)
answer = combine_remainders(r5, r7, moduli=[5, 7])  # 11
```

The final remainders are 1 and 4; only 11 among numbers from 0 to 34 satisfies both. The illustration shows these computed transitions and the reconstruction. The actual implementation caches transition powers for faster composition and can refresh stale transition caches during evaluation.

This exact submission scored **100% on Hard**, certified through **T=64 on both seen and unseen modulus identities**, and averaged **99.35% across our ten Easy reruns**. The leaderboard’s older ranked attempt, `d83efe07` from 24 August, trained residue tables using generated arithmetic targets. We inspected that older source for comparison; the worked example and measurements here describe the 29 August version.

**Yash Kant (`86e9acf2`) learned digit operations inside a supplied arithmetic procedure.** Small networks predict multiplication, addition, subtraction, carries, and borrows. A multiplication cell can learn `7 × 8 + carry 3 = 59`: output digit 9 and carry 5.

```python
for _ in range(T):
    squared = square_using_learned_digit_cells(z)
    z = remainder_using_learned_digit_cells(squared, N)
```

The final answer circuit is not trained end to end.

**Sahil Verma (`9fa00118`) selected from supplied exact programs.** The code contains 63,525 polynomial formulas, including squaring. Four input-reading choices produce 254,100 candidates. Training examples eliminate incorrect candidates, learned weights select the formula and output placement.

```python
candidates = keep_programs_matching(supplied_programs, training_examples)
chosen = learn_which_program_to_select(candidates)
answer = execute_exactly(chosen, x, N, T)
```

## The winner

The [written rules](https://github.com/tilde-research/one-layer-deeper#rules) mix restrictions on what must be learned with restrictions on resources and evaluator control.

| Rule excerpt | Effect on the learning claim |
|---|---|
| **7:** “No hard-coded algorithm in the forward pass.” | Sahil selects among supplied exact programs. CodeReclaimers computes the selected arithmetic law directly rather than learning its transition values. |
| **8:** “End-to-end learning only.” | Yash trains components separately from the final answer computation. CodeReclaimers selects and evaluates a supplied arithmetic law. Sahil differentiates selection weights, not the arithmetic.|
| **12/14:** custom losses are allowed; task-specific solvers and hidden training are prohibited. | CodeReclaimers removed arithmetic-generated target losses, but still contains explicit arithmetic and loss-side state fitting. Removing those losses alone does not establish compliance. A custom loss or a call to a component cell does not by itself establish a violation. |
| **9:** “Everything stays on the GPU.” | CodeReclaimers includes CPU fitting and Yash generates targets on CPU. Neither observation establishes that they used an oversized model. |

We intended the CPU restriction to prevent escaping the memory budget. Rule 5 separately caps persistent model state at 500 million elements. The stored Hard result for CodeReclaimers’ 29 August attempt reports 273,410,979 model-state elements (about 273.4 million). The reported counts for Yash and Sahil were 66,137 and 254,149, respectively.

And so with all of this mind, we believe the winner of this competition is [Yash Kant](https://yashkant.github.io/) since his network learns reusable digit operations and composes them across repeated computation. Congrats Yash!



## Honourable Mention

**alirezashirvani-jr (`f5d75083`) reached 10.38% Hard accuracy and 98.15% mean accuracy across our ten Easy reruns.** Among the submissions in our analysis classified as both __Learned and Clear__, it had the highest stored Hard accuracy.

The model represents numbers using periodic features, like positions on clocks with different periods. For smaller values, it learns finite-state transitions; for larger values, it uses learned Fourier mappings to predict the resulting positions. Learned routing chooses which periods to use, and learned votes combine the predicted positions to select an integer answer. At evaluation, it can compose learned operators to handle greater requested depths.

The interesting distinction is that the transition itself is learned. Fixed remainder features and the output decoder provide numerical structure, but do not independently calculate the target squaring operation. An independent expert review classified the submission as learned and Clear, revising the initial static assessment. That is a source-review judgment, not a guarantee of compliance under every interpretation of the rules.

## Techniques
We spent time with codex to read through submissions, reviewing them for rule breaks. Out of the 146 submissions on the final leaderboard 9 were the algorithm descibed on the website hard coded, a further 26 used the algorithm in some way to assist learning, an additional 11 were flagged as rule breaks.

We also pruned through for general classes of techniques people used:

| Technique | How participants used it |
| --- | --- |
| Recurrent computation and shared weights | Many participants tried recurrent architectures, repeatedly applying the same learned block. Some conditioned depth on `T`, learned when to stop, or ran longer at evaluation. adwhit (`14344ca3`) used 40 training iterations and 70 evaluation iterations, differentiating through only the final eight. Chuk Uzowihe (`a4e053c3`) stopped when the state changed little enough, up to 160 evaluation steps. Neither achieved T=1 certification. |
| Structured numerical representations | Organized inputs by decimal place, field, residue, or periodic phase so the network could work with numerical structure. These representations still left a transformation to learn. |
| Attention and workspaces | Used attention to read prompts into latent slots, tapes, or memory states, then refined those states before decoding the answer. |
| Multiple experts and candidate selection | Trained or compared alternative routes, periods, branches, or programs and learned which to use. What the candidates already compute matters when assessing the learning claim. |
| Output alignment and decoding | Aligned answer digits, selected output positions, or combined learned votes over possible integer answers. A decoder can interpret learned predictions without independently calculating the target answer. |
| Auxiliary objectives and curricula | Added reconstruction, intermediate-state supervision, routing losses, or changes in recurrent depth during training. Encoding supplied labels and generating new arithmetic targets are different operations and require different rule assessments. |
| Optimization and training schedules | Combined Adam or AdamW with warmup and decay, often measured against elapsed time. Other submissions used Muon, preconditioning, parameter averaging, or different updates for different parameter groups. |
| Evaluator-controlled batch reuse | Requested additional updates on the current batch through the documented evaluator interface. The evaluator retained control of the training steps. |

## Novel techniques

Here, “novel” means distinctive within the reviewed submissions, not proven new to the research literature. These are ideas worth examining, selected from the source reviews rather than ranked by score. We did not run ablations to establish whether any individual mechanism improved performance.

We searched primary research papers available before each submission, comparing the implemented mechanisms with several candidate relatives. “Closest” is our assessment of technical overlap, not evidence that a participant used or copied that work; the cited papers need not contain the full combination used here.

| Submission | Distinctive idea | What made it interesting | Closest work we could find |
| --- | --- | --- | --- |
| Blake Camp (`0f119b89`) | Matrix memory with learned read/write gates | A shared MLP produces queries, keys, values, and retention gates. Per-head key–value matrices and 32 workspace nodes give repeated computation an explicit learned memory. | [xLSTM’s mLSTM (2024)](https://arxiv.org/abs/2405.04517) is closest to the gated matrix memory, key normalizer, and normalized query read. [Linear Transformers (2020)](https://arxiv.org/abs/2006.16236) matches the positive feature map; [Universal Transformers (2018)](https://arxiv.org/abs/1807.03819) shares updates across depth. Unlike token-by-token mLSTM, this submission aggregates workspace-node writes at each reasoning step and uses different gates. |
| Ayush Nangia (`179288ee`) | Cross-attention into a latent tape | The prompt is read into 44 latent slots, processed by shared recurrent convolutional modules, and read back into answer positions. Input representation and working memory occupy separate spaces. | [Perceiver IO (2021)](https://arxiv.org/abs/2107.14795) closely matches the cross-attention read/process/write structure, while [Neural GPU (2015)](https://arxiv.org/abs/1511.08228) uses shared recurrent convolutions for arithmetic. The submission combines these patterns in a 44-slot tape with numerical place structure and separate learned square/reduction modules. |
| AHappyCPU (`1f7c46f2`) | A learned discrete recurrent state | A codebook turns continuous states into increasingly hard assignments. Prompt information is reinjected after quantization, while training differentiates through a bounded suffix of the rollout. | [DiscoLoop (2026)](https://arxiv.org/abs/2607.00341) combines repeated Transformer passes with a decoded and re-embedded discrete channel. [VQ-VAE (2017)](https://arxiv.org/abs/1711.00937) is closer to the learned codebook and straight-through quantization. This submission adds gradual hardening, prompt reinjection, and gradients restricted to the final part of the rollout. |
| Ethan Davis (`24c58fc1`) | Learned cyclic transport conditioned on the modulus | The modulus representation determines kernels that move information by cyclic offsets along latent tapes. A learned representation of `T` mixes predictions from several depths. | [Neural Turing Machines (2014)](https://arxiv.org/abs/1410.5401) use circular convolution for addressing; [Dynamic Filter Networks (2016)](https://arxiv.org/abs/1605.09673) generate kernels from an input. Here the modulus generates full-tape cyclic transport kernels. [ACT (2016)](https://arxiv.org/abs/1603.08983) is related to mixing recurrent outputs, although this submission selects among fixed computed depths rather than halting adaptively. |
| Aditya Ramabadran (`3a14c748`) | Rereading a prediction before answering | The model embeds its first prediction and makes another attention pass. A learned gate uses changes in confidence and entropy to choose or blend the initial and refined answers. | [Deliberation Networks (2017)](https://papers.neurips.cc/paper_files/paper/2017/hash/c6036a69be21cb660499b75718a3ef24-Abstract.html) and [iterative-refinement decoding (2018)](https://aclanthology.org/D18-1149/) feed a draft into another learned pass. [Adaptive Multi-pass Decoding (2018)](https://aclanthology.org/D18-1048/) learns whether to continue refining. This submission instead uses one extra pass and a learned gate over the initial and refined distributions. |
| Dmp (`5d253b31`) | Soft selection of recurrent depth | One shared linear operator repeatedly updates a 4,096-dimensional state. A learned head chooses a soft mixture of depths; the rollout extends from eight training steps to 64 evaluation steps. | [Adaptive Computation Time (2016)](https://arxiv.org/abs/1603.08983) learns weighted outputs over recurrent steps; [Universal Transformers (2018)](https://arxiv.org/abs/1807.03819) shares a block across depth. Dmp uses a normalized linear operator and predicts the center and sharpness of a mixture over states. It computes every step, so its soft depth selection does not save computation through early halting. |
| Jonathan Whitaker (`e55f8d46`) | Associative memory over a small workspace | Learned output queries retrieve prompt information through cross-attention, and a shared associative cell refines an answer workspace whose size follows the prompt. The requested depth selects the state to decode. | [The Dragon Hatchling (2025)](https://arxiv.org/abs/2509.26507) is closest: the source explicitly labels its cell a simplified BDH update. [Universal Hopfield Networks (2022)](https://proceedings.mlr.press/v162/millidge22a.html) explains the associative-retrieval operations, and [Perceiver (2021)](https://proceedings.mlr.press/v139/jaegle21a.html) is related to learned-query cross-attention. This submission adds residue-aligned answer states and `T`-based selection from its recurrent trajectory. |
| alirezashirvani-jr (`f5d75083`) | Periodic state machines and Fourier operators | Learned transitions, period routing, and output votes combine numerical structure with learned computation. The preceding section describes its 10.38% Hard result and the limits of that result. | [Prime Fourier Embeddings (2026)](https://arxiv.org/abs/2606.23044) is close to selecting periodic residue channels; [FoNE (2025)](https://arxiv.org/abs/2502.09741) encodes and decodes numbers through circular coordinates. [State-Regularized RNNs (2019)](https://proceedings.mlr.press/v97/wang19j.html) offers a learned finite-state precedent. The submission combines these themes with routed period banks, repeated learned operators, and voting over integer answers. |
| Chris Wood (`7f055799`) | A learned distribution over integer programs | Eighty logits parameterize choices for five instruction slots in a 16-operation language. Training optimizes endpoint likelihood from supplied labels; evaluation executes the selected program recurrently. | [TerpreT (2016)](https://arxiv.org/abs/1608.04428) is closest in learning discrete program choices from input/output examples. Its differentiable inference propagates approximate intermediate-state probabilities. This submission instead enumerates exact executions of all five-slot programs, sums their probabilities at the observed endpoints, and executes the most probable program at evaluation. |
| Rohan Sehgal (`3854d99a`) | A population of discrete arithmetic experts | 49,152 three-gate experts learn operand routing, operation choices, integer scales, and biases. Hard choices are used in the forward pass with differentiable approximations for training, and a learned selector chooses an expert. | [Operand Selective Logic Gate Network (2025 manuscript)](https://openreview.net/pdf?id=Wnlo7zeUDY) learns operand routes and gate choices with straight-through decisions. [NALU (2018)](https://arxiv.org/abs/1808.00508) learns arithmetic combinations, while [Differentiable Logic Gate Networks (2022)](https://arxiv.org/abs/2210.08277) learns discrete operators. Here the operations are integer addition/multiplication, with learned scales and biases across 49,152 recurrent experts. |
| Aqua (`55bec2ea`) | An abacus-style workspace with learned routing | Decimal fields occupy 32 slots. Shared steps combine learned pair interactions, place-based routing, an upward parallel scan, and attention, while a time-based curriculum increases the recurrent horizon. | [Abacus Embeddings (McLeish et al., 2024)](https://arxiv.org/abs/2405.17399) combines digit-place alignment, input injection, and looped Transformers. [Neural GPU (2015)](https://arxiv.org/abs/1511.08228) supplies a recurrent arithmetic precedent, and [Blelloch (1990)](https://www.cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html) describes parallel scans. Aqua adds explicit workspace slots, summed-place routing of learned pair interactions, and a learned affine scan. |

For the learned-program entries, the source reviews distinguish a generic instruction set whose composition is learned from a supplied program that already solves the target task. Exact arithmetic primitives alone do not settle that distinction. Across the shortlist, the common research question is how much of the representation, transition, and control flow can be learned while still supporting longer computations.

## What we learnt
Orgainising a model training competition in 2026 is difficult, some users will leave claudex running on their personal H100s for the whole month while others will try their own ideas using the provided rate limited compute. We tried to balance this as best as we could by giving a task which was truely difficult.

We felt the reversing N trick was fair for finding truely valuable solutions, helping to distingush learning from disgusing the algorihm known to be the generating function.

## Next steps
All 15,602 uploads are preserved in the [private GPU MODE dataset on Hugging Face](https://huggingface.co/datasets/GPUMODE/one-layer-deeper-submissions) for further analysis.

We have also updated the public github to contain the hard dataset.
