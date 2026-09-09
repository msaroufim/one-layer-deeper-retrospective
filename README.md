# We still don't know if the models want to learn

We received **15,602 accepted uploads from 206 GitHub accounts** to One Layer Deeper.

## The challenge

The public problem is repeated modular squaring:

```python
z = x % N
for _ in range(T):
    z = z * z % N
```

Our [launch post](https://blog.tilderesearch.com/blog/one-layer-deeper) asked participants to design architectures, optimizers, and losses that learn this computation and use greater depth at test time. Loops were allowed; supplying an arithmetic solver or generating additional training targets with one was prohibited.

Hard allowed one H100, one hour of training, 30 minutes of evaluation, and 500 million model-state elements, including buffers. Depth was unrestricted. Certification required every example correct at consecutive rungs: T=1, 2, 4, 8, 16, 32, 64.

Only **18 Hard uploads from five accounts** certified any depth on seen-modulus tests: 16 reached T=64, one T=32, and one T=8. Two accounts had excluded entries containing code that encoded hidden training inputs and labels into reported losses, violating the metric-recorder rule. The other 143 ranked accounts failed T=1.

Thanks to az and apaz for submissions that helped us refine the task and rules.

## What the three leaders did

**Hard reversed the decimal digits of N in the prompt.** A true modulus of `253` appeared as `352`; x, T, and the answer kept their normal digit order. The recurrence was still forward squaring. Yash and Sahil considered both operand digit orders; CodeReclaimers' modulus-mapping search included reversal.

We discuss each leader's latest archived Hard upload with 100% aggregate accuracy: CodeReclaimers' `58311e4a`, Yash Kant's `86e9acf2`, and Sahil Verma's `9fa00118`. All certified T=64 on both seen and unseen moduli. These are stored results; we did not rerun these models.

![Three illustrated mechanisms: computed residue transitions, learned digit cells, and selection among supplied programs.](assets/submission-ideas.svg)

**CodeReclaimers (`58311e4a`) selects an arithmetic law, then computes its remainder tables.** It compares candidate laws with learned predictions. After selecting a law, it constructs the transitions directly with this exact expression:

```python
(r + D * r * (r - 1) // 2) % m
```

For D=2, this is `r*r % m`. The code caches maps for 1, 2, 4, and more transitions, then composes them using the bits of T. Separate small-modulus results constrain the final integer answer, as the illustration shows.

Before law selection, it uses learned maps. This August 29 version differs from the older ranked upload, `d83efe07`, which trained tables using generated arithmetic targets. The later version computes the selected law instead of learning every transition value.

**Yash Kant (`86e9acf2`) learned digit operations inside a supplied calculator.** Small networks predict multiplication, addition, subtraction, carries, and borrows. A multiplication cell can learn `7 × 8 + carry 3 = 59`: output digit 9 and carry 5.

The supplied procedure repeats those learned operations. In pseudocode:

```python
for _ in range(T):
    squared = square_using_learned_digit_cells(z)
    z = remainder_using_learned_digit_cells(squared, N)
```

The digit cells are trained from arithmetic targets. The complete answer circuit is not trained end to end.

**Sahil Verma (`9fa00118`) selected from supplied exact programs.** The code contains 63,525 polynomial formulas, including squaring. Four input-reading choices produce 254,100 candidates. Examples eliminate incorrect candidates; learned weights select the formula and output placement.

```python
candidates = keep_programs_matching(supplied_programs, training_examples)
chosen = learn_which_program_to_select(candidates)
answer = execute_exactly(chosen, x, N, T)
```

## The winner

The [written rules](https://github.com/tilde-research/one-layer-deeper#rules) mix restrictions on learning, resources, and evaluator control.

| Rule | Effect on the learning claim |
|---|---|
| **7:** “No hard-coded algorithm in the forward pass.” | Sahil selects supplied exact programs. CodeReclaimers directly computes its selected arithmetic law. |
| **8:** “End-to-end learning only.” | Yash trains components separately from the full answer circuit. CodeReclaimers and Sahil select and execute supplied arithmetic laws. |
| **12/14:** custom losses allowed; solvers and hidden training prohibited | Whether arithmetic-generated targets count as a solver needs clarification. Custom losses or calls to component cells alone do not establish a violation. |
| **9:** “Everything stays on the GPU.” | CPU fitting or target generation conflicts with the wording, but does not establish use of an oversized model. |

We intended the CPU restriction to prevent escaping the memory budget. The reported model-state counts were 273,410,979 for the later CodeReclaimers upload, 66,137 for Yash, and 254,149 for Sahil, all below the 500-million ceiling.

**Our winner is [Yash Kant](https://yashkant.github.io/).** His network learns reusable digit operations and composes them across repeated computation. The calculator's wiring is supplied, so this is a narrower result than learning the entire algorithm. Congrats Yash!

## Honourable mention

**alirezashirvani-jr (`f5d75083`) reached 10.38% aggregate Hard accuracy.** It represents numbers as positions on clocks with different periods, learns transitions between those positions, and combines learned votes into an integer answer. Its larger-number branch uses learned Fourier operators that can be composed at evaluation.

The transitions are learned rather than computed by an explicit squaring formula. It failed T=1 certification.

## Interesting ideas that didn't quite work

None of these submissions passed all the single-step (T=1) tests on Hard.

| Submission | Implemented idea |
|---|---|
| AHappyCPU (`1f7c46f2`) | Gradually replaces continuous recurrent states with discrete choices from a learned 64-entry codebook. |
| Ethan Davis (`24c58fc1`) | Uses N to generate convolution kernels that move information around a cyclic tape, making the update operation depend on the modulus. |
| Chris Wood (`7f055799`) | Learns a five-instruction program from 16 supplied integer operations by comparing program outputs with labels. The interpreter is fixed; instruction composition is learned. |
| Rohan Sehgal (`3854d99a`) | Learns wiring and add/multiply choices for three parallel gates per expert, with exact integer forward values and approximate gradients. Modular reduction is supplied. |
| Aqua (`55bec2ea`) | Sends learned digit-pair interactions to fixed decimal places `i+j`, then combines affine updates with a parallel scan. The arithmetic layout is supplied. |

## Next steps

Could we learn the calculator's wiring as well as its digit operations? Train the small operations first, then teach a controller to compose them from execution traces. The question is whether those parts transfer to unfamiliar algorithms and longer executions.

All 15,602 uploads are preserved in the [private GPU MODE dataset on Hugging Face](https://huggingface.co/datasets/GPUMODE/one-layer-deeper-submissions) for further analysis.
