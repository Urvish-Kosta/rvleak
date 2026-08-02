# Security policy

## Scope

rvleak is a research and educational simulator. It analyses toy victim programs
that it generates itself. It does not process untrusted input, does not open
network connections, and does not execute code outside its own interpreter.

## What this tool does and does not tell you

**It does not evaluate physical devices.** Results describe a model: an in-order
RV32IM core with a configurable cache and branch predictor, and a first-order
Hamming-weight/Hamming-distance activity model with additive Gaussian noise.
Correspondence with any real silicon is unvalidated.

Do not use a negative result from rvleak as evidence that a deployed
implementation is side-channel resistant. A negative result rules out
first-order HW/HD leakage and structural timing leakage on the modelled
microarchitecture, and nothing else. Second-order leakage, glitch power,
coupling, speculative execution, and every physical effect are out of scope.

A positive result is more informative: if a victim leaks under this model, it
will very likely leak on hardware too, because the model is a simplification in
the defender's favour.

## Responsible use

The included attacks target victim programs shipped with this repository, using
a seeded random permutation as a stand-in lookup table rather than any real
cipher's S-box. Applying the same techniques to systems you do not own or have
permission to test may be unlawful.

## Reporting a vulnerability

For defects in this tool — in particular, any case where the analysis reports a
false negative — please open an issue with a reproducing script and the seed
used. False negatives are the most serious class of bug in a leakage detector
and are treated as such.

Do not report vulnerabilities in third-party software here.
