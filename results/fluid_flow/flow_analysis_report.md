# Tensor DMD Analysis Diagnostic Report

## System Overview
- Tensor dimensions: (80, 50, 50)
- Number of time steps: 80
- Time step (dt): 0.1
- Tucker decomposition rank: (None, 20, 20)
- DMD truncation rank: 10

## System Stability
- Spectral radius: 1.0003
- System stability: Unstable
  ⚠️ The system has growing modes which indicate instability or transient growth.

## Reconstruction Accuracy
- Relative error: 4.0638e-01
- Maximum absolute error: 1.2998e+00
- Mean absolute error: 7.8808e-02

## Most Significant Modes
|    |   Mode |   Significance |   Normalized_Amplitude |   Magnitude |
|---:|-------:|---------------:|-----------------------:|------------:|
|  0 |      0 |      1.0003    |              1         |    1.0003   |
|  1 |      1 |      0.168557  |              0.168825  |    0.998415 |
|  2 |      2 |      0.116339  |              0.116524  |    0.998415 |
|  6 |      6 |      0.0915407 |              0.0920231 |    0.994758 |
|  5 |      5 |      0.0908211 |              0.0912997 |    0.994758 |

## Mode Frequency Analysis
|    |   Mode |   Frequency |   Growth_Rate |   Magnitude |   Amplitude |   Normalized_Amplitude |
|---:|-------:|------------:|--------------:|------------:|------------:|-----------------------:|
|  0 |      0 |     0       |     0.0030009 |    1.0003   |   15.9903   |              0.550612  |
|  1 |      1 |     4.15215 |    -0.0158667 |    0.998415 |    2.69956  |              0.0929571 |
|  2 |      2 |    -4.15215 |    -0.0158667 |    0.998415 |    1.86325  |              0.0641594 |
|  6 |      6 |    -3.6914  |    -0.0525604 |    0.994758 |    1.47148  |              0.050669  |
|  5 |      5 |     3.6914  |    -0.0525604 |    0.994758 |    1.45991  |              0.0502707 |
|  9 |      9 |     2.84227 |    -0.10036   |    0.990014 |    1.35955  |              0.0468147 |
|  8 |      8 |     2.35292 |    -0.0616195 |    0.993857 |    1.26139  |              0.0434349 |
|  4 |      4 |     2.70719 |    -0.0402219 |    0.995986 |    1.05624  |              0.0363707 |
|  7 |      7 |    -2.35292 |    -0.0616195 |    0.993857 |    0.959596 |              0.0330428 |
|  3 |      3 |    -2.70719 |    -0.0402219 |    0.995986 |    0.919689 |              0.0316687 |

## Recommendations
- Investigate growing modes to understand system instabilities.
- For prediction tasks, be cautious about extrapolating too far into the future.
- Focus analysis on the top 5 modes by significance for key dynamics.

## Physical Interpretation
- DMD modes represent coherent structures in the tensor data with specific frequencies and growth/decay rates.
- Modes with eigenvalues close to the unit circle represent persistent dynamics.
- Modes with eigenvalues inside the unit circle represent decaying dynamics.
- Modes with eigenvalues outside the unit circle represent growing dynamics.
