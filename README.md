# CSPC - Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

SETUP

Create the environment for a given lab:

conda env create -f PW<n>/Lab <X>/environment.yml
conda activate cspc

PW1 - Lab A: Reproducible Foundations

What I built:

- Created a reproducible Conda environment for the CSPC course.
- Used Git and GitHub for version control and implemented a radioactive decay simulation with Python and NumPy.

Speed comparison (loop vs NumPy):

- loop: 1.9980 s
- numpy: 0.0002 s
- speed-up: 10193.52 x faster

Tests: all passing? yes

Conclusion:

- I learned how to create and manage a Conda environment and use Git for version control.
- I implemented and tested a radioactive decay simulation and compared a pure Python loop with a NumPy implementation.
- The NumPy implementation was much faster in this benchmark, showing the benefit of vectorized numerical operations.