**signaldecoding**
-------------------
Scripts used for Sig2Fate framework and the manuscript "Interpretable decoding of cell fate from a snapshot of combinatorial signaling".

Quantification of image data was performed using MATLAB version 2023b; the processed and cleaned data used for Sig2Fate is stored in Dryad at https://doi.org/10.5061/dryad.rbnzs7ht9 (private for the time being.)

Analysis of quantified data and the resulting figures were produced in Python. Under the "2D_gastruloids_V5" folder, "fns_NN" contains the framework that implements Sig2Fate as a 
Variational Information Bottleneck (VAE), as well as Variational Autoencoder (VAE) to determine the compressibility of our signaling space. "IIF_analysis" contains all of the functions used for analyzing the quantified data 
and the outcomes of Sig2Fate predictions. 

The script "fig2n.ipynb" trains Sig2Fate on our main iterative immunofluorescence data and produces the panels for Figure 2, Figure S2, and Figure S3. "Fig3n_6D.ipynb" computes the mutual informations 
between individual signals/signaling combinations, constructs the 2D signaling code, and produces the panels for Figure 3, Figure S4, Figure S5, and Figure S7. "toy_model.ipynb" applies the Sig2Fate framework
for a toy dataset in which fate outcomes depends on the difference between two signals, and produces the panels for Figure S6. "Fig4n_6D.ipynb" tests the generalizability of the B50-trained signal-to-fate map
on different initial doses of BMP, and produces the panels for Figure 4 and Figure S8. "Fig5n_6D.ipynb" tests the generalizability of the B50-trained signal-to-fate map on perturbations of other signal pathways,
implements zero-shot predictions by projecting onto targeting signaling states in the control, and produces the panels for Figure 5 and Figure S9.
