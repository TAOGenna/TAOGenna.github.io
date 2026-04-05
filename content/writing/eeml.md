---
title: "Advancements in Representing Atomic Environments for Machine Learning Interatomic Potentials"
date: 2025-04-12
categories: ["physics"]
---

<div style="text-align: center;">
    Extended Abstract
</div>

> Accurate and efficient interatomic potentials are crucial for simulating materials at the atomic level, and machine learning interatomic potentials (MLIPs) have emerged as a powerful alternative to traditional methods. A central aspect of MLIP development is the representation of local atomic environments, which significantly impacts the accuracy, transferability, and computational cost of the resulting potentials. This extended abstract reviews recent advancements in representing atomic environments for MLIPs. 

# Introduction 

This work surveys the development of atomic environment representations for machine learning interatomic potentials (MLIPs), crucial for accurate force fields. We trace the evolution from Behler-Parrinello's symmetry functions through general descriptors like SOAP to the recent Cartesian Atomic Cluster Expansion (CACE). This progression highlights a shift towards systematic, symmetry-adapted representations, addressing expressiveness, efficiency, and physical fidelity by increasingly unifying completeness, invariance, and learnability. We conclude with a reflection on the current state of MLIP design.

# Reviewed Papers 

## Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces [Behler and Parinello, 2009]

Behler and Parrinello (BP) proposed a neural network-based approach to approximate potential energy surfaces (PES) with near-DFT accuracy at a fraction of the computational cost, enabling simulations of large systems over small timescales.

The method is based on atomic energy decomposition, where the total energy $E$ of a system is expressed as a sum of atomic contributions $E=\sum_i E_i$. Each atomic energy $E_i$ depends solely on the local chemical environment of atom $i$, in the spirit of empirical interatomic potentials.

In a first stage, we transform raw Cartesian coordinates corresponding to atom positions into a more suitable representation for a neural network. BP introduce a new type of symmetry function to describe the energetically relevant local environment of each atom. These symmetry functions must satisfy several crucial criteria such as uniqueness, rotational invariance, and coordination number independence. The way they are constructed is by taking positions of all neighboring atoms within a defined cutoff radius $R_c$ using a cutoff function $f_c(R_{ij})$. Finally, for each atom $i$, a vector of symmetry function values $\{G^i\}$ is generated, which encapsulates the essential features of its local atomic environment in a physically meaningful and invariant way.

Now, how are these symmetry function vectors used to predict the atomic energy contributions? For each atom $i$, there is a neural network, refered to as a "subnet" $S_i$. The input to this subnet is the set of symmetry function values $\{G^i\}$ calculated for that atom. A crucial aspect is that all these subnets $S_i$ have the same neural network architecture (i.e., the same number of hidden layers and nodes in each layer). More importantly, all the subnets share the same set of weight parameters after the training process. Then, the neural net outputs a single value, which represents the energy contribution $E_i$ of atom $i$ to the total energy.

## On Representing Chemical Environments [Bartok et al., 2013]

While Behler and Parinello modeled the local chemical environment of an atom using symmetry functions, Bartók et al. step back and ask: what are the general requirements for a good representation of these local atomic environments? According to them, a suitable representation or descriptor must be differentiable with respect to the movement of atoms, which is essential for calculating forces needed in molecular dynamics simulations; invariant to symmetries—specifically, rotation, reflection, translation of the entire system, and permutation (swapping) of atoms of the same element—to ensure that the energy prediction doesn't change simply because we've rotated our system in space or reordered identical atoms in our description; and faithful (ideally complete), meaning the descriptor should uniquely determine the atomic environment up to these symmetries.

Bartók et al. examined several families of descriptors commonly used to represent atomic environments, going beyond the specific symmetry functions introduced by BP. These families included bond-order parameters, which were shown to be related to the SO(3) power spectrum and a subset of the more general SO(3) bispectrum. The paper also explored the SO(4) power spectrum and bispectrum as a means to inherently incorporate radial information. A key methodology employed to assess these descriptors was numerical reconstruction experiments, where the ability to recover a known atomic environment from its descriptor, after a perturbation and subsequent optimization, served as a measure of the descriptor's faithfulness. A significant finding across these families was a tendency for their faithfulness to diminish as the number of neighboring atoms increased for a descriptor of a fixed size (number of components). 

In contrast to the explicit construction of descriptor vectors, Bartók et al. introduced the Smooth Overlap of Atomic Positions (SOAP) as an alternative paradigm that directly defines a similarity measure (a kernel) between two atomic environments. SOAP achieves this by calculating the overlap of atomic neighbor density functions, typically modeled as sums of Gaussians centered on each neighboring atom. 

## Cartesian Atomic Cluster Expansion (CACE) for Efficient MLIPs [Bingquing Cheng, 2024]

Cheng introduces the Cartesian Atomic Cluster Expansion (CACE), a framework that reformulates the widely-used Atomic Cluster Expansion (ACE) methodology entirely in Cartesian coordinates. Traditional MLIP frameworks—such as ACE and those employing equivariant message-passing neural networks—typically express angular dependencies using spherical harmonics and impose rotational symmetry via Clebsch-Gordan coupling. In contrast, CACE constructs rotationally invariant features directly through symmetrized polynomial functions of Cartesian vectors, eliminating the need for spherical harmonics and Clebsch-Gordan coefficients while retaining mathematical equivalence.

ACE provides a systematic and complete basis for representing the potential energy of atomic systems, decomposing the local energy of an atom into contributions from its chemical environment through a body-ordered expansion: 

$$
E_i = V^{(1)}(\mathbf{r}_i) + \frac{1}{2} \sum_{j} V^{(2)}(\mathbf{r}_i, \mathbf{r}_j) + \frac{1}{6} \sum_{j,k} V^{(3)}(\mathbf{r}_i, \mathbf{r}_j, \mathbf{r}_k) + \frac{1}{24} \sum_{j,k,l} V^{(4)}(\mathbf{r}_i, \mathbf{r}_j, \mathbf{r}_k, \mathbf{r}_l) + \cdots
$$ 

CACE preserves this hierarchy but constructs the basis using a direct polynomial expansion in Cartesian components, followed by a rigorous symmetrization procedure to ensure rotational invariance. This yields a complete and minimal set of invariant features that are polynomially independent, in contrast to traditional ACE implementations, which may include redundant terms due to overcomplete bases.

Beyond its novel basis construction, CACE integrates several modern ML innovations, including learnable element embeddings, flexible radial channel parametrizations, and an optional message-passing mechanism. These design choices enable CACE to achieve high accuracy, robustness, and transferability across a range of material systems, including bulk water, small organic molecules, and high-entropy alloys.

## Additional Interpretations and Insights 

While symmetry functions and atomic cluster expansions provide explicit routes to invariant representations, a deeper insight lies in recognizing that these methods implicitly construct a basis in a space of geometric invariants. Behler-Parrinello's symmetry functions, by encoding local atomic environments through radial and angular terms with specific functional forms, effectively define a non-orthogonal basis in this invariant space. ACE and CACE, with their systematic body-order expansion, aim for a more complete and polynomially independent basis. The crucial non-triviality is that the choice of this implicit geometric algebra—whether through predefined functions or a more systematic polynomial basis—strongly influences the expressiveness and the inductive bias of the resulting potential. Modern approaches, especially those incorporating message passing, can be viewed as learning to navigate and utilize this high-dimensional invariant space more effectively, adapting the effective basis functions through learnable parameters rather than relying solely on pre-engineered ones.

## Conclusions 

The development of effective representations for atomic environments is central to advancing machine learning interatomic potentials (MLIPs). This extended abstract outlines the evolution of the field, tracing a trajectory from early empirical descriptors such as the Behler-Parrinello symmetry functions to more theoretically grounded and systematically constructed representations like SOAP and CACE. We have also discussed the limitations of each approach and how subsequent work has addressed these shortcomings. Each successive development has contributed to improving the physical accuracy, completeness, and computational efficiency of MLIPs.

# References

[1] B. Cheng, "Cartesian atomic cluster expansion for machine learning interatomic potentials," *npj Computational Materials*, vol. 10, no. 1, p. 157, 2024. DOI: [10.1038/s41524-024-01332-4](https://doi.org/10.1038/s41524-024-01332-4).

[2] J. Behler and M. Parrinello, "Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces," *Phys. Rev. Lett.*, vol. 98, no. 14, p. 146401, Apr. 2007. DOI: [10.1103/PhysRevLett.98.146401](https://link.aps.org/doi/10.1103/PhysRevLett.98.146401).

[3] A.P. Bartók, R. Kondor, and G. Csányi, "On representing chemical environments," *Phys. Rev. B*, vol. 87, no. 18, p. 184115, May 2013. DOI: [10.1103/PhysRevB.87.184115](https://link.aps.org/doi/10.1103/PhysRevB.87.184115).
