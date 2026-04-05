---
title: "Application of the self-consistent Hartree-Fock method"
date: 2025-01-28
categories: ["math"]
order: 1
---

This is a summary of the computational method I used for my [master's thesis](/Files/master_thesis.pdf) [1]. I will focus on the Hartree-Fock approximation.

## Model

The problem we were trying to attack was the correct modeling of the Majorana zero modes in a superconducting nanowire with a quantum dot attached at one of the ends of the wire. The Hamiltonian describing the system is

$$
\begin{align*}
H &= \sum_{j=1}^{N-1} \left( -t c_{j+1}^\dagger c_j + \Delta c_{j+1}^\dagger c_j^\dagger + \text{H.c.} \right) - \mu \sum_{j=1}^{N} c_j^\dagger c_j \\
 &+ \epsilon_d d^\dagger d - t' \left( d^\dagger c_1 + \text{H.c.} \right) \\
 &+ V \left( n_d - \frac{1}{2} \right) \left( n_1 - \frac{1}{2} \right)
\end{align*}
$$

The line is the Hamiltonian for the Kitaev chain, the second for the quantum dot and its hopping with the first site of the nanowire, and the third one accounts for the Coulomb repulsion between the quantum dot and the first site.

## Hartree-Fock
Applying the [unrestrictive Hartree-Fock approximation](https://journals.aps.org/pr/abstract/10.1103/PhysRev.102.1303) to the repulsion term:

$$
\begin{align*}
n_d n_1 &\simeq \langle n_d \rangle n_1 + n_d \langle n_1 \rangle - \langle n_d \rangle \langle n_1 \rangle \\
&\quad - \langle d^\dagger c_1 \rangle c_1^\dagger d - d^\dagger c_1 \langle c_1^\dagger d \rangle + \langle d^\dagger c_1 \rangle \langle c_1^\dagger d \rangle \\
&\quad + \langle d^\dagger c_1^\dagger \rangle c_1 d + d^\dagger c_1^\dagger \langle c_1 d \rangle - \langle d^\dagger c_1^\dagger \rangle \langle c_1 d \rangle.
\end{align*}
$$

## Mapping

There is a very useful technique that allows us to go from fermionic operators to kets and bras, making the computation of the Hamiltonian matrix elements easy. After performing the mapping we end up with

$$
    \begin{matrix}
    \bra{d,c}\tilde H\ket{d,c}= \epsilon_d + V\langle n_1\rangle & \bra{d,a}\tilde H\ket{d,a}= -\epsilon_d - V\langle n_1\rangle\\
     \bra{1,c}\tilde H\ket{d,c} = -t' - V\langle c^\dagger_d c_1\rangle & \bra{1,a}\tilde H\ket{d,a} = t' + V\langle c^\dagger_d c_1\rangle\\
      \bra{d,c}\tilde H\ket{1,a} = -V\langle c^\dagger_d c^\dagger_1 \rangle & \bra{d,a}\tilde H\ket{1,c} = V\langle c^\dagger_d c^\dagger_1 \rangle\\
      \bra{1,c}\tilde H\ket{1,c} = ...+ V\langle n_d\rangle & \bra{1,a}\tilde H\ket{1,a} = ...- V\langle n_d\rangle
    \end{matrix}
$$

where $j$ is the site index, $c$ stands for creation and $a$ for annihilation. Then, using some techniques from superconductivity, we can relate the expectation values to the eigenvalues and eigenvectors of the diagonalized Hamiltonian defined above.

## Self-consistency

Self-consistency means that we take the expectation values and plug them again into the model until convergence is reached. The process would look like

- Set initial values for the EVs(expectation values)
- Add them to the Hamiltonian
- Compute the eigenvalues and eigenvectors
- Substract them from the Hamiltonian
- Update the EV's values based on the eigenvalues and eigenvectors
- Go back to step 2

## Code

`ini!` is the function where you define your Hamiltonian `H` as a matrix after the mapping. Usually the code below is inside another loop going through several values of the occupation energy of some site, in this case it would be the one of the quantum dot so that at the end we end up with a plot of `expectation_value vs occupation_energy_QD`.

```julia
# ITERATIONS FOR CONVERGE
for x in 1:iterations
    # SET EV values to the Hamiltonian
    ini!(H,initial_ev[1],initial_ev[2],(initial_ev[3]-0.5),(initial_ev[4]-0.5),V)
    # COMPUTE EIGENVALUES AND EIGENVECTORS
    espec=eigvals(H)
    espec_vec=eigvecs(H)
    if x==iterations
        matrix_plot=deepcopy(H)
        push!(store_x,values_ed)
        push!(store_y,initial_ev[3])        
    end
    # SUBSTRACT EV VALUES FOR THE NEXT UPDATE
    ini!(H,-initial_ev[1],-initial_ev[2],-(initial_ev[3]-0.5),-(initial_ev[4]-0.5),V)
    ev=[0.0,0.0,0.0,0.0]
    # COMPUTE THE NEW EV VALUES
    for i in 1:N
        if espec[i]<0.
            if abs(espec[i])<10^(-8)
                ev[1] += conj(espec_vec[1,i]) * espec_vec[4,i]*(1/2) # < c+d c+1 >
                ev[2] += conj(espec_vec[1,i]) * espec_vec[3,i]*(1/2) # < c+d c_1 >
                ev[3] += conj(espec_vec[1,i]) * espec_vec[1,i]*(1/2) # < c+d c_d >
                ev[4] += conj(espec_vec[3,i]) * espec_vec[3,i]*(1/2) # < c+1 c_1 >
            else 
                ev[1] += conj(espec_vec[1,i]) * espec_vec[4,i]
                ev[2] += conj(espec_vec[1,i]) * espec_vec[3,i]
                ev[3] += conj(espec_vec[1,i]) * espec_vec[1,i]
                ev[4] += conj(espec_vec[3,i]) * espec_vec[3,i]
            end
        end
    end
    global initial_ev = deepcopy(ev)
end
```
## References
[1] K. Takagui and A. A. Aligia, "Effect of interatomic repulsion on Majorana zero modes in a coupled quantum-dot--superconducting-nanowire hybrid system," Phys. Rev. B 109, 075416 (2024), https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.075416
