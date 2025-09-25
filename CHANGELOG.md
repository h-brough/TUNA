# Changelog

## TUNA 0.8.0 — 25/09/2025

### Added

- Restricted MP4, MP4(SDQ) and MP4(DQ) energy
- Spatial orbital (SCS-)MP2 energy and density and (SCS-)MP3 energy, speeds up calculations on RHF references
- Core electrons can now be frozen for correlated calculations using the `FREEZECORE` keyword, with optional number of orbitals to freeze
- Amplitude convergence for coupled cluster calculations via `AMPCONV` keyword
- Improved (non-arbitrary) dynamic damping scheme, now uses Zerner-Hehenberger scheme based on Mulliken gross populations
- New `DAMP` keyword, followed by a number between 0 and 1 to define a static damping parameter
- Custom basis sets using `CUSTOM` basis with `BASIS [filename.tuna]` keyword
- Extrapolation to complete basis set limit compatible with all calculations via `EXTRAPOLATE` keyword
- New basis sets: STO-2G, STO-4G, STO-5G, 6-31+G*, 6-31++G*, 6-31+G**, 6-31++G**, 6-311+G*, 6-311++G*, 6-311+G**, 6-311++G**
- New basis sets: def2-SVP, def2-SVPD, def2-TZVP, def2-TZVPD, def2-TZVPP, def2-TZVPPD, def2-QZVP, def2-QZVPD, def2-QZVPP, def2-QZVPPD
- New basis sets: 6-31G(d,p), 6-31G(2df,p), 6-31G(3df,3pd), 6-311G(d,p), 6-311G(2df,2pd), 6-311+G(2d,p), 6-311++G(2d,2p), 6-311++G(3df,3pd)
- New basis sets: pc-0, pc-1, pc-2, pc-3, pc-4, aug-pc-0, aug-pc-1, aug-pc-2, aug-pc-3, aug-pc-4
- New basis sets: pcseg-0, pcseg-1, pcseg-2, pcseg-3, pcseg-4, aug-pcseg-0, aug-pcseg-1, aug-pcseg-2, aug-pcseg-3, aug-pcseg-4
- New basis sets: aug-cc-pVDZ, aug-cc-pVTZ, aug-cc-pVQZ, aug-cc-pV5Z, aug-cc-pV6Z, d-aug-cc-pVDZ, d-aug-cc-pVTZ, d-aug-cc-pVQZ, d-aug-cc-pV5Z, d-aug-cc-pV6Z
- New basis sets: cc-pCVDZ, cc-pCVTZ, cc-pCVQZ, cc-pCV5Z, aug-cc-pCVDZ, aug-cc-pCVTZ, aug-cc-pCVQZ, aug-cc-pCV5Z 
- New basis sets: cc-pwCVDZ, cc-pwCVTZ, cc-pwCVQZ, cc-pwCV5Z, aug-cc-pwCVDZ, aug-cc-pwCVTZ, aug-cc-pwCVQZ, aug-cc-pwCV5Z 
- New basis sets: ano-pVDZ, ano-pVTZ, ano-pVQZ, ano-pV5Z, aug-ano-pVDZ, aug-ano-pVTZ, aug-ano-pVQZ, aug-ano-pV5Z
- New basis sets: 6-31G(d), 6-31G(d,p), 6-31+G(d), 6-31+G(d,p), 6-31++G(d), 6-31++G(d,p), 6-311G(d), 6-311G(d,p), 6-311+G(d), 6-311+G(d,p), 6-311++G(d), 6-311++G(d,p)

### Changed

- Dunning basis sets have been rotated and had redundant functions removed, increasing computational efficiency
- Decimals points now align nicely for all calculations with all elements
- Removed repeated logging of transforming integrals in MP2
- Nicer logging at the start of a geometry optimisation
- Basis information is now printed with the additional print keyword, `P`
- Design of logging is now more consistent
- Increased default orbital-optimised MP2 iterations to 30 (originally 20)
- Improved PyPI publishing page, README and manual
- For rotated guesses, the degrees by which the HOMO and LUMO are rotated is now printed
- Only the first 10 virtual orbitals are now printed when `P` is used

### Fixed

- Distributions for MacOS and Linux now work correctly
- Python 3.13 and Python 3.14 are now supported in addition to Python 3.12
- Frequency calculations on a fixed geometry were not working at all
- Ghost atoms were not working correctly under some circumstances
- Convergence acceleration logging was sometimes contradictory

<br>

## TUNA 0.7.0 — 27/08/2025

### Added

- Energy and linearised density from CEPA0, LCCD, LCCSD, CCD, CCSD, CCSD(T) and CCSDT
- Convergence control for coupled cluster calculations with damping via `CCDAMP` keyword (default off)
- Convergence control for coupled cluster calculations and DIIS via `DIIS` keyword (default on) 
- Support for atoms beyond the first period — atoms up to argon are now implemented
- Cython-based Gaussian integral engine via the McMurchie-Davidson algorithm allows higher angular momentum orbitals
- New basis sets: 6-31G*, 6-31G**, 6-311G*, 6-311G**, cc-pVDZ, cc-pVTZ, cc-pVQZ, cc-pV5Z, cc-pV6Z
- Largest coupled cluster amplitudes are found and printed with the T1 diagnostic
- Keyword `CCCONV` for coupled cluster convergence control, `CCMAXITER` for maximum iterations
- Keyword `NOSINGLES` to turn off single excitations in coupled cluster
- Population analysis, dipole moment, natural orbitals using coupled cluster linearised density matrices
- Keyword for changing the number of DIIS matrices to store (for SCF and coupled cluster), `DIIS [Num. Matrices]`
- Keywords to turn on and off natural orbital calculations, `NATORBS` and `NONATORBS`
- Optional natural orbital calculation for UHF
- Keywords for custom atomic masses in AMU, `M1` and `M2`
- The overlap matrix eigenvalues are now checked, use the `STHRESH` keyword to change the threshold
- Warning message for small overlap matrix eigenvalues that are close to the threshold
- Redesigned plotting functions in TUNA, various methods can be plotted on the same axes with the `ADDPLOT` keyword
- Keywords `SAVEPLOT` and `DELPLOT` to save and delete plots
- Dashed lines on plot via `DASH` and dotted lines via `DOT`
- Option to change colour in `SCANPLOT` with `RED`, `BLUE`, `GREEN`, `CYAN`, `MAGENTA`, `YELLOW`, `BLACK`, `WHITE` keywords

### Changed

- By default, DIIS now remembers 6 matrices instead of 10
- The number of MD steps, `MDNUMBER` or `NUM`, now defaults to 50 rather than 30
- Orbital-optimised MP2 can now be accessed via `(U)OOMP2` in addition to `(U)OMP2`
- TUNA now prints the number of minutes, in addition to the number of seconds, for long calculations
- Removed `DENSPLOT` and `SPINDENSPLOT` keywords, and the ability to generate a three-dimensional electron density plot
- Default atomic masses slightly altered
- Calculations with zero electrons now exit with error
- Molecule and basis information now includes number of occupied and virtual orbitals
- Condensed additional printing of molecular orbitals and eigenvalues for UHF
- TUNA now exits gracefully if interrupted by the user with Ctrl+C
- New tuna_molecule and tuna_cc modules; removed tuna_plot module; renamed tuna_mpn into tuna_mp
- Information is now printed during the spin-orbital transformation for post-Hartree-Fock
- Keywords `SCANNUMBER` and `MDNUMBER` replaced by simpler keyword `NUM`
- Keywords `SCANSTEP` and `TIMESTEP` replaced by simpler keyword `STEP`
- Major code improvements and restructuring

### Fixed

- Fixed error handling of requesting RHF calculation on a molecule with an odd number of electrons
- You can no longer put more electrons in a molecule than the number of spin-orbitals
- Incorrect unit conversions in `SCANSTEP`
- Incorrect unit conversions in `SCANPLOT`
- The `GEOMMAXITER` keyword was not working correctly
- The SCF convergence keyword was being overridden if an optimisation is requested 
- The geometry convergence keyword was being overridden if an subsequent frequency calculation is requested 

<br>

## TUNA 0.6.1 — 26/01/2025

### Added

- Keyword for splotting the spin density, `SPINDENSPLOT`
- Virial ratio is calculated and printed, which indicates the proximity to an optimised geometry
- Degenerate excited states are now grouped and averaged before printing
- The singlet or triplet character of excited states is now printed for RHF references

### Changed

- The default SCF convergence criteria for CIS calculations is now `TIGHT`
- Threshold for CIS contributions decreased from 5% to 1%
- Removed printing weights for RHF references, as these are calculated in a spin-orbital basis

### Fixed

- Requested orbital rotation with a tiny basis no longer causes a crash
- Electron affinity calculation was crashing when no virtual orbitals were present
- Excited state calculations were crashing when no virtual orbitals were present
- Spin density matrix for one-electron systems was calculated incorrectly
- Error handling for non-existent root in CIS calculations

<br>

## TUNA 0.6.0 — 11/01/2025

### Added

- Transition intensities for harmonic frequency calculations
- Excited state energy and density by configuration interaction singles, `CIS`
- Perturbative doubles correction to the CIS excitation energy with CIS(D) via `CIS[D]` keyword
- Excited state coordinate scans, geometry optimisations, harmonic frequencies and MD simulations
- Orbital-optimised MP2 energy and density, `OMP2`
- Support for unrestricted references for SCS-MP2, SCS-MP3 and OMP2
- Unrelaxed density matrix for unrestricted MP2, SCS-MP2 and OMP2
- Keywords for same-spin, opposite-spin and MP3 scaling for SCS-MP3; `SSS`, `OSS` and `MP3S`
- Keywords for orbital-optimised MP2 convergence criteria and maximum iterations; `OMP2CONV` and `OMP2MAXITER`
- Keywords for state of interest in CIS, threshold for printing contributions, and number of states to print; `ROOT`, `CISTHRESH` and `NSTATES`
- Optional spin contamination calculation for MP2 calculations
- Optional population analysis and dipole moment calculations using CIS unrelaxed density matrix
- Faster one- and two-electron integrals

### Changed

- Rotational constant is now printed in both GHz and cm<sup>—1</sup>
- Molecular information at the beginning of a calculation now prints number of alpha and beta electrons
- Reorganised and improved prettiness of console log
- Maximum SCF iterations is now 100 by default, instead of 50
- Various and widespread low level optimisations, and all code is now fully documented
- Keywords `NORMALSCF` and `NORMALOPT` for SCF and geometry convergence replaced by `MEDIUMSCF` and `MEDIUMOPT`
- Default geometry convergence criteria set to `MEDIUMOPT` rather than `TIGHTOPT` by default, except for `OPTFREQ` calculations
- Absorbed tuna_dispersion module into tuna_energy, and added new tuna_ci module for current and future spin orbital calculations
- Used more colour in the console log

### Fixed

- Density matrix is now read in from previous optimisation step, except when initial guess orbitals are rotated, as previously intended
- Unrestricted MP2 was not working correctly for esoteric charge and multiplicity combinations
- Absolute change in density matrix is now checked, rather than signed change, for SCF convergence
- DIIS now works much more reliably for UHF, by combining the alpha and beta error vectors, converges faster for both RHF and UHF
- One extra SCF cycle is no longer undertaken for no reason
- Various miscellaneous bug fixes and improvements to error handling

<br>

## TUNA 0.5.1 — 14/11/2024

### Added

- Keyword for the default Hessian used in geometry optimisations, `DEFAULTHESS`
- Keyword for the maximum step size for geometry optimisations, `MAXSTEP`
- Optional parameter can be used with the `LEVELSHIFT` keyword to adjust the degree of level shift

### Changed

- Improved logging to console for a more consistent user experience
- Molecular dynamics simulations now read in the density matrix from the previous step by default
- Refactored and optimised all the code, ready for future updates and added comments and docstrings
- Separated two- and three-dimensional plotting functions into new tuna_plot module
- Keyword for the angle with which to rotate orbitals for initial guess, `THETA` has been removed and replaced by `ROTATE [Angle]`
- Reading in orbitals from previous coordinate scan step now turned off by default for UHF calculations

### Fixed

- Point group and molecular structure are now detected correctly for ghost atom calculations
- Nuclear repulsion energy is now not calculated for ghost atoms
- Individual energy components now work correctly for UHF energies
- Removed Koopmans' theorem parameter calculations for UHF references
- Molecular orbital eigenvalues and coefficients are now printed correctly for UHF, split into alpha and beta orbitals
- Stopped reading in orbitals for one-electron systems in coordinate scans, as there is no SCF cycle
- Orbitals are now rotated correctly by the requested angle for UHF initial guesses
- DIIS for UHF now works as intended when the equations can't be solved

<br>

## TUNA 0.5.0 — 21/09/2024

### Added

- *Ab initio* molecular dynamics
- Unrestricted Hartree-Fock energy and density
- Unrestricted MP2 energies
- Restricted and unrestricted MP3 energies
- Spin-component-scaled MP3 energies
- Keyword to decontract basis functions, `DECONTRACT`
- New basis sets: 4-31G, 6-31+G, 6-31++G, and 6-311+G
- Mayer bond order, free and total valences
- Spin contamination for UHF calculations
- Orbital rotation and `ROTATE` and `NOROTATE` keywords for UHF guess density
- Optimisations and molecular dynamics simulations optionally print to XYZ file with `TRAJ` and `NOTRAJ` keywords
- Option to optimize to a maximum with `OPTMAX` keyword
- Terminal output now has colour for warning and errors
- Increased speed of all TUNA calculations by 50–95% through making full use of permutational symmetry in the two-electron integrals
- Much better error handling and clear errors and warnings
- New changelog, manual, GitHub and PyPI pages 
- TUNA can now be installed simply by `pip install QuantumTUNA`

### Changed

- Rewrote all the code to make things object-oriented, improve efficiency and reduce redundancy
- Slimmed down the fish logo :(
- Optimised and simplified integral engine
- Reduced default SCF and optimisation convergence criteria by fixing associated bug
- Better handling of print levels; optimizations now only calculate properties at the end by default
- Now use more energy evaluations for gradients and Hessians, making them more robust but slower 
- Generally refined the output, making information more precise and clear

### Fixed

- When its equations can't be solved, DIIS now resets instead of crashing the program
- Fixed frequency calculations being far too sensitive to SCF convergence when guess density was read in
- SCF convergence was checking that DeltaE was less than the criteria, rather than its magnitude leading to too early convergence
- Fixed the thermochemistry module mixing up the temperature and pressure variables
- Formatting issues with population analysis
- Fixed handling of ghost atoms, accessible by `XH` or `XHe`

<br>

## TUNA 0.4.0 

### Added

- Fock matrix extrapolation for SCF convergence (DIIS)
- Electronic and total dipole moment
- Unrelaxed MP2 density and natural orbitals
- Thermochemistry after frequency calculations, `TEMPERATURE` and `PRESSURE` keywords
- New 3-21G basis set

### Changed

- Density matrix is now read by default from previous step in coordinate scans and optimisations

### Fixed

- Unbroke level shift, added keywords

<br>

## TUNA 0.3.0

### Added

- Geometry optimisations
- Harmonic frequencies, optionally linked with prior optimisation with `OPTFREQ` calculation type
- Rotational constants
- Nuclear dipole moment
- Optional exact or approximate (default) Hessian for optimisation
- Keywords for geometry convergence tolerance and maximum iterations
- High static damping option for difficult SCF convergence cases, `SLOWCONV`

<br>

## TUNA 0.2.0

### Added

- Conventional and spin-component-scaled MP2
- Mulliken and Löwdin population analysis
- Keywords for additional print, `P`, and SCF damping, `DAMP`
- Identification of point group

### Changed

- Updated to Python 3.12
- Significantly increased integral efficiency using vectorised operations

<br>

## TUNA 0.1.0

### Added

- Restricted Hartree–Fock
- Single point energy and coordinate scans
- New basis sets: STO-3G, STO-6G, 6-31G, 6-311G, 6-311++G
- Dynamic damping and level shift
- Ghost atoms
- Molecular orbitals and energies, Koopman's theorem parameters
- Electron density 3D plots
- Dispersion correction with semi-empirical D2 scheme
- Convergence criteria keywords for SCF
- Interface with matplotlib for coordinate scan via `SCANPLOT` keyword
