# EMG Toolbox

## Overview 
This repository contains functions to analyse electromyography (EMG) signals.

## Table of Contents
- [Installation](#installation)
- [Quick start](#quickstart)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)


#### Installation
To set up the project locally do the following:

1. Clone the repository:
    ```sh
    git clone https://github.com/imendezguerra/emg_toolbox.git
    ```
2. Navigate to the project directory:
    ```sh
    cd emg_toolbox
    ```
3. Create the conda environment from the `environment.yml` file:
    ```sh
    conda env create -f environment.yml
    ```
4. Activate the environment:
    ```sh
    conda activate emg_toolbox
    ```
5. Install toolbox
    ```
    pip install -e .
    ```

## Quick start 
The package is composed of the following modules:
- `tools.py`: Functions to deal with bad channels and rearrange EMG signals.
- `prepro.py`: Functions for EMG preprocessing such as filtering.
- `feats.py`: Functions to extract EMG features.
- `freq.py`: Functions to analyse the EMG signals in the frequency domain.
- `plots.py`: Functions to plot EMG signals

## Contributing
We welcome contributions! Here’s how you can contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/newfeature`).
3. Commit your changes (`git commit -m 'Add some newfeature'`).
4. Push to the branch (`git push origin feature/newfeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite this repository:

```sh
@software{Mendez_Guerra_EMG_Toolbox,
author = {Mendez Guerra, Irene},
title = {{EMG Toolbox}},
url = {https://github.com/imendezguerra/EMG_toolbox},
version = {1.0}
}
```
## Contact

For any questions or inquiries, please contact us at:
```sh
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```