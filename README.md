
Pytorch implementation of EDMDDL [1].

This is my first project written in Python and PyTorch. If you find anything that can be improved, please let me know.

## Highlights

- **Programming-Mathematics Mapping**: We carefully designed the classes to ensure a clear one-to-one correspondence between programming constructs and mathematical concepts.
- **Framework**: The implementation is carried out in `Pytorch`.

## Quick Start

We highly recommend using `Anaconda` to manage the environment. If you have it installed, you can simply run:

``` bash
git clone https://github.com/ReichtumQian/KoopmanDL.git
cd KoopmanDL
# Create a new environment
conda create -n KoopmanDL python=3.8
conda activate KoopmanDL
# By default using CPU
pip install -r requirements.txt
```

Some examples are provided in the `example` folder.

## Documentation

Comprehensive mathematics details and design documentation are available in the `doc` folder. To compile the document, make sure your computer has `texlive` installed.

``` bash
git clone https://github.com/ReichtumQian/KoopmanDL.git
cd KoopmanDL/doc
make
```

## Contributors

- Yixiao Qian, Zhejiang University <yixiaoqian@zju.edu.com>
- Weizhen Li, Zhejiang University

## References

- [TensorFlow Implementation](https://github.com/MLDS-NUS/KoopmanDL/tree/main)
- [1] Extended dynamic mode decomposition with dictionary learning: A data-driven adaptive spectral decomposition of the Koopman operator
