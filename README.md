# Risk-Parity portfolio optimization

A command line applet written in python for risk-parity based portfolio optimization.

## Theory

## Usage

To install, run
```shell
git clone https://github.com/borab96/risk-parity.git
cd risk-parity
pip install .
```

Run ``rpp -h`` for a list of settings that can be specified. The only requirement is the specification of a list of symbols.
The help command returns 
```shell
rpp [-h] [--cash CASH] [--period {1mo,3mo,6mo,1y,2y,5y,10y,ytd,max}]
           [--rebalance REBALANCE] [--to_pdf TO_PDF] [--leverage LEVERAGE]
           [--short SHORT] [--tune TUNE TUNE TUNE TUNE] [--gamma GAMMA]
           list of symbols [list of symbols ...]
...
```

The hyperparameters of the optimization algorithm are $\gamma$

 
