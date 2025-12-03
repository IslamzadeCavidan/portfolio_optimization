Portfolio Optimization – Markowitz Efficient Frontier (Crypto Assets)

This project implements Modern Portfolio Theory (Markowitz, 1952) to analyze and optimize a cryptocurrency portfolio.
It computes both:

the Monte Carlo approximation of the Efficient Frontier

the exact Efficient Frontier using convex quadratic optimization (SLSQP)

the Maximum Sharpe Ratio (Tangency) Portfolio

The project is built with a fully modular structure separating:

src/ → financial logic

notebooks/ → analysis & visualization

data/ → stored datasets and generated plots

This is a professional, production-style research pipeline.

🚀 Features
✓ Data Processing

Automatic price download from Yahoo Finance (yfinance)

Log returns computation

Annualized expected returns and covariance matrix

✓ Portfolio Optimization

Random portfolio generation (Monte Carlo)

Exact Efficient Frontier via quadratic optimization

Maximum Sharpe Ratio portfolio

No-short-selling constraints 
𝑤(𝑖)∈[0,1]

Fully invested portfolio 
∑(𝑖)𝑤(𝑖)=1

✓ Visualizations

Monte Carlo risk–return cloud

Exact frontier overlay

Highlighted maximum Sharpe portfolio

High-resolution plots saved to /data/plots/

🧠 Mathematical Background

For weights vector 
𝑤
w, asset returns 
𝜇
μ, and covariance matrix 
Σ
Σ:

Expected Return
𝐸[𝑅(𝑝)]=𝑤⊤𝜇


Portfolio Volatility
𝜎(𝑝)=sqrt(𝑤⊤Σ𝑤)

Sharpe Ratio
𝑆(𝑝)=(𝐸[𝑅(𝑝)]−𝑟(𝑓))/𝜎(p)

Exact Efficient Frontier

Computed by solving the convex program:

min(𝑤) 𝑤⊤Σ𝑤

subject to:

𝑤⊤𝜇=𝜇^^∗ , ∑w(i)=1, 𝑤(𝑖)∈[0,1]

📂 Project Structure
portfolio-optimization/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── plots/
│       ├── monte_carlo_frontier.png
│       ├── exact_efficient_frontier.png
│       └── frontier_with_max_sharpe.png
│
├── notebooks/
│   └── markowitz_efficient_frontier.ipynb
│
├── src/
│   ├── __init__.py
│   └── optimization.py
│
├── requirements.txt
└── README.md

🧪 Running the Project
1. Install dependencies
pip install -r requirements.txt

2. Launch JupyterLab
jupyter lab

3. Open the notebook
notebooks/markowitz_efficient_frontier.ipynb


Run all cells to reproduce:

Random portfolios

Exact frontier

Max Sharpe portfolio

High-resolution figures saved to /data/plots/

📊 Example Outputs
Monte Carlo Frontier + Exact Efficient Frontier

(Your plot will appear here once pushed to GitHub)

Efficient Frontier with Tangency Portfolio

(Your plot will appear here once pushed to GitHub)

🏁 Conclusion

This project demonstrates:

- practical application of Modern Portfolio Theory

- real convex optimization (not just random sampling)

- clear, modular research workflow suitable for finance/quant roles

It is a strong addition to a GitHub portfolio and can be extended to:

- factor models

- regularization

- risk parity

- crypto-specific risk adjustments

- backtesting

- portfolio constraints (max/min per asset)