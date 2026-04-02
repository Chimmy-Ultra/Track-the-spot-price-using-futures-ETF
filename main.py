"""
UNG Natural Gas ETF Replication Project - Main Runner
======================================================
Homework 1: Replication of commodity ETFs using commodity futures

Research Question:
  Does UNG's massive tracking error come from:
  (A) Wrong hedge ratio design?  → h* analysis
  (B) Market structure (contango)?  → roll yield decomposition

Answer: (B) — contango dominates. h* ≈ 0.43, R² ≈ 3.4% (daily)
  The 1:1 hedge ratio UNG uses is not "wrong"; spot and futures
  simply reflect different information sets (weather vs storage).

Steps:
  1. Data Download  (UNG ETF, NG=F, SHY from Yahoo; DHHNGSP from FRED)
  2. Data Cleaning  (align daily, compute returns, basis)
  3. NAV Simulation (front-month replication strategy)
  4. Analysis       (basis, seasonal, hedge ratio, tracking error decomposition)
  5. Comparison     (performance charts, annual returns, summary table)
"""
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)


def main():
    start_time = time.time()

    print('\n' + '=' * 60)
    print('  UNG Natural Gas ETF Replication Project')
    print('  United States Natural Gas Fund (UNG) Analysis')
    print('  Benchmark: Henry Hub Daily Spot (FRED DHHNGSP)')
    print('=' * 60 + '\n')

    from src.s01_data_download import run as step1
    step1()

    from src.s02_data_clean import run as step2
    step2()

    from src.s03_replication import run as step3
    step3()

    from src.s04_analysis import run as step4
    step4()

    from src.s05_comparison import run as step5
    step5()

    elapsed = time.time() - start_time
    print(f'\n{"=" * 60}')
    print(f'  All steps complete! Elapsed time: {elapsed:.1f}s')
    print(f'  Charts saved to: output/figures/')
    print(f'  Tables saved to: output/tables/')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()
