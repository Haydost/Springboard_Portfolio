#!/usr/bin/env python3
"""This module is designed to provide functions for the final report.

This module builds on prior modules: adnidatawrangling, eda, sda, and ml.
"""

if 'pd' not in globals():
    import pandas as pd
    
if 'np' not in globals():
    import numpy as np
    
if 'plt' not in globals():
    import matplotlib.pyplot as plt
    
if 'sns' not in globals():
    import seaborn as sns