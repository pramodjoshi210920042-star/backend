from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import subprocess
import os
import tempfile
import shutil
from datetime import datetime
import uuid
import requests

app = Flask(__name__)
CORS(app)

class LDABeamerGenerator:
    def __init__(self, class1_data, class2_data, class1_name="\\omega_1", class2_name="\\omega_2"):
        self.class1_data = np.array(class1_data)
        self.class2_data = np.array(class2_data)
        self.class1_name = class1_name
        self.class2_name = class2_name
        
    def compute_lda(self):
        """Perform LDA calculations step by step"""
        # Step 1: Compute means
        self.mu1 = np.mean(self.class1_data, axis=0)
        self.mu2 = np.mean(self.class2_data, axis=0)
        
        # Step 2: Compute covariance matrices
        n1, n2 = len(self.class1_data), len(self.class2_data)
        
        # Deviations from mean
        self.dev1 = self.class1_data - self.mu1
        self.dev2 = self.class2_data - self.mu2
        
        # Compute outer products sum
        outer_sum1 = np.zeros((2, 2))
        outer_sum2 = np.zeros((2, 2))
        
        # Store individual outer products for detailed display
        self.outer_products1 = []
        self.outer_products2 = []
        self.deviations1 = []
        self.deviations2 = []
        
        for i, point in enumerate(self.class1_data):
            dev = point - self.mu1
            self.deviations1.append(dev)
            # Correct outer product calculation: dev is column vector, dev.T is row vector
            outer_prod = np.outer(dev, dev)  # This should be correct
            self.outer_products1.append(outer_prod)
            outer_sum1 += outer_prod
            
        for i, point in enumerate(self.class2_data):
            dev = point - self.mu2
            self.deviations2.append(dev)
            outer_prod = np.outer(dev, dev)  # This should be correct
            self.outer_products2.append(outer_prod)
            outer_sum2 += outer_prod
        
        # Covariance matrices with N-1 denominator
        self.S1 = outer_sum1 / (n1 - 1) if n1 > 1 else outer_sum1
        self.S2 = outer_sum2 / (n2 - 1) if n2 > 1 else outer_sum2
        
        # Within-class scatter matrix
        self.S_W = self.S1 + self.S2
        
        # Overall mean
        all_data = np.vstack([self.class1_data, self.class2_data])
        self.mu_overall = np.mean(all_data, axis=0)
        
        # Between-class scatter matrix
        diff1 = (self.mu1 - self.mu_overall).reshape(-1, 1)
        diff2 = (self.mu2 - self.mu_overall).reshape(-1, 1)
        
        self.S_B = n1 * (diff1 @ diff1.T) + n2 * (diff2 @ diff2.T)
        
        # Store intermediate calculations for display
        self.diff1 = diff1.flatten()
        self.diff2 = diff2.flatten()
        self.S_B1 = n1 * (diff1 @ diff1.T)
        self.S_B2 = n2 * (diff2 @ diff2.T)
        
        # Step 3: Solve eigenvalue problem
        self.S_W_inv = np.linalg.inv(self.S_W)
        self.S_W_inv_S_B = self.S_W_inv @ self.S_B
        
        eigenvalues, eigenvectors = np.linalg.eig(self.S_W_inv_S_B)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx].real
        self.eigenvectors = eigenvectors[:, idx].real
        
        # Take the first eigenvector (largest eigenvalue)
        self.w = self.eigenvectors[:, 0]
        
        # Ensure consistent direction
        if self.w[0] < 0:
            self.w = -self.w
        
        # Alternative: direct computation
        self.w_direct = self.S_W_inv @ (self.mu1 - self.mu2)
        if self.w_direct[0] < 0:
            self.w_direct = -self.w_direct
        
        # Project data
        self.proj1 = self.class1_data @ self.w
        self.proj2 = self.class2_data @ self.w
    
    def format_matrix(self, matrix, precision=2):
        """Format a matrix for LaTeX"""
        if matrix.ndim == 1:
            return " \\\\ ".join(f"{val:.{precision}f}" for val in matrix)
        rows = []
        for row in matrix:
            row_str = " & ".join(f"{val:.{precision}f}" for val in row)
            rows.append(row_str)
        return " \\\\ ".join(rows)
    
    def generate_latex_string(self):
        """Generate LaTeX Beamer presentation with detailed covariance calculations"""
        
        # Format data points
        class1_points_str = ", ".join([f"({x:.0f},{y:.0f})" for x, y in self.class1_data])
        class2_points_str = ", ".join([f"({x:.0f},{y:.0f})" for x, y in self.class2_data])
        
        # Format mean calculations
        class1_sum = " + ".join([f"\\begin{{pmatrix}}{x:.0f}\\\\{y:.0f}\\end{{pmatrix}}" 
                                for x, y in self.class1_data])
        class2_sum = " + ".join([f"\\begin{{pmatrix}}{x:.0f}\\\\{y:.0f}\\end{{pmatrix}}" 
                                for x, y in self.class2_data])
        
        # Use consistent 4-decimal precision throughout
        PRECISION = 4
        
        # Generate detailed covariance calculations with explicit outer products
        cov1_deviation_calcs = []
        cov1_outer_product_calcs = []
        cov1_matrices = []
        
        for i, dev in enumerate(self.deviations1):
            x_orig, y_orig = self.class1_data[i]
            dev_x, dev_y = dev[0], dev[1]
            
            # Deviation calculation
            dev_calc = f"x_{{{i+1}}} - \\mu_1 = \\begin{{pmatrix}}{x_orig:.0f}\\\\{y_orig:.0f}\\end{{pmatrix}} - \\begin{{pmatrix}}{self.mu1[0]:.2f}\\\\{self.mu1[1]:.2f}\\end{{pmatrix}} = \\begin{{pmatrix}}{dev_x:.2f}\\\\{dev_y:.2f}\\end{{pmatrix}}"
            cov1_deviation_calcs.append(dev_calc)
            
            # Correct outer product calculation
            outer_prod = self.outer_products1[i]
            outer_calc = f"\\begin{{pmatrix}}{dev_x:.2f}\\\\{dev_y:.2f}\\end{{pmatrix}}\\begin{{pmatrix}}{dev_x:.2f} & {dev_y:.2f}\\end{{pmatrix}} = \\begin{{pmatrix}}{outer_prod[0,0]:.2f} & {outer_prod[0,1]:.2f}\\\\{outer_prod[1,0]:.2f} & {outer_prod[1,1]:.2f}\\end{{pmatrix}}"
            cov1_outer_product_calcs.append(outer_calc)
            
            # Just the resulting matrix for summation
            matrix_str = f"\\begin{{pmatrix}}{outer_prod[0,0]:.2f} & {outer_prod[0,1]:.2f}\\\\{outer_prod[1,0]:.2f} & {outer_prod[1,1]:.2f}\\end{{pmatrix}}"
            cov1_matrices.append(matrix_str)
        
        # Same for class 2
        cov2_deviation_calcs = []
        cov2_outer_product_calcs = []
        cov2_matrices = []
        
        for i, dev in enumerate(self.deviations2):
            x_orig, y_orig = self.class2_data[i]
            dev_x, dev_y = dev[0], dev[1]
            
            dev_calc = f"x_{{{i+1}}} - \\mu_2 = \\begin{{pmatrix}}{x_orig:.0f}\\\\{y_orig:.0f}\\end{{pmatrix}} - \\begin{{pmatrix}}{self.mu2[0]:.2f}\\\\{self.mu2[1]:.2f}\\end{{pmatrix}} = \\begin{{pmatrix}}{dev_x:.2f}\\\\{dev_y:.2f}\\end{{pmatrix}}"
            cov2_deviation_calcs.append(dev_calc)
            
            outer_prod = self.outer_products2[i]
            outer_calc = f"\\begin{{pmatrix}}{dev_x:.2f}\\\\{dev_y:.2f}\\end{{pmatrix}}\\begin{{pmatrix}}{dev_x:.2f} & {dev_y:.2f}\\end{{pmatrix}} = \\begin{{pmatrix}}{outer_prod[0,0]:.2f} & {outer_prod[0,1]:.2f}\\\\{outer_prod[1,0]:.2f} & {outer_prod[1,1]:.2f}\\end{{pmatrix}}"
            cov2_outer_product_calcs.append(outer_calc)
            
            matrix_str = f"\\begin{{pmatrix}}{outer_prod[0,0]:.2f} & {outer_prod[0,1]:.2f}\\\\{outer_prod[1,0]:.2f} & {outer_prod[1,1]:.2f}\\end{{pmatrix}}"
            cov2_matrices.append(matrix_str)

        # Format the lists for LaTeX
        cov1_deviations_str = " \\\\\n".join(cov1_deviation_calcs)
        cov1_outer_products_str = " \\\\\n".join(cov1_outer_product_calcs)
        cov1_matrices_str = " + ".join(cov1_matrices)
        
        cov2_deviations_str = " \\\\\n".join(cov2_deviation_calcs)
        cov2_outer_products_str = " \\\\\n".join(cov2_outer_product_calcs)
        cov2_matrices_str = " + ".join(cov2_matrices)

        # Eigenvalue calculation details
        A = self.S_W_inv_S_B
        a, b = A[0, 0], A[0, 1]
        c, d = A[1, 0], A[1, 1]
        trace = a + d
        det = a * d - b * c

        frame_title = "Calculate inverse of S_W and its product with S_B"

        latex_content = f"""\\documentclass{{beamer}}
\\usepackage{{amsmath, amssymb, array, booktabs, xcolor}}
\\usetheme{{Madrid}}
\\usecolortheme{{dolphin}}
\\setbeamertemplate{{headline}}{{}}
\\setbeamertemplate{{footline}}{{}}
\\setbeamertemplate{{navigation symbols}}{{}}

\\title{{Linear Discriminant Analysis (LDA)}}
\\author{{}}
\\date{{}}

\\begin{{document}}

\\frame{{\\titlepage}}

% Slide 1: Dataset - Fixed format
\\begin{{frame}}{{Dataset}}
\\large
\\[
\\text{{Samples for class {self.class1_name}: }} X_1 = (x_1, x_2) = {class1_points_str}
\\]

\\vspace{{1cm}}

\\[
\\text{{Samples for class {self.class2_name}: }} X_2 = (x_1, x_2) = {class2_points_str}
\\]
\\end{{frame}}

% Slide 2: Class Means
\\begin{{frame}}{{The Classes Means Are}}
\\[
\\mu_1 = \\frac{{1}}{{N_1}} \\sum_{{x \\in \\omega_1}} x = \\frac{{1}}{{{len(self.class1_data)}}} \\left[ {class1_sum} \\right] = \\begin{{pmatrix}}{self.mu1[0]:.2f}\\\\{self.mu1[1]:.2f}\\end{{pmatrix}}
\\]

\\vspace{{0.8cm}}

\\[
\\mu_2 = \\frac{{1}}{{N_2}} \\sum_{{x \\in \\omega_2}} x = \\frac{{1}}{{{len(self.class2_data)}}} \\left[ {class2_sum} \\right] = \\begin{{pmatrix}}{self.mu2[0]:.2f}\\\\{self.mu2[1]:.2f}\\end{{pmatrix}}
\\]
\\end{{frame}}

% Slide 3: Covariance Matrix of First Class - DETAILED
\\begin{{frame}}{{Covariance Matrix of the First Class}}
\\tiny
\\[
S_1 = \\frac{{1}}{{N-1}} \\sum_{{x \\in \\omega_1}}(x - \\mu_1)(x - \\mu_1)^T
\\]

\\textbf{{Step 1: Calculate deviations from mean}}
\\[
\\begin{{aligned}}
{cov1_deviations_str}
\\end{{aligned}}
\\]

\\textbf{{Step 2: Calculate outer products $(x - \\mu_1)(x - \\mu_1)^T$}}
\\[
\\begin{{aligned}}
{cov1_outer_products_str}
\\end{{aligned}}
\\]

\\textbf{{Step 3: Sum all matrices and divide by $(N-1) = {len(self.class1_data)-1}$}}
\\[
S_1 = \\frac{{1}}{{{len(self.class1_data)-1}}} \\left[ {cov1_matrices_str} \\right] = \\begin{{pmatrix}}{self.S1[0,0]:.2f} & {self.S1[0,1]:.2f}\\\\{self.S1[1,0]:.2f} & {self.S1[1,1]:.2f}\\end{{pmatrix}}
\\]
\\end{{frame}}

% Slide 4: Covariance Matrix of Second Class - DETAILED
\\begin{{frame}}{{Covariance Matrix of the Second Class}}
\\tiny
\\[
S_2 = \\frac{{1}}{{N-1}} \\sum_{{x \\in \\omega_2}}(x - \\mu_2)(x - \\mu_2)^T
\\]

\\textbf{{Step 1: Calculate deviations from mean}}
\\[
\\begin{{aligned}}
{cov2_deviations_str}
\\end{{aligned}}
\\]

\\textbf{{Step 2: Calculate outer products $(x - \\mu_2)(x - \\mu_2)^T$}}
\\[
\\begin{{aligned}}
{cov2_outer_products_str}
\\end{{aligned}}
\\]

\\textbf{{Step 3: Sum all matrices and divide by $(N-1) = {len(self.class2_data)-1}$}}
\\[
S_2 = \\frac{{1}}{{{len(self.class2_data)-1}}} \\left[ {cov2_matrices_str} \\right] = \\begin{{pmatrix}}{self.S2[0,0]:.2f} & {self.S2[0,1]:.2f}\\\\{self.S2[1,0]:.2f} & {self.S2[1,1]:.2f}\\end{{pmatrix}}
\\]
\\end{{frame}}

% Slide 5: Within-class Scatter Matrix
\\begin{{frame}}{{Within-class Scatter Matrix S_W}}
\\[
S_W = S_1 + S_2 = \\begin{{pmatrix}}
{self.S1[0,0]:.2f} & {self.S1[0,1]:.2f} \\\\
{self.S1[1,0]:.2f} & {self.S1[1,1]:.2f}
\\end{{pmatrix}} + \\begin{{pmatrix}}
{self.S2[0,0]:.2f} & {self.S2[0,1]:.2f} \\\\
{self.S2[1,0]:.2f} & {self.S2[1,1]:.2f}
\\end{{pmatrix}}
\\]

\\vspace{{0.5cm}}

\\[
= \\begin{{pmatrix}}
{self.S_W[0,0]:.2f} & {self.S_W[0,1]:.2f} \\\\
{self.S_W[1,0]:.2f} & {self.S_W[1,1]:.2f}
\\end{{pmatrix}}
\\]
\\end{{frame}}

% Slide 6: Overall Mean Calculation
\\begin{{frame}}{{Overall Mean Calculation}}
\\footnotesize
\\textbf{{Step 1: Calculate Overall Mean}}

All data points: {", ".join([f"({x:.0f},{y:.0f})" for x, y in self.class1_data])} + {", ".join([f"({x:.0f},{y:.0f})" for x, y in self.class2_data])}

\\[
\\mu = \\frac{{1}}{{N_1 + N_2}} \\sum_{{all}} x = \\frac{{1}}{{{len(self.class1_data)} + {len(self.class2_data)}}} \\left[ {" + ".join([f"\\begin{{pmatrix}}{x:.0f}\\\\{y:.0f}\\end{{pmatrix}}" for x, y in np.vstack([self.class1_data, self.class2_data])])} \\right]
\\]

\\[
= \\frac{{1}}{{{len(self.class1_data) + len(self.class2_data)}}} \\begin{{pmatrix}}{np.sum(np.vstack([self.class1_data, self.class2_data]), axis=0)[0]:.0f}\\\\{np.sum(np.vstack([self.class1_data, self.class2_data]), axis=0)[1]:.0f}\\end{{pmatrix}} = \\begin{{pmatrix}}{self.mu_overall[0]:.2f}\\\\{self.mu_overall[1]:.2f}\\end{{pmatrix}}
\\]

\\textbf{{Step 2: Calculate differences from overall mean}}
\\[
\\mu_1 - \\mu = \\begin{{pmatrix}}{self.mu1[0]:.2f}\\\\{self.mu1[1]:.2f}\\end{{pmatrix}} - \\begin{{pmatrix}}{self.mu_overall[0]:.2f}\\\\{self.mu_overall[1]:.2f}\\end{{pmatrix}} = \\begin{{pmatrix}}{self.diff1[0]:.2f}\\\\{self.diff1[1]:.2f}\\end{{pmatrix}}
\\]

\\[
\\mu_2 - \\mu = \\begin{{pmatrix}}{self.mu2[0]:.2f}\\\\{self.mu2[1]:.2f}\\end{{pmatrix}} - \\begin{{pmatrix}}{self.mu_overall[0]:.2f}\\\\{self.mu_overall[1]:.2f}\\end{{pmatrix}} = \\begin{{pmatrix}}{self.diff2[0]:.2f}\\\\{self.diff2[1]:.2f}\\end{{pmatrix}}
\\]
\\end{{frame}}

% Slide 7: Between-class Scatter Matrix
\\begin{{frame}}{{Between-class Scatter Matrix S_B}}
\\footnotesize

\\[
S_B = n_1(\\mu_1 - \\mu)(\\mu_1 - \\mu)^T + n_2(\\mu_2 - \\mu)(\\mu_2 - \\mu)^T
\\]

\\[
= {len(self.class1_data)} \\begin{{pmatrix}}{self.diff1[0]:.2f}\\\\{self.diff1[1]:.2f}\\end{{pmatrix}}\\begin{{pmatrix}}{self.diff1[0]:.2f} & {self.diff1[1]:.2f}\\end{{pmatrix}} + {len(self.class2_data)} \\begin{{pmatrix}}{self.diff2[0]:.2f}\\\\{self.diff2[1]:.2f}\\end{{pmatrix}}\\begin{{pmatrix}}{self.diff2[0]:.2f} & {self.diff2[1]:.2f}\\end{{pmatrix}}
\\]

\\[
= {len(self.class1_data)} \\begin{{pmatrix}}{self.diff1[0]*self.diff1[0]:.2f} & {self.diff1[0]*self.diff1[1]:.2f}\\\\{self.diff1[1]*self.diff1[0]:.2f} & {self.diff1[1]*self.diff1[1]:.2f}\\end{{pmatrix}} + {len(self.class2_data)} \\begin{{pmatrix}}{self.diff2[0]*self.diff2[0]:.2f} & {self.diff2[0]*self.diff2[1]:.2f}\\\\{self.diff2[1]*self.diff2[0]:.2f} & {self.diff2[1]*self.diff2[1]:.2f}\\end{{pmatrix}}
\\]

\\[
= \\begin{{pmatrix}}{self.S_B1[0,0]:.2f} & {self.S_B1[0,1]:.2f}\\\\{self.S_B1[1,0]:.2f} & {self.S_B1[1,1]:.2f}\\end{{pmatrix}} + \\begin{{pmatrix}}{self.S_B2[0,0]:.2f} & {self.S_B2[0,1]:.2f}\\\\{self.S_B2[1,0]:.2f} & {self.S_B2[1,1]:.2f}\\end{{pmatrix}} = \\begin{{pmatrix}}{self.S_B[0,0]:.2f} & {self.S_B[0,1]:.2f}\\\\{self.S_B[1,0]:.2f} & {self.S_B[1,1]:.2f}\\end{{pmatrix}}
\\]
\\end{{frame}}

% Slide 8: Calculate $S_W^{{-1}}$ and $S_W^{{-1}}S_B$

\\begin{{frame}}{{{frame_title}}}
\\footnotesize
\\[
S_W^{{-1}} = \\frac{{1}}{{\\det(S_W)}} \\begin{{pmatrix}} d & -b \\\\ -c & a \\end{{pmatrix}}, \\quad
S_W^{{-1}} S_B
\\]

\\textbf{{Step 1: Calculate inverse of $S_W$}}

\\[
S_W = \\begin{{pmatrix}} a & b \\\\ c & d \\end{{pmatrix}} =
\\begin{{pmatrix}} {self.S_W[0,0]:.2f} & {self.S_W[0,1]:.2f} \\\\ {self.S_W[1,0]:.2f} & {self.S_W[1,1]:.2f} \\end{{pmatrix}}
\\]

Determinant:

\\[
\\det(S_W) = ad - bc =
({self.S_W[0,0]:.2f})({self.S_W[1,1]:.2f}) -
({self.S_W[0,1]:.2f})({self.S_W[1,0]:.2f}) = {np.linalg.det(self.S_W):.4f}
\\]

\\[
S_W^{{-1}} = \\frac{{1}}{{det(S_W)}} \\begin{{pmatrix}}d & -b\\\\-c & a\\end{{pmatrix}} = \\frac{{1}}{{{np.linalg.det(self.S_W):.4f}}} \\begin{{pmatrix}}{self.S_W[1,1]:.2f} & {-self.S_W[0,1]:.2f}\\\\{-self.S_W[1,0]:.2f} & {self.S_W[0,0]:.2f}\\end{{pmatrix}}
\\]

\\[
= \\begin{{pmatrix}}{self.S_W_inv[0,0]:.4f} & {self.S_W_inv[0,1]:.4f}\\\\{self.S_W_inv[1,0]:.4f} & {self.S_W_inv[1,1]:.4f}\\end{{pmatrix}}
\\]

\\textbf{{Step 2: Calculate $S_W^{{-1}}S_B$}}

\\[
S_W^{{-1}}S_B = \\begin{{pmatrix}}{self.S_W_inv[0,0]:.4f} & {self.S_W_inv[0,1]:.4f}\\\\{self.S_W_inv[1,0]:.4f} & {self.S_W_inv[1,1]:.4f}\\end{{pmatrix}} \\begin{{pmatrix}}{self.S_B[0,0]:.2f} & {self.S_B[0,1]:.2f}\\\\{self.S_B[1,0]:.2f} & {self.S_B[1,1]:.2f}\\end{{pmatrix}}
\\]

\\[
= \\begin{{pmatrix}}{a:.4f} & {b:.4f}\\\\{c:.4f} & {d:.4f}\\end{{pmatrix}}
\\]
\\end{{frame}}

% Slide 9: Find Eigenvalues
\\begin{{frame}}{{Find Eigenvalues}}
\\footnotesize
\\textbf{{Solve:}} $S_W^{{-1}} S_B w = \\lambda w \\Rightarrow |S_W^{{-1}} S_B - \\lambda I| = 0$

\\textbf{{Step 1: Set up characteristic equation}}
\\[
\\left| S_W^{{-1}}S_B - \\lambda I \\right| = \\left| \\begin{{pmatrix}}{a:.4f} & {b:.4f}\\\\{c:.4f} & {d:.4f}\\end{{pmatrix}} - \\lambda \\begin{{pmatrix}}1 & 0\\\\0 & 1\\end{{pmatrix}} \\right| = 0
\\]

\\[
= \\left| \\begin{{pmatrix}}{a:.4f} - \\lambda & {b:.4f}\\\\{c:.4f} & {d:.4f} - \\lambda\\end{{pmatrix}} \\right| = 0
\\]

\\textbf{{Step 2: Calculate determinant}}
\\[
= ({a:.4f} - \\lambda)({d:.4f} - \\lambda) - ({b:.4f})({c:.4f}) = 0
\\]

\\[
= {a:.4f} \\cdot {d:.4f} - {a:.4f}\\lambda - {d:.4f}\\lambda + \\lambda^2 - {b*c:.6f} = 0
\\]

\\[
= \\lambda^2 - ({a:.4f} + {d:.4f})\\lambda + ({a*d:.6f} - {b*c:.6f}) = 0
\\]

\\[
\\Rightarrow \\lambda^2 - {trace:.4f}\\lambda + {det:.6f} = 0
\\]

\\textbf{{Step 3: Solve quadratic equation}}
\\[
\\lambda = \\frac{{{trace:.4f} \\pm \\sqrt{{({trace:.4f})^2 - 4({det:.6f})}}}}{{2}} = \\frac{{{trace:.4f} \\pm \\sqrt{{{trace**2:.6f} - {4*det:.6f}}}}}{{2}}
\\]

\\[
\\Rightarrow \\lambda_1 = {self.eigenvalues[0]:.4f}, \\lambda_2 = {self.eigenvalues[1]:.6f}
\\]
\\end{{frame}}

% Slide 10: Find Eigenvectors - FIXED VERSION
\\begin{{frame}}{{Find Eigenvectors}}
\\footnotesize
\\textbf{{For largest eigenvalue $\\lambda_1 = {self.eigenvalues[0]:.4f}$:}}

\\[
\\left( S_W^{{-1}} S_B - \\lambda_1 I \\right) w = 0
\\]

\\[
\\begin{{pmatrix}} 
{a:.4f} - {self.eigenvalues[0]:.4f} & {b:.4f} \\\\ 
{c:.4f} & {d:.4f} - {self.eigenvalues[0]:.4f} 
\\end{{pmatrix}}
\\begin{{pmatrix}} w_1 \\\\ w_2 \\end{{pmatrix}}
= \\begin{{pmatrix}} 0 \\\\ 0 \\end{{pmatrix}}
\\]

\\[
\\begin{{pmatrix}} 
{a - self.eigenvalues[0]:.6f} & {b:.6f} \\\\ 
{c:.6f} & {d - self.eigenvalues[0]:.6f} 
\\end{{pmatrix}}
\\begin{{pmatrix}} w_1 \\\\ w_2 \\end{{pmatrix}}
= \\begin{{pmatrix}} 0 \\\\ 0 \\end{{pmatrix}}
\\]

\\textbf{{This gives us the system:}}
\\[
({a - self.eigenvalues[0]:.4f}) w_1 + ({b:.4f}) w_2 = 0
\\]
\\[
({c:.4f}) w_1 + ({d - self.eigenvalues[0]:.4f}) w_2 = 0
\\]

\\textbf{{From first equation (if first coefficient $\\neq 0$):}}
\\[
w_1 = -\\frac{{{b:.4f}}}{{{a-self.eigenvalues[0]:.6f}}} w_2
\\]

{"\\[w_1 = " + f"{-b/(a-self.eigenvalues[0]):.4f}" + " w_2\\]" if abs(a-self.eigenvalues[0]) > 1e-10 else "\\textbf{Note: First coefficient near zero, using second equation}"}

\\textbf{{Normalized eigenvector:}}
\\[
w^* = \\begin{{pmatrix}} {self.w[0]:.4f} \\\\ {self.w[1]:.4f} \\end{{pmatrix}}
\\]

\\vspace{{0.3cm}}
\\textbf{{Alternative direct method:}}
\\[
w = S_W^{{-1}} (\\mu_1 - \\mu_2)
\\]

\\[
w = \\begin{{pmatrix}} 
{self.S_W_inv[0,0]:.4f} & {self.S_W_inv[0,1]:.4f} \\\\ 
{self.S_W_inv[1,0]:.4f} & {self.S_W_inv[1,1]:.4f} 
\\end{{pmatrix}}
\\begin{{pmatrix}} 
{self.mu1[0]-self.mu2[0]:.4f} \\\\ 
{self.mu1[1]-self.mu2[1]:.4f} 
\\end{{pmatrix}}
= \\begin{{pmatrix}} 
{self.w_direct[0]:.4f} \\\\ 
{self.w_direct[1]:.4f} 
\\end{{pmatrix}}
\\]
\\end{{frame}}

% Slide 11: Project Data - Detailed Calculations
\\begin{{frame}}{{Project Data - Step by Step}}
\\scriptsize
\\[
\\text{{Using }} w^* = \\begin{{pmatrix}}{self.w[0]:.3f}\\\\{self.w[1]:.3f}\\end{{pmatrix}}, 
\\text{{compute projection }} y = w^T x = {self.w[0]:.3f} \\cdot x_1 + {self.w[1]:.3f} \\cdot x_2
\\]

\\textbf{{Class {self.class1_name} projections:}}
\\[
{" \\\\\n".join([f"y_{{{i+1}}} = {self.w[0]:.3f} \\cdot {x:.0f} + {self.w[1]:.3f} \\cdot {y:.0f} = {x*self.w[0] + y*self.w[1]:.3f}" for i, (x, y) in enumerate(self.class1_data)])}
\\]

\\textbf{{Class {self.class2_name} projections:}}
\\[
{" \\\\\n".join([f"y_{{{i+1+len(self.class1_data)}}} = {self.w[0]:.3f} \\cdot {x:.0f} + {self.w[1]:.3f} \\cdot {y:.0f} = {x*self.w[0] + y*self.w[1]:.3f}" for i, (x, y) in enumerate(self.class2_data)])}
\\]
\\end{{frame}}

% Slide 12: Final Results Summary
\\begin{{frame}}{{Final LDA Results Summary}}
\\footnotesize
\\textbf{{Optimal projection vector:}} $w^* = \\begin{{pmatrix}}{self.w[0]:.4f}\\\\{self.w[1]:.4f}\\end{{pmatrix}}$

\\textbf{{Complete projection results:}}

\\begin{{center}}
\\begin{{tabular}}{{|c|c|c|c|}}
\\hline
Class & X1 & X2 & Projection \\\\
\\hline
{" \\\\\n".join([f"{self.class1_name} & {x:.0f} & {y:.0f} & {x*self.w[0] + y*self.w[1]:.3f}" for x, y in self.class1_data])} \\\\
\\hline
{" \\\\\n".join([f"{self.class2_name} & {x:.0f} & {y:.0f} & {x*self.w[0] + y*self.w[1]:.3f}" for x, y in self.class2_data])} \\\\
\\hline
\\end{{tabular}}
\\end{{center}}

\\vspace{{0.3cm}}

\\textbf{{1D projected data:}}
\\begin{{itemize}}
\\item Class {self.class1_name}: $\\{{{", ".join([f"{p:.0f}" for p in self.proj1])}\\}}$
\\item Class {self.class2_name}: $\\{{{", ".join([f"{p:.0f}" for p in self.proj2])}\\}}$
\\end{{itemize}}

The LDA transformation successfully projects the 2D data onto a 1D line that maximizes class separation.
\\end{{frame}}

\\end{{document}}"""

        return latex_content

@app.route('/api/latex-string', methods=['POST'])
def get_latex_string():
    try:
        data = request.json
        class1_data = data['class1Data']
        class2_data = data['class2Data']
        class1_name = data.get('class1Name', 'Class1')
        class2_name = data.get('class2Name', 'Class2')

        # DEBUG: Print received data
        print("=== RECEIVED DATA ===")
        print(f"Class 1 data: {class1_data}")
        print(f"Class 2 data: {class2_data}")
        print(f"Class 1 name: {class1_name}")
        print(f"Class 2 name: {class2_name}")

        lda_gen = LDABeamerGenerator(class1_data, class2_data, class1_name, class2_name)
        lda_gen.compute_lda()
        
        # DEBUG: Print calculated means
        print(f"Calculated mean 1: {lda_gen.mu1}")
        print(f"Calculated mean 2: {lda_gen.mu2}")
        
        # DEBUG: Print some deviations and outer products
        print(f"First deviation class 1: {lda_gen.deviations1[0]}")
        print(f"Second deviation class 1: {lda_gen.deviations1[1]}")
        print(f"Second outer product class 1:\n{lda_gen.outer_products1[1]}")
        
        latex_str = lda_gen.generate_latex_string()

        return jsonify({'latex': latex_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':

    app.run(debug=True, host=0.0.0.0 ,port=5000)


