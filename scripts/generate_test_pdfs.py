#!/usr/bin/env python3
"""Generate test PDFs with math symbols and charts for DoodleDoc sanity testing."""

import argparse
import random
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


# Math symbols and Greek letters
GREEK_LETTERS = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
                 "ν", "ξ", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
                 "Γ", "Δ", "Θ", "Λ", "Ξ", "Π", "Σ", "Φ", "Ψ", "Ω"]

MATH_OPERATORS = ["∫", "∑", "∏", "√", "∂", "∞", "≈", "≠", "≤", "≥", "±", "×", "÷"]

FORMULAS = [
    "E = mc²",
    "F = ma",
    "a² + b² = c²",
    "∫f(x)dx",
    "Σ(i=1 to n)",
    "∂f/∂x",
    "lim x→∞",
    "∇·F = 0",
    "dy/dx",
    "∫∫ dA",
    "x = (-b ± √(b²-4ac)) / 2a",
    "eiπ + 1 = 0",
    "∂²u/∂t² = c²∇²u",
]


def generate_math_pdf(output_path: Path, num_pages: int = 3) -> None:
    """Generate a PDF with random math symbols and formulas."""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    for page in range(num_pages):
        # Title
        c.setFont("Helvetica-Bold", 24)
        c.drawString(1 * inch, height - 1 * inch, f"Math Symbols - Page {page + 1}")

        # Random Greek letters scattered
        c.setFont("Helvetica", 36)
        for _ in range(random.randint(8, 15)):
            x = random.uniform(0.5 * inch, width - 1 * inch)
            y = random.uniform(2 * inch, height - 2 * inch)
            symbol = random.choice(GREEK_LETTERS + MATH_OPERATORS)
            c.drawString(x, y, symbol)

        # Random formulas
        c.setFont("Helvetica", 18)
        y_pos = height - 2.5 * inch
        for formula in random.sample(FORMULAS, min(5, len(FORMULAS))):
            x = random.uniform(0.5 * inch, width - 3 * inch)
            c.drawString(x, y_pos, formula)
            y_pos -= 0.8 * inch

        # Draw some lines/arrows to simulate handwriting
        c.setLineWidth(2)
        for _ in range(random.randint(3, 6)):
            x1 = random.uniform(1 * inch, width - 2 * inch)
            y1 = random.uniform(1 * inch, height - 3 * inch)
            x2 = x1 + random.uniform(-1 * inch, 1 * inch)
            y2 = y1 + random.uniform(-0.5 * inch, 0.5 * inch)
            c.line(x1, y1, x2, y2)

        c.showPage()

    c.save()


def generate_chart_pdf(output_path: Path, chart_type: str = "random") -> None:
    """Generate a PDF with matplotlib charts."""

    # Use xkcd style for hand-drawn look
    with plt.xkcd():
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle(f"Charts & Graphs", fontsize=16)

        # Sine wave
        ax = axes[0, 0]
        x = np.linspace(0, 4 * np.pi, 100)
        ax.plot(x, np.sin(x), linewidth=2)
        ax.set_title("Sine Wave")
        ax.set_xlabel("x")
        ax.set_ylabel("sin(x)")

        # Bar chart
        ax = axes[0, 1]
        categories = ["A", "B", "C", "D", "E"]
        values = np.random.randint(1, 10, 5)
        ax.bar(categories, values)
        ax.set_title("Bar Chart")

        # Scatter plot
        ax = axes[1, 0]
        x = np.random.randn(30)
        y = x + np.random.randn(30) * 0.5
        ax.scatter(x, y)
        ax.set_title("Scatter Plot")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Line plot with multiple series
        ax = axes[1, 1]
        x = np.linspace(0, 10, 50)
        ax.plot(x, x**2, label="x²")
        ax.plot(x, x**1.5, label="x^1.5")
        ax.plot(x, x, label="x")
        ax.legend()
        ax.set_title("Comparison")

        plt.tight_layout()
        plt.savefig(str(output_path), format="pdf", dpi=150)
        plt.close()


def generate_geometry_pdf(output_path: Path) -> None:
    """Generate a PDF with geometric shapes and diagrams."""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    # Page 1: Basic shapes
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1 * inch, height - 1 * inch, "Geometric Shapes")

    # Draw circles
    c.setLineWidth(2)
    for _ in range(5):
        x = random.uniform(2 * inch, width - 2 * inch)
        y = random.uniform(2 * inch, height - 3 * inch)
        r = random.uniform(0.3 * inch, 0.8 * inch)
        c.circle(x, y, r)

    # Draw rectangles
    for _ in range(4):
        x = random.uniform(1 * inch, width - 3 * inch)
        y = random.uniform(1 * inch, height - 4 * inch)
        w = random.uniform(0.5 * inch, 1.5 * inch)
        h = random.uniform(0.5 * inch, 1.5 * inch)
        c.rect(x, y, w, h)

    # Draw triangles (as paths)
    for _ in range(3):
        x = random.uniform(2 * inch, width - 2 * inch)
        y = random.uniform(2 * inch, height - 3 * inch)
        size = random.uniform(0.5 * inch, 1 * inch)
        path = c.beginPath()
        path.moveTo(x, y)
        path.lineTo(x + size, y)
        path.lineTo(x + size/2, y + size)
        path.close()
        c.drawPath(path)

    c.showPage()

    # Page 2: Coordinate system with labeled points
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1 * inch, height - 1 * inch, "Coordinate Geometry")

    # Draw axes
    center_x, center_y = width / 2, height / 2
    c.setLineWidth(2)
    c.line(1 * inch, center_y, width - 1 * inch, center_y)  # x-axis
    c.line(center_x, 1 * inch, center_x, height - 1.5 * inch)  # y-axis

    # Draw arrow heads
    c.drawString(width - 1.2 * inch, center_y + 0.1 * inch, "x")
    c.drawString(center_x + 0.1 * inch, height - 1.8 * inch, "y")

    # Plot some points with labels
    c.setFont("Helvetica", 12)
    points = [
        (center_x + 1 * inch, center_y + 1 * inch, "(1,1)"),
        (center_x - 1.5 * inch, center_y + 0.8 * inch, "(-1.5,0.8)"),
        (center_x + 0.5 * inch, center_y - 1.2 * inch, "(0.5,-1.2)"),
    ]
    for px, py, label in points:
        c.circle(px, py, 4, fill=1)
        c.drawString(px + 5, py + 5, label)

    c.showPage()
    c.save()


def generate_calculus_pdf(output_path: Path) -> None:
    """Generate a PDF with calculus-related content."""
    with plt.xkcd():
        fig = plt.figure(figsize=(8.5, 11))

        # Function and its derivative
        ax1 = fig.add_subplot(2, 2, 1)
        x = np.linspace(-2, 2, 100)
        ax1.plot(x, x**3, label="f(x) = x³", linewidth=2)
        ax1.plot(x, 3*x**2, label="f'(x) = 3x²", linewidth=2, linestyle="--")
        ax1.legend()
        ax1.set_title("Function & Derivative")
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)

        # Area under curve (integral visualization)
        ax2 = fig.add_subplot(2, 2, 2)
        x = np.linspace(0, 3, 100)
        y = np.sin(x) + 1
        ax2.plot(x, y, linewidth=2)
        ax2.fill_between(x, y, alpha=0.3)
        ax2.set_title("∫ sin(x)+1 dx")

        # Tangent line
        ax3 = fig.add_subplot(2, 2, 3)
        x = np.linspace(-1, 3, 100)
        ax3.plot(x, x**2, label="f(x) = x²", linewidth=2)
        # Tangent at x=1
        x_t = 1
        slope = 2 * x_t
        tangent = slope * (x - x_t) + x_t**2
        ax3.plot(x, tangent, label="Tangent at x=1", linewidth=2, linestyle="--")
        ax3.scatter([x_t], [x_t**2], color="red", s=50, zorder=5)
        ax3.legend()
        ax3.set_title("Tangent Line")
        ax3.set_ylim(-1, 5)

        # Limit visualization
        ax4 = fig.add_subplot(2, 2, 4)
        x = np.linspace(0.01, 2, 100)
        ax4.plot(x, np.sin(x)/x, linewidth=2)
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax4.set_title("lim(x→0) sin(x)/x = 1")
        ax4.set_xlabel("x")

        plt.tight_layout()
        plt.savefig(str(output_path), format="pdf", dpi=150)
        plt.close()


def generate_statistics_pdf(output_path: Path) -> None:
    """Generate a PDF with statistics-related charts."""
    with plt.xkcd():
        fig = plt.figure(figsize=(8.5, 11))

        # Histogram / Normal distribution
        ax1 = fig.add_subplot(2, 2, 1)
        data = np.random.randn(1000)
        ax1.hist(data, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_title("Normal Distribution")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")

        # Box plot
        ax2 = fig.add_subplot(2, 2, 2)
        data = [np.random.randn(50) + i for i in range(4)]
        ax2.boxplot(data, labels=["A", "B", "C", "D"])
        ax2.set_title("Box Plot Comparison")

        # Pie chart
        ax3 = fig.add_subplot(2, 2, 3)
        sizes = [35, 25, 20, 15, 5]
        labels = ["A", "B", "C", "D", "E"]
        ax3.pie(sizes, labels=labels, autopct='%1.0f%%')
        ax3.set_title("Pie Chart")

        # Correlation scatter
        ax4 = fig.add_subplot(2, 2, 4)
        x = np.random.randn(50)
        y = 2*x + np.random.randn(50)*0.5
        ax4.scatter(x, y, alpha=0.7)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax4.plot(x, p(x), "r--", linewidth=2)
        ax4.set_title("Linear Regression")
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")

        plt.tight_layout()
        plt.savefig(str(output_path), format="pdf", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test PDFs for DoodleDoc")
    parser.add_argument("--output", "-o", type=Path, default=Path("test_pdfs"),
                        help="Output directory for generated PDFs")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Generating test PDFs in {args.output}/...")

    # Generate math symbol PDFs
    for i in range(5):
        path = args.output / f"math_symbols_{i+1}.pdf"
        generate_math_pdf(path, num_pages=random.randint(2, 5))
        print(f"  Created: {path}")

    # Generate chart PDFs
    for i in range(3):
        path = args.output / f"charts_{i+1}.pdf"
        generate_chart_pdf(path)
        print(f"  Created: {path}")

    # Generate geometry PDF
    path = args.output / "geometry.pdf"
    generate_geometry_pdf(path)
    print(f"  Created: {path}")

    # Generate calculus PDF
    path = args.output / "calculus.pdf"
    generate_calculus_pdf(path)
    print(f"  Created: {path}")

    # Generate statistics PDF
    path = args.output / "statistics.pdf"
    generate_statistics_pdf(path)
    print(f"  Created: {path}")

    total = len(list(args.output.glob("*.pdf")))
    print(f"\nDone! Generated {total} test PDFs in {args.output}/")
    print(f"\nTo index these PDFs, run:")
    print(f"  make index ROOT={args.output}")


if __name__ == "__main__":
    main()
