import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


@dataclass(slots=True)
class PointCloudGenerator:

    number_of_points: int

    def horizontal_plane(self, width: float, length: float) -> np.ndarray:
        x = np.random.uniform(0.0, width, self.number_of_points)
        y = np.random.uniform(0.0, length, self.number_of_points)
        z = np.zeros(self.number_of_points)
        return np.column_stack((x, y, z))

    def vertical_plane(self, width: float, height: float) -> np.ndarray:
        x = np.random.uniform(0.0, width, self.number_of_points)
        z = np.random.uniform(0.0, height, self.number_of_points)
        y = np.zeros(self.number_of_points)
        return np.column_stack((x, y, z))

    def cylindrical_surface(self, radius: float, height: float) -> np.ndarray:
        theta = np.random.uniform(0.0, 2.0 * np.pi, self.number_of_points)
        z = np.random.uniform(0.0, height, self.number_of_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.column_stack((x, y, z))

    @staticmethod
    def save_xyz(points: np.ndarray, filepath: str | Path) -> None:
        Path(filepath).write_text(
            "\n".join(" ".join(f"{coord:.6f}" for coord in p) for p in points),
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate point clouds on analytic 3‑D surfaces (.xyz)."
    )
    subparsers = parser.add_subparsers(dest="surface", required=True)

    # Horizontal plane
    ph = subparsers.add_parser("horizontal", help="Horizontal plane z = 0")
    ph.add_argument("points", type=int, help="Number of points to generate")
    ph.add_argument("--width", type=float, required=True)
    ph.add_argument("--length", type=float, required=True)
    ph.add_argument("-o", "--output", default="horizontal.xyz")

    # Vertical plane
    pv = subparsers.add_parser("vertical", help="Vertical plane y = 0")
    pv.add_argument("points", type=int)
    pv.add_argument("--width", type=float, required=True)
    pv.add_argument("--height", type=float, required=True)
    pv.add_argument("-o", "--output", default="vertical.xyz")

    # Cylinder
    pc = subparsers.add_parser("cylinder", help="Cylindrical surface")
    pc.add_argument("points", type=int)
    pc.add_argument("--radius", type=float, required=True)
    pc.add_argument("--height", type=float, required=True)
    pc.add_argument("-o", "--output", default="cylinder.xyz")

    # Common options
    for p in (ph, pv, pc):
        p.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Optional random seed for reproducibility",
        )

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.seed is not None:
        np.random.seed(args.seed)

    generator = PointCloudGenerator(args.points)

    if args.surface == "horizontal":
        pts = generator.horizontal_plane(args.width, args.length)
    elif args.surface == "vertical":
        pts = generator.vertical_plane(args.width, args.height)
    elif args.surface == "cylinder":
        pts = generator.cylindrical_surface(args.radius, args.height)
    else:
        sys.exit("Unknown surface.")

    PointCloudGenerator.save_xyz(pts, args.output)
    print(f"Wrote {len(pts)} points to {args.output}")


if __name__ == "__main__":
    main()
