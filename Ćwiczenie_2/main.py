import argparse

from classes.point_cloud import PointCloud
from classes.dbscan import AnalyzeDBSCAN
from classes.pyransac import PlaneRANSAC


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyse point‑cloud planes with DBSCAN + pyransac3d")
    p.add_argument("xyz", help="Input .xyz file")
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--k", type=int, default=3)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print("Analyze RANSAC:")
    cloud = PointCloud(args.xyz, args.k, args.iterations, args.threshold)
    cloud.cluster_points()
    cloud.fit_planes()
    cloud.analyse_planes(0.1)

    print("")
    print("")
    print("Analyze DBSCAN:")
    dbscan = AnalyzeDBSCAN(args.xyz)
    dbscan.analyze_cloud(args.k, args.eps)

    print("")
    print("")
    print("Analyze RANSAC3D:")
    for idx, cl in enumerate(cloud.clusters, 1):
        pr = PlaneRANSAC(args.iterations, args.threshold)
        pr.fit(cl)
        pr.print_result(cl, idx, planar_tol=0.1)


if __name__ == "__main__":
    main()
