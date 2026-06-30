import math
import random
import time
import csv
from array import array

#確認

# ============================================================
# 共通処理
# ============================================================

def make_points(n, seed=0, width=100.0):
    """
    2次元空間上にn個の都市を一様乱数で発生させる。
    seedを変えると別の例題になる。
    """
    rng = random.Random(seed)
    return [(rng.random() * width, rng.random() * width) for _ in range(n)]


def make_dist(points):
    """
    都市間距離行列を作成する。
    """
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]

    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            d = math.hypot(xi - xj, yi - yj)
            dist[i][j] = d
            dist[j][i] = d

    return dist


def tour_length(route, dist):
    """
    巡回路の総距離を計算する。
    route = [0, 3, 1, 2] の場合、
    0 -> 3 -> 1 -> 2 -> 0 の距離を計算する。
    """
    n = len(route)
    total = 0.0

    for i in range(n):
        total += dist[route[i]][route[(i + 1) % n]]

    return total


def route_to_string(route):
    """
    経路を見やすい文字列に変換する。
    """
    return " -> ".join(map(str, route)) + f" -> {route[0]}"


# ============================================================
# 1-(a) 最近隣法
# ============================================================

def nearest_neighbor(dist, start=0):
    """
    最近隣法によって巡回路を構成する。

    現在の都市から、まだ訪問していない都市のうち
    最も近い都市へ移動する。
    """
    n = len(dist)

    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)

    current = start

    while unvisited:
        next_city = min(unvisited, key=lambda city: dist[current][city])
        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return route


# ============================================================
# 2-(a) 挿入近傍法
# ============================================================

def insertion_local_search(dist, init_route):
    """
    挿入近傍法による局所探索。

    ある都市を1つ取り出し、別の位置に挿入する。
    巡回路が短くなる移動があれば採用し、
    改善できなくなるまで繰り返す。
    """
    route = init_route[:]
    n = len(route)

    best_len = tour_length(route, dist)

    improved = True

    while improved:
        improved = False
        best_route = None

        # すべての都市について「抜き取り」→「挿入」を試す
        for i in range(n):
            city = route[i]
            rest = route[:i] + route[i + 1:]

            # restにcityを挿入する位置をすべて試す
            for j in range(n):
                new_route = rest[:j] + [city] + rest[j:]

                if new_route == route:
                    continue

                new_len = tour_length(new_route, dist)

                if new_len < best_len - 1e-12:
                    best_len = new_len
                    best_route = new_route

        if best_route is not None:
            route = best_route
            improved = True

    return route


# ============================================================
# 3-(a) 多出発法
# ============================================================

def multi_start_nn_insertion(dist):
    """
    多出発法。

    出発都市を0,1,2,...,n-1のように変化させる。
    各出発都市について、

    最近隣法で初期解を作る
    ↓
    挿入近傍法で改善する
    ↓
    最も良い解を採用する

    という流れで探索する。
    """
    n = len(dist)

    best_route = None
    best_len = float("inf")

    for start in range(n):
        route = nearest_neighbor(dist, start=start)
        route = insertion_local_search(dist, route)
        length = tour_length(route, dist)

        if length < best_len:
            best_len = length
            best_route = route

    return best_route


# ============================================================
# 4 大域的最適解の探索
# Held-Karp法：動的計画法による厳密解法
# ============================================================

def held_karp_exact(dist):
    """
    巡回セールスマン問題の大域的最適解を求める。

    単純な全列挙は O(n!) であり、20都市程度では現実的でない。
    ここでは Held-Karp法を用いる。
    計算量は O(n^2 * 2^n) で、厳密な最適解を返す。
    """
    n = len(dist)

    if n == 1:
        return [0]

    # 都市0を出発点として固定する。
    # 巡回路なので、出発点を固定しても最適性は失われない。
    m = n - 1
    num_masks = 1 << m
    INF = float("inf")

    # dp[mask, last]
    # 都市0から出発し、maskで表される都市集合を訪問し、
    # lastで終わるときの最短距離。
    #
    # lastは0番都市を除いた番号で管理する。
    # 実際の都市番号は last + 1。
    dp = array("d", [INF]) * (num_masks * m)
    parent = array("h", [-1]) * (num_masks * m)

    # 初期化：0番都市から各都市へ直接行く
    for last in range(m):
        mask = 1 << last
        dp[mask * m + last] = dist[0][last + 1]

    # DP本体
    for mask in range(1, num_masks):
        mm = mask

        while mm:
            lsb = mm & -mm
            last = lsb.bit_length() - 1
            prev_mask = mask ^ (1 << last)

            idx = mask * m + last

            if prev_mask != 0:
                best = INF
                best_prev = -1

                pp = prev_mask

                while pp:
                    p_lsb = pp & -pp
                    prev = p_lsb.bit_length() - 1

                    cand = (
                        dp[prev_mask * m + prev]
                        + dist[prev + 1][last + 1]
                    )

                    if cand < best:
                        best = cand
                        best_prev = prev

                    pp ^= p_lsb

                dp[idx] = best
                parent[idx] = best_prev

            mm ^= lsb

    # 最後に0番都市へ戻る
    full_mask = num_masks - 1

    best_len = INF
    best_last = -1

    for last in range(m):
        cand = dp[full_mask * m + last] + dist[last + 1][0]

        if cand < best_len:
            best_len = cand
            best_last = last

    # 経路復元
    reverse_route = []
    mask = full_mask
    last = best_last

    while last != -1:
        reverse_route.append(last + 1)

        idx = mask * m + last
        prev = parent[idx]

        mask ^= (1 << last)
        last = prev

    route = [0] + list(reversed(reverse_route))

    return route


# ============================================================
# 1つの例題に対して4手法を実行
# ============================================================

def solve_one_instance(n, seed, do_exact=True):
    points = make_points(n, seed=seed)
    dist = make_dist(points)

    results = []

    # 1-(a) 最近隣法
    start_time = time.perf_counter()
    route1 = nearest_neighbor(dist, start=0)
    time1 = time.perf_counter() - start_time
    len1 = tour_length(route1, dist)

    results.append({
        "method": "1-(a) 最近隣法",
        "length": len1,
        "time": time1,
        "route": route1
    })

    # 2-(a) 挿入近傍法
    # 指示どおり、最近隣法の結果を初期解にする
    start_time = time.perf_counter()
    init_route = nearest_neighbor(dist, start=0)
    route2 = insertion_local_search(dist, init_route)
    time2 = time.perf_counter() - start_time
    len2 = tour_length(route2, dist)

    results.append({
        "method": "2-(a) 挿入近傍法",
        "length": len2,
        "time": time2,
        "route": route2
    })

    # 3-(a) 多出発法
    # 最近隣法 + 挿入近傍法を出発都市を変えて繰り返す
    start_time = time.perf_counter()
    route3 = multi_start_nn_insertion(dist)
    time3 = time.perf_counter() - start_time
    len3 = tour_length(route3, dist)

    results.append({
        "method": "3-(a) 多出発法",
        "length": len3,
        "time": time3,
        "route": route3
    })

    # 4 大域的最適解
    exact_len = None

    if do_exact:
        start_time = time.perf_counter()
        route4 = held_karp_exact(dist)
        time4 = time.perf_counter() - start_time
        len4 = tour_length(route4, dist)
        exact_len = len4

        results.append({
            "method": "4 大域的最適解",
            "length": len4,
            "time": time4,
            "route": route4
        })

    return points, results, exact_len


# ============================================================
# 実験用
# ============================================================

def run_experiments(n_values, seeds, exact_limit=20, csv_name="tsp_results.csv"):
    """
    複数の都市数、複数の例題に対して実験する。

    n_values:
        調査する都市数のリスト

    seeds:
        例題を変えるための乱数シード

    exact_limit:
        大域的最適解を計算する最大都市数
        20程度なら計算可能だが、21以上はかなり重くなる。
    """
    all_rows = []

    for n in n_values:
        for case_id, seed in enumerate(seeds, start=1):
            do_exact = n <= exact_limit

            points, results, exact_len = solve_one_instance(
                n=n,
                seed=seed,
                do_exact=do_exact
            )

            print("=" * 80)
            print(f"都市数 n = {n}, 例題 {case_id}, seed = {seed}")

            if exact_len is not None:
                print(f"最適距離 = {exact_len:.6f}")
            else:
                print("最適距離 = 未計算")

            print("-" * 80)
            print(f"{'手法':<20} {'距離':>12} {'時間[s]':>12} {'誤差率[%]':>12}")

            for r in results:
                if exact_len is not None:
                    error_rate = (r["length"] / exact_len - 1.0) * 100.0
                    error_text = f"{error_rate:.4f}"
                else:
                    error_rate = None
                    error_text = "---"

                print(
                    f"{r['method']:<20} "
                    f"{r['length']:>12.6f} "
                    f"{r['time']:>12.6f} "
                    f"{error_text:>12}"
                )

                all_rows.append({
                    "n": n,
                    "case": case_id,
                    "seed": seed,
                    "method": r["method"],
                    "length": r["length"],
                    "time_sec": r["time"],
                    "optimal_length": exact_len,
                    "error_rate_percent": error_rate,
                    "route": route_to_string(r["route"])
                })

            print()

    # CSV保存
    with open(csv_name, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "n",
            "case",
            "seed",
            "method",
            "length",
            "time_sec",
            "optimal_length",
            "error_rate_percent",
            "route"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"結果を {csv_name} に保存しました。")


# ============================================================
# メイン処理
# ============================================================

if __name__ == "__main__":
    # 3題の例題を作る
    # seedを0,1,2にすることで、異なる都市配置を3つ作る。
    seeds = [0, 1, 2]

    # まずは20都市で3題実行する
    run_experiments(
        n_values=[20],
        seeds=seeds,
        exact_limit=20,
        csv_name="tsp_results_n20.csv"
    )

    # 都市数による影響も調べたい場合は、下のコメントを外す。
    # 20都市の大域的最適解は少し時間がかかる。
    #
    # run_experiments(
    #     n_values=[8, 10, 12, 15, 18, 20],
    #     seeds=seeds,
    #     exact_limit=20,
    #     csv_name="tsp_results_all.csv"
    # )
