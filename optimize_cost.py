"""
对每个收费站的常规车道数量和应急系统数量进行关于总成本的优化
> 数据输出中，应急系统关闭时间 指的是应急系统开启状态的最后一个小时。例如开启关闭分别为8和9，则实际开启了2小时。
"""

import pandas as pd
import pulp

import dataOutput

# 定义常量
C_x = 24 * 6 * 25.8  # 每条常规车道的成本
C_y = 414.8   # 每套应急收费系统每小时的成本
P = 3600 / (0.25 * 3.5 + 0.75 * 19)  # 每条车道每小时能处理的车流量
P_e = P  # 每套应急系统每小时能处理的车流量

def optimize_cost(D, L, R):
    # 创建问题实例
    problem = pulp.LpProblem("Minimize_Cost", pulp.LpMinimize)

    # 定义变量
    x = pulp.LpVariable("x", lowBound=1, cat='Integer')  # 常规车道数量
    y = pulp.LpVariable("y", lowBound=0, cat='Integer')  # 应急系统数量

    # 目标函数（总费用）
    problem += C_x * x + C_y * y * (R - L)

    # 约束条件
    for t in range(24 * 6):
        hour_of_day = t % 24
        if L <= hour_of_day <= R:  # 开启应急系统，应急系统和常规车道同时处理车流
            problem += (x * P + y * P_e >= D[t])
        else:
            problem += (x * P >= D[t])

    # 求解问题
    problem.solve()

    # 返回结果
    return pulp.LpStatus[problem.status], x.varValue, y.varValue, pulp.value(problem.objective)

if __name__ == '__main__':
    df = pd.read_csv("./doc/表1表2_各收费站逐小时进出车流.csv")
    stations = df["收费站"].unique()
    results = []

    for station in stations:
        D = list(map(lambda x: x * 281689 / 17542, (df[df["收费站"] == station]["入站车流"] + df[df["收费站"] == station]["出站车流"]).values))
        min_cost = 999999
        rec = []

        for L in range(24):  # 应急系统开启
            for R in range(L, 24):  # 应急系统关闭
                status, x, y, cost = optimize_cost(D, L, R)  # 求解最优

                if status == 'Optimal' and cost < min_cost:  # 如果找到更优解
                    min_cost = cost
                    rec = [x, y, L, R]

        results.append([station, rec[2], rec[3], rec[0], rec[1], min_cost])

    # 保存结果到CSV文件
    results_df = pd.DataFrame(results, columns=["收费站", "应急系统开启时间", "应急系统关闭时间", "常规车道数量", "应急系统数量", "总费用"])
    dataOutput.dump(results_df, "优化结果.csv")
