import json
from queue import Queue
import math

import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf) # np取消科学计数法显示

from scipy import optimize as op
import pulp


def readfile(filepath):
    with open(filepath, 'r') as f:
        goods_json = ""
        for line in f.readlines():
            goods_json += line
    # print(goods_json)
    return goods_json

# 用pulp库求解
# ref：
# 1.https://www.cnblogs.com/youcans/archive/2021/04/28/14714085.html#:~:text=PuLP%E6%98%AF%E4%B8%80%E4%B8%AA%E5%BC%80,%E5%90%88%E6%95%B4%E6%95%B0%E8%A7%84%E5%88%92%E9%97%AE%E9%A2%98%E3%80%82
# 2.https://www.jianshu.com/p/9be417cbfebb
def pulp_sol(goods, money, coupon, limit_buy):
    variables = []    # 变量及其取值范围
    target_coefficients = []    # 目标式子的系数
    price_coefficients = []    # 限制条件：券后商品支付价格
    coupon_coefficients = []    # 限制条件：商品消耗卡券的数量

    i = 1
    for goods in goods_list:
        # var_n = (0, limit_buy)
        # 添加变量，cat指定变量类型：Continuous-连续变量；Integer-离散变量；Binary-0/1变量
        variables.append(pulp.LpVariable(f'x{i}', lowBound=0, upBound=limit_buy, cat="Integer"))  
        target_coefficients.append(1)   # 目标函数的系数
        price_coefficients.append(goods.get("price")) # 限制条件1的系数
        coupon_coefficients.append(goods.get("coupon"))  #限制条件2的系数
        i += 1

    # 设定目标场景，最大化目标还是最小化目标
    m = pulp.LpProblem(sense=pulp.LpMaximize)
    # 添加目标函数
    m += pulp.lpDot(target_coefficients, variables)
    # 添加限制条件1
    m += (pulp.lpDot(price_coefficients, variables) <= money)
    # 添加限制条件2
    m += (pulp.lpDot(coupon_coefficients, variables) <= coupon)

    m.solve()
    print(pulp.value(m.objective))
    print([pulp.value(var) for var in variables])
    # x = pulp.LpVariable()
    print(m)
    print(variables)



# scipy_sol
# 变量为离散型，因此问题为整数线性规划
# 整数规划一般方法：1. 分支界定法；2.割平面法；3.0/1规划及隐枚举法；4.指派问题
# 这里尝试使用分支界定法
def scipy_sol(goods, money, coupon, limit_buy):
    variables = []    # 变量及其取值范围
    target_coefficients = []    # 目标式子的系数
    price_coefficients = []    # 限制条件：券后商品支付价格
    coupon_coefficients = []    # 限制条件：商品消耗卡券的数量

    for goods in goods_list:
        var_n = (0, limit_buy)
        variables.append(var_n) 
        target_coefficients.append(-1)   # scipy库默认为最小化函数，所以要取反
        price_coefficients.append(goods.get("price")) # 不等式限制条件默认<=，所以这里系数不用取反
        coupon_coefficients.append(goods.get("coupon"))

    c = np.array(target_coefficients)
    A_ub = np.array([price_coefficients, coupon_coefficients])

    rhs = [money, coupon]
    B_ub = np.array(rhs)
    print(c.shape)
    print(A_ub.shape)
    print(B_ub.shape)

    # , options={"tol" : 1e-12}
    # 求解之后，数量不是整数，还需要做整数规划（分支定界算法）
    res = op.linprog(c=c,A_ub=A_ub, b_ub=B_ub, bounds=variables)
    print(res)
    # print(len(res.x))
    if not res.success:
        raise ValueError('当前问题不可行')
    # 会死循环
    integer_solve(res, A_ub, B_ub, c, variables)

    
# 整数规划-分支界定法，有点问题，再研究
# ref：
# 原理：https://blog.csdn.net/qq_28087491/article/details/111662130
# 实现：https://www.jianshu.com/p/70794d091b39
def integer_solve(res, A_ub, B_ub, c, variables):
    # 下界：该情境中下不可能出现小于0的情况
    LOWER_BOUND = 0
    # 上界：为松弛条件（不考虑整数的情况）下的最优解
    UPPER_BOUND = -res.fun
    # 最优解及其取值
    opt_val = None
    opt_x = None

    # Queue：https://blog.csdn.net/weixin_43533825/article/details/89155648
    # Q是同步的，线程安全的队列类，可以直接再多线程中使用
    # Queue是FIFO，LifoQueue是FILO(First In Last Out)
    Q = Queue()
    # 将松弛条件的解和约束条件放入队列
    Q.put((res, A_ub, B_ub))

    while not Q.empty():
        # 取出当前问题
        res, A_ub, B_ub = Q.get(block=False) # 不阻塞
        # print(-res.fun)

        # 如果当前最优值小于下界，则排除该区域(剪枝，定界过程)
        if -res.fun < LOWER_BOUND:
            continue
            
        # 若结果 x 中全为整数，则尝试更新全局下界、全局最优值和最优解
        if all(list(map(lambda f: min(f-math.floor(f), math.ceil(f)-f)<1e-8, res.x))):
            if LOWER_BOUND < -res.fun:
                LOWER_BOUND = -res.fun
            
            print("Lower bound: ", LOWER_BOUND)
            
            if opt_val is None or opt_val < -res.fun:
                opt_val = -res.fun
                opt_x = res.x
            continue
        
        else:    # 进行分支
            # 寻找松弛条件下，x中第一个不是整数的，取其下标
            index = 0
            # enumerate: 将一个可遍历的数据对象组合成一个索引序列，同时列出数据和下标
            for i, x in enumerate(res.x):
                # print((x-math.floor(x)<1.0e-8) or (math.ceil(x)-x<1.0e-8))
                if not min(x-math.floor(x), math.ceil(x)-x) < 1e-8: # 与边界只差小于某个阈值范围视为整数
                    print(f'x{i}={x} || \
                        ceil(x)={math.ceil(x)} || \
                        floor(x)={math.floor(x)} || \
                        x-floor={x-math.floor(x)} || \
                        ceil-x={math.ceil(x)-x}')
                    index = i
                    break
            
            # 构建新的约束条件（分割）
            # np.zeros(返回1个给定维度和类型的，用0填充的数组)
            # 分支1-新条件：x_i >= ceil(A)，需要取反，所以系数是-1
            new_con1 = np.zeros(A_ub.shape[1])
            new_con1[index] = -1
            # 分支2-新条件：x_i <= floor(A)，小于等于不需要取反
            new_con2 = np.zeros(A_ub.shape[1])
            new_con2[index] = 1 
            
            # 插入新条件到原限制条件
            # numpy.insert可以有三个参数（arr，obj，values），也可以有4个参数（arr，obj，values，axis）：
            # 第一个参数arr是一个数组，可以是一维的也可以是多维的，在arr的基础上插入元素
            # 第二个参数obj是元素插入的位置
            # 第三个参数values是需要插入的数值
            # 第四个参数axis是指示在哪一个轴上对应的插入位置进行插入
            new_A_ub1 = np.insert(A_ub, A_ub.shape[0], new_con1, axis=0)
            new_A_ub2 = np.insert(A_ub, A_ub.shape[0], new_con2, axis=0)
            new_B_ub1 = np.insert(B_ub, B_ub.shape[0], -math.ceil(res.x[index]), axis=0) # 大于等于，需要取反
            new_B_ub2 = np.insert(B_ub, B_ub.shape[0], math.floor(res.x[index]), axis=0) 

            # 将新约束条件的求解加入队列，先加最优值大的那一支
            res1 = op.linprog(c=c, A_ub=new_A_ub1, b_ub=new_B_ub1, bounds=variables)
            res2 = op.linprog(c=c, A_ub=new_A_ub2, b_ub=new_B_ub2, bounds=variables)
            
            if not res1.success and res2.success:
                # print(f"2 ok, status={res2.status}")
                
                Q.put((res2, new_A_ub2, new_B_ub2))
            elif not res2.success and res1.success:
                # print(f"1 ok, status={res1.status}")
                Q.put((res1, new_A_ub1, new_B_ub2))
            elif res1.success and res2.success:
                # print(f"both ok, 1 status={res1.status}, 2 status={res2.status}")
                if -res1.fun > -res2.fun:
                    Q.put((res1, new_A_ub1, new_B_ub1))
                    Q.put((res2, new_A_ub2, new_B_ub2))
                else:
                    Q.put((res2, new_A_ub2, new_B_ub2))
                    Q.put((res1, new_A_ub1, new_B_ub1))
    print(opt_val, opt_x)
    return (opt_val, opt_x)



if __name__ == "__main__":
    goods_json = readfile("test.json")
    goods_list = json.loads(goods_json)
    print(len(goods_list))
    
    limit_buy = 5    # 假设每个商品只能买5件
    total_money = 200000
    total_coupon = 10000
    # rhs = [total_money, total_coupon]

    scipy_sol(goods_list, total_money, total_coupon, limit_buy)
    # pulp_sol(goods_list, total_money, total_coupon, limit_buy)




