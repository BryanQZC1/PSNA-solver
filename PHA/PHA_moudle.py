import numpy as np
import time
from typing import Callable, List, Tuple, Dict, Optional, Union, Any


class SolverBase:
    """求解器基类，提供通用功能"""
    
    def __init__(self, 
             stage_operators: List[Callable], 
             n_dims: List[int],
             random_variables: List[List],
             projection_func: Callable,
             step_size: Union[float, Callable],
             max_iterations: int, 
             epsilon: float,
             max_time: float,
             verbose: bool,
             projection_grad_max_iter: int,
             random_variable_probs: Optional[List[List]] = None):
        """初始化求解器基类"""
        self.stage_operators = stage_operators
        self.n_dims = n_dims
        self.random_variables = random_variables
        self.total_n = sum(n_dims)
        self.stage_number = len(n_dims)
        self.scenario_number = np.prod([len(stage) for stage in random_variables])
        
        # 处理随机变量概率
        if random_variable_probs is None:
            self.random_variable_probs = []
            for stage_vars in random_variables:
                stage_probs = [1.0 / len(stage_vars)] * len(stage_vars)
                self.random_variable_probs.append(stage_probs)
        else:
            self.random_variable_probs = random_variable_probs
        
        self.projection_func = projection_func
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.max_time = max_time
        self.verbose = verbose
        self.projection_grad_max_iter = projection_grad_max_iter
        
        # 生成场景树和场景概率
        self.scenario_tree, self.scenario_probs = self._generate_scenario_tree_with_probs()
    
    def _generate_scenario_tree_with_probs(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成场景树和场景概率"""
        total_scenarios = self.scenario_number
        scenario_tree = np.zeros((self.stage_number, total_scenarios))
        scenario_probs = np.ones(total_scenarios)
        
        stage_sizes = [len(stage) for stage in self.random_variables]
        
        indices = [0] * self.stage_number
        for scenario_idx in range(total_scenarios):
            for stage_idx in range(self.stage_number):
                var_idx = indices[stage_idx]
                scenario_tree[stage_idx, scenario_idx] = self.random_variables[stage_idx][var_idx]
                scenario_probs[scenario_idx] *= self.random_variable_probs[stage_idx][var_idx]
            
            for stage_idx in range(self.stage_number-1, -1, -1):
                indices[stage_idx] += 1
                if indices[stage_idx] < stage_sizes[stage_idx]:
                    break
                indices[stage_idx] = 0
        
        # 确保概率和为1
        prob_sum = np.sum(scenario_probs)
        if abs(prob_sum - 1.0) > 1e-10 and prob_sum > 0:
            scenario_probs = scenario_probs / prob_sum
            
        return scenario_tree, scenario_probs
    
    def _combined_operator(self, x: np.ndarray, random_var: np.ndarray) -> np.ndarray:
        """将所有阶段的算子组合成一个单一的算子函数"""
        result = []
        
        # 从平面数组x中提取各阶段变量
        stage_vars = []
        start_idx = 0
        for dim in self.n_dims:
            stage_vars.append(x[start_idx:start_idx+dim])
            start_idx += dim
        
        # 将随机变量转换为列表，保持原始形式
        xi = random_var.tolist()
        
        # 按照三个阶段的调用约定处理
        F1 = self.stage_operators[0]
        F1_args = [stage_vars[0], stage_vars[1]]  # [x1, x2]
        result.append(F1(F1_args, xi))
        
        F2 = self.stage_operators[1]
        F2_args = [stage_vars[0], stage_vars[1], stage_vars[2]]  # [x1, x2, x3]
        result.append(F2(F2_args, xi))
        
        F3 = self.stage_operators[2]
        F3_args = [stage_vars[0], stage_vars[1], stage_vars[2]]  # [x1, x2, x3]
        result.append(F3(F3_args, xi))
        
        return np.concatenate(result)
    
    def _compute_stepsize(self, scenario: np.ndarray, iteration: int) -> float:
        """计算步长，支持固定步长或函数形式的步长"""
        if callable(self.step_size):
            return self.step_size(scenario, iteration)
        return self.step_size
        
    def _projection_gradient(self, 
                             gradient_fun: Callable, 
                             stepsize: float, 
                             x0: np.ndarray, 
                             epsilon: float = None, 
                             max_iter: int = None) -> np.ndarray:
        """投影梯度法求解子问题"""
        if epsilon is None:
            epsilon = self.epsilon * 0.1
        if max_iter is None:
            max_iter = self.projection_grad_max_iter
            
        x = np.asarray(x0, dtype=float)
        
        for k in range(max_iter):
            previous_x = x.copy()
            grad = gradient_fun(x)
            x = self.projection_func(x, stepsize, grad)
            
            change = np.linalg.norm(x - previous_x)
            if change <= epsilon:
                break
                
        return x
        
    def calculate_residual(self, solution: np.ndarray) -> float:
        """
        计算解的残差，使用PHA已经计算好的每个场景下的最优解
        
        参数:
            solution: 第一阶段解或完整解（会自动提取第一阶段部分）
            
        返回:
            float: 残差值
        """
        # 确保是第一阶段解
        if solution.size > self.n_dims[0]:
            x_first_stage = solution[:self.n_dims[0]]
        else:
            x_first_stage = solution
        
        # 初始化期望值
        expectation_F1 = np.zeros_like(x_first_stage)
        
        # 遍历所有场景
        for i in range(self.scenario_number):
            scenario = self.scenario_tree[:, i]
            scenario_probability = self.scenario_probs[i]
            
            # 提取当前场景的第二阶段解
            # 这里假设solution包含了所有场景的解，按照场景顺序排列
            start_idx = i * self.total_n + self.n_dims[0]
            end_idx = start_idx + self.n_dims[1]
            x_second_stage = solution[start_idx:end_idx]
            
            # 计算F1值，使用当前场景的最优第二阶段解
            F1_value = self.stage_operators[0]([x_first_stage, x_second_stage], scenario.tolist())
            
            # 按概率累积期望值
            expectation_F1 += scenario_probability * F1_value
        
        # 应用投影
        projected_point = self.projection_func(x_first_stage, 1.0, expectation_F1)
        
        # 计算残差 = ||x₁ - Proj(x₁ - E[F₁])||
        residual = np.linalg.norm(x_first_stage - projected_point)
        
        return residual


class PHAModule(SolverBase):
    """渐进对冲算法(Progressive Hedging Algorithm, PHA)模块"""
    
    def __init__(self, 
             stage_operators: List[Callable], 
             n_dims: List[int],
             random_variables: List[List],
             projection_func: Callable,
             initial_points: List[np.ndarray],
             initial_multiplier: Optional[np.ndarray],
             penalty_parameter: float,
             step_size: Union[float, Callable],
             max_iterations: int, 
             epsilon: float,
             max_time: float,
             verbose: bool,
             previous_solution_as_initial: bool,
             projection_grad_max_iter: int,
             random_variable_probs: Optional[List[List]] = None):
        """初始化PHA模块"""
        # 调用父类初始化
        super().__init__(
            stage_operators=stage_operators,
            n_dims=n_dims,
            random_variables=random_variables,
            projection_func=projection_func,
            step_size=step_size,
            max_iterations=max_iterations,
            epsilon=epsilon,
            max_time=max_time,
            verbose=verbose,
            projection_grad_max_iter=projection_grad_max_iter,
            random_variable_probs=random_variable_probs
        )
        
        # 处理初始点
        if initial_points is None:
            self.initial_points = [np.zeros(dim) for dim in n_dims]
            self.x0 = np.zeros(self.scenario_number * self.total_n)
        else:
            self.initial_points = initial_points
            self.x0 = np.zeros(self.scenario_number * self.total_n)
            for s in range(self.scenario_number):
                start_idx = s * self.total_n
                for i, point in enumerate(initial_points):
                    dim_start = sum(n_dims[:i])
                    dim_end = dim_start + n_dims[i]
                    self.x0[start_idx + dim_start:start_idx + dim_end] = point
        
        # 处理初始拉格朗日乘子
        if initial_multiplier is None:
            self.w0 = np.zeros(self.scenario_number * self.total_n)
        else:
            self.w0 = initial_multiplier
        
        self.penalty_parameter = penalty_parameter
        self.previous_solution_as_initial = previous_solution_as_initial
        
        # 初始化统计信息
        self.stats = {
            "iterations": 0,
            "time": 0,
            "convergence": False,
            "error": None,
            "residual": None
        }
    
    def solve(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """求解多阶段随机规划问题"""
        iter_count = 0
        x0 = self.x0
        w0 = self.w0
        r = self.penalty_parameter
        
        # 计算累积乘积，用于重构决策变量
        number = [len(stage) for stage in self.random_variables]
        cumulative_products = np.cumprod(number)
        
        # 初始化解数组
        x_solver_scenario = [np.zeros((self.scenario_number, self.n_dims[i])) for i in range(self.stage_number)]
        x_solver_scenario_mean = [np.zeros_like(arr) for arr in x_solver_scenario]
        
        time_start = time.time()
        
        while iter_count < self.max_iterations:
            # 检查运行时间是否超过限制
            if time.time() - time_start > self.max_time:
                if self.verbose:
                    print(f"达到最大运行时间 {self.max_time} 秒，停止计算")
                self.stats["error"] = "超过最大运行时间"
                break
            
            # 为每个场景求解子问题
            solutions = np.zeros((self.scenario_tree.shape[1], self.total_n))
            previous_solution = None
            
            for i in range(self.scenario_tree.shape[1]):
                scenario = self.scenario_tree[:, i]
                step = self._compute_stepsize(scenario, iter_count)
                
                # 定义子问题目标函数（添加拉格朗日项和惩罚项）
                def subproblem_objective(x):
                    start_idx = i * self.total_n
                    end_idx = (i + 1) * self.total_n
                    return self._combined_operator(x, scenario) + w0[start_idx:end_idx] + r * (x - x0[start_idx:end_idx])
                
                # 确定初始点
                if self.previous_solution_as_initial and previous_solution is not None:
                    initial_x = previous_solution
                else:
                    initial_x = np.zeros(self.total_n)
                
                # 使用投影梯度法求解子问题
                solutions[i] = self._projection_gradient(
                    subproblem_objective,
                    step,
                    initial_x,
                    epsilon=self.epsilon * 0.1,
                    max_iter=self.projection_grad_max_iter
                )
                
                # 保存当前解作为下一个场景的初始点
                previous_solution = solutions[i]
            
            # 将解重构到每个阶段
            for stage in range(self.stage_number):
                start = sum(self.n_dims[:stage])
                end = start + self.n_dims[stage]
                x_solver_scenario[stage] = solutions[:, start:end]
            
            # 计算每个阶段的平均解（非叶节点）
            for stage in range(self.stage_number - 1):
                group_size = self.scenario_number // cumulative_products[stage]
                x_solver_scenario_mean[stage] = np.mean(
                    x_solver_scenario[stage].reshape(-1, group_size, self.n_dims[stage]), 
                    axis=1
                ).repeat(group_size, axis=0)
            
            # 最后一个阶段不需要平均
            x_solver_scenario_mean[self.stage_number-1] = x_solver_scenario[self.stage_number-1]
            
            # 重建x0和x_hat，先阶段后场景，对计算的结果平均和重排
            x0_mean = []
            x_hat = []
            
            for i in range(self.scenario_number):
                for stage in range(self.stage_number):
                    x0_mean.append(x_solver_scenario_mean[stage][i])
                    x_hat.append(x_solver_scenario[stage][i])
            
            x_hat = np.array(x_hat)
            x0_mean = np.array(x0_mean)
            x0_new = x0_mean.flatten()
            
            # 更新拉格朗日乘子
            w0 = w0 + r * (x_hat.flatten() - x0_new)
            
            # 计算收敛误差
            error = np.linalg.norm(x0_new - x0)
            
            # 如果达到收敛精度，则停止
            if error <= self.epsilon:
                if self.verbose:
                    print(f"达到收敛精度 {self.epsilon}，停止计算")
                self.stats["convergence"] = True
                break
            
            # 更新初始点
            x0 = x0_new
            iter_count += 1
            
            # 输出迭代信息
            if self.verbose and iter_count % 10 == 0:
                print(f"迭代 {iter_count}: 误差 = {error:.6e}")
        
        # 计算运行时间
        time_end = time.time() - time_start
        
        # 更新统计信息
        self.stats["iterations"] = iter_count
        self.stats["time"] = time_end
        
        # 计算残差 - 使用场景的详细解而不仅仅是第一阶段
        # 保存所有场景的最优解到一个平坦数组
        all_scenario_solutions = np.zeros(self.scenario_number * self.total_n)
        for i in range(self.scenario_number):
            start_idx = i * self.total_n
            for stage in range(self.stage_number):
                stage_start = start_idx + sum(self.n_dims[:stage])
                stage_end = stage_start + self.n_dims[stage]
                all_scenario_solutions[stage_start:stage_end] = x_solver_scenario[stage][i]
        
        # 使用改进的残差计算方法
        self.stats["residual"] = self.calculate_residual(all_scenario_solutions)
        
        # 提取第一阶段解
        first_stage_solution = x0[:self.n_dims[0]]
        
        return first_stage_solution, self.stats


class ExternalSolutionEvaluator:
    """外部解残差计算器"""
    
    def __init__(self, pha_solver: PHAModule):
        """初始化外部解评估器"""
        self.pha_solver = pha_solver
        self.stage_operators = pha_solver.stage_operators
        self.n_dims = pha_solver.n_dims
        self.random_variables = pha_solver.random_variables
        self.random_variable_probs = pha_solver.random_variable_probs
        self.projection_func = pha_solver.projection_func
        self.step_size = pha_solver.step_size
        self.max_iterations = pha_solver.max_iterations
        self.epsilon = pha_solver.epsilon
        self.max_time = pha_solver.max_time
        self.verbose = pha_solver.verbose
        self.projection_grad_max_iter = pha_solver.projection_grad_max_iter
        self.penalty_parameter = pha_solver.penalty_parameter
        
        # 验证是三阶段问题
        if len(self.n_dims) != 3 or len(self.stage_operators) != 3:
            raise ValueError("外部解残差计算器目前只支持三阶段问题")
    
    def _solve_subproblem_for_xi2(self, first_stage_solution: np.ndarray, xi2_val: float) -> np.ndarray:
        """针对特定第二阶段随机变量值构造并求解子问题，使用PHA方法"""
        x1 = first_stage_solution
        
        # 子问题的维度（只包含第二和第三阶段）
        subproblem_dims = self.n_dims[1:3]  # [dim2, dim3]
        
        # 子问题的随机变量
        subproblem_random_vars = [
            [xi2_val],  # 固定的第二阶段随机变量
            self.random_variables[2]  # 第三阶段随机变量
        ]
        
        # 子问题的随机变量概率
        subproblem_random_probs = [
            [1.0],  # 第二阶段固定，概率为1
            self.random_variable_probs[2]  # 第三阶段概率保持不变
        ]
        
        # 创建子问题专用的算子
        def modified_F2(args, xi):
            # 在子问题中，args是[x2, x3]，需要将x1添加到前面
            xi_full = [self.random_variables[0][0], xi2_val, xi[1]]
            return self.stage_operators[1]([x1, args[0], args[1]], xi_full)
        
        def modified_F3(args, xi):
            # 在子问题中，args是[x2, x3]，需要将x1添加到前面
            xi_full = [self.random_variables[0][0], xi2_val, xi[1]]
            return self.stage_operators[2]([x1, args[0], args[1]], xi_full)
        
        # 创建一个简化的PHAModule类，专用于求解子问题
        class SubproblemPHA(SolverBase):
            def _combined_operator(self2, x, random_var):
                result = []
                # 从平面数组x中提取各阶段变量
                stage_vars = []
                start_idx = 0
                for dim in subproblem_dims:
                    stage_vars.append(x[start_idx:start_idx+dim])
                    start_idx += dim
                
                # 将随机变量转换为列表
                xi = random_var.tolist()
                
                # 子问题中只有两个算子和两个阶段变量
                result.append(modified_F2(stage_vars, xi))
                result.append(modified_F3(stage_vars, xi))
                
                return np.concatenate(result)
        
        # 创建子问题的PHA求解器
        subproblem_pha = PHAModule(
            stage_operators=[modified_F2, modified_F3],  # 这里的算子顺序不重要，会被覆盖
            n_dims=subproblem_dims,
            random_variables=subproblem_random_vars,
            projection_func=self.projection_func,
            initial_points=[np.zeros(dim) for dim in subproblem_dims],
            initial_multiplier=None,
            penalty_parameter=self.penalty_parameter,
            step_size=self.step_size,
            max_iterations=max(50, self.max_iterations // 2),
            epsilon=self.epsilon,
            max_time=self.max_time / 10,
            verbose=self.verbose,
            previous_solution_as_initial=True,
            projection_grad_max_iter=self.projection_grad_max_iter,
            random_variable_probs=subproblem_random_probs
        )
        
        # 使用我们重写的_combined_operator方法
        subproblem_pha._combined_operator = SubproblemPHA._combined_operator.__get__(subproblem_pha, PHAModule)
        
        # 求解子问题，返回第二阶段最优解
        x2_solution, stats = subproblem_pha.solve()
        
        if self.verbose:
            print(f"子问题求解完成: 迭代次数={stats['iterations']}, 残差={stats['residual']:.6e}")
        
        return x2_solution
    
    def solve_all_second_stage_subproblems(self, first_stage_solution: np.ndarray) -> Dict[int, np.ndarray]:
        """针对所有第二阶段随机变量支撑点求解子问题"""
        # 确保是第一阶段解
        if first_stage_solution.size > self.n_dims[0]:
            x_first_stage = first_stage_solution[:self.n_dims[0]]
        else:
            x_first_stage = first_stage_solution.copy()
        
        # 获取第二阶段随机变量
        second_stage_vars = self.random_variables[1]
        
        # 存储每个第二阶段随机变量对应的解
        stage2_solutions = {}
        
        # 开始计时
        start_time = time.time()
        total_time_allowed = self.max_time
        
        if self.verbose:
            print(f"开始求解所有第二阶段子问题，共有 {len(second_stage_vars)} 个支撑点...")
        
        # 为每个第二阶段随机变量支撑点求解对应的子问题
        for idx, xi2_val in enumerate(second_stage_vars):
            # 检查是否超时
            if time.time() - start_time > total_time_allowed:
                if self.verbose:
                    print(f"求解子问题超时，已完成 {idx}/{len(second_stage_vars)} 个支撑点")
                break
                
            if self.verbose:
                print(f"求解第 {idx+1}/{len(second_stage_vars)} 个支撑点的子问题，ξ₂ = {xi2_val}")
            
            # 为特定第二阶段随机变量求解子问题
            solution = self._solve_subproblem_for_xi2(x_first_stage, xi2_val)
            
            # 保存解
            stage2_solutions[idx] = solution
            
            if self.verbose:
                print(f"支撑点 {idx+1} 子问题求解完成")
        
        return stage2_solutions
    
    def calculate_external_residual(self, first_stage_solution: np.ndarray) -> float:
        """计算外部解的残差"""
        # 确保是第一阶段解
        if first_stage_solution.size > self.n_dims[0]:
            x_first_stage = first_stage_solution[:self.n_dims[0]]
        else:
            x_first_stage = first_stage_solution.copy()
        
        # 步骤1: 求解所有第二阶段子问题
        stage2_solutions = self.solve_all_second_stage_subproblems(x_first_stage)
        
        if not stage2_solutions:
            if self.verbose:
                print("求解子问题失败，无法计算残差")
            return float('inf')
        
        # 步骤2: 计算第一阶段算子的期望值
        if self.verbose:
            print("第二步：计算第一阶段算子的期望值...")
        
        # 初始化期望F1值
        expectation_F1 = np.zeros_like(x_first_stage)
        
        # 获取第二阶段随机变量及其概率
        second_stage_vars = self.random_variables[1]
        second_stage_probs = self.random_variable_probs[1]
        
        # 计算第一阶段算子的期望值
        for i, xi2_idx in enumerate(stage2_solutions.keys()):
            xi2_val = second_stage_vars[xi2_idx]
            xi2_prob = second_stage_probs[xi2_idx]
            
            # 获取对应的第二阶段解
            x2_solution = stage2_solutions[xi2_idx]
            
            # 对第一阶段随机变量
            for first_stage_idx, xi1_val in enumerate(self.random_variables[0]):
                xi1_prob = self.random_variable_probs[0][first_stage_idx]
                
                # 构建随机变量向量
                xi = [xi1_val, xi2_val]
                
                # 计算F1值
                F1_value = self.stage_operators[0]([x_first_stage, x2_solution], xi)
                
                # 按概率累积期望值
                expectation_F1 += xi1_prob * xi2_prob * F1_value
            
            if self.verbose and (i+1) % max(1, len(stage2_solutions)//10) == 0:
                print(f"残差计算进度: {i+1}/{len(stage2_solutions)}")
        
        # 应用投影
        projected_point = self.projection_func(x_first_stage, 1.0, expectation_F1)
        
        # 计算残差 = ||x₁ - Proj(x₁ - E[F₁(x₁, x₂(ξ₂), ξ)])||
        residual = np.linalg.norm(x_first_stage - projected_point)
        
        if self.verbose:
            print(f"外部解残差: {residual:.6e}")
        
        return residual
