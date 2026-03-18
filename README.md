下面是**完整体的 Codex 工程说明书**。
目标不是“介绍思路”，而是让 Codex 能据此**直接搭建一个可运行、可扩展、可逐版本验证的 Python 仿真工程**。

你可以直接复制给 Codex。

Bluetooth Channel Sounding 多设备并发测距仿真系统工程说明书

一、工程目标

本工程用于构建一个基于 Python 的 Bluetooth Channel Sounding 仿真系统，围绕 PBR（Phase-Based Ranging）测距机制，从理想单链路逐步扩展到多设备并发、噪声、多径和室内衰落环境，最终形成一个可批量实验、可统计分析、可输出图表和数据表的完整仿真平台。

本工程有两个目标层次：

1. 标准测距能力复现
   复现 Bluetooth Channel Sounding 中基于相位的单设备测距基础模型，包括多频点相位建模、往返传播建模、噪声影响分析、随机相位影响分析。

2. 论文核心创新验证
   构建“一个主设备 + 多个从设备”的并发测距模型，验证在多个从设备同时返回信号时，主设备是否仍能通过目标链路匹配补偿和非目标链路失配抑制的方法，恢复目标设备距离。

二、工程核心背景与理论要求

2.1 标准一对一 PBR 的基本原理

在 Bluetooth Channel Sounding 的 PBR 中，主设备和从设备围绕某个频点进行双向相位测量。对于第 k 个频点，单链路接收信号可抽象为：

H(f_k) = α · exp(-j·2π·f_k·τ) · exp(j·ψ_k)

其中：

* α：链路幅度
* τ：传播相关时延，往返模型中 τ = 2d/c
* ψ_k：链路相位偏置，来源于 Initiator 和 Reflector 两端本振/PLL 等因素

单侧测得的相位不是纯传播相位，因此一对一标准场景中需要结合双向 peer 测量信息消除链路相位偏置，提取与传播距离相关的净相位，再利用多频点相位-频率斜率估计距离。

本工程在 V0-V3 阶段主要复现这种“测距物理内核”，但不要求完整复现蓝牙协议栈。

2.2 标准方法只能一对一的原因

标准一对一 PBR 成立的前提是：

* 当前只有一个 Reflector 对应当前测量链路
* Initiator 与 Reflector 的测量数据可唯一配对
* 链路相位偏置可通过双向 peer 关系消除

当多个 Reflector 同时返回时，主设备端接收的是多路复信号叠加：

H(f_k) = Σ(i=1..K) α_i · exp(-j·2π·f_k·τ_i) · exp(j·ψ_i,k)

这时接收相位不再属于某个单独链路，而是多个复向量叠加后的非线性结果。不同设备的 peer 对应关系被破坏，标准一对一方法无法直接使用，因此标准实现通常采用时分测距。

2.3 本课题的一对多并发核心思想

本工程必须实现并验证如下论文核心机制：

* 允许多个 Reflector 在同一测量周期内同时返回
* 主设备接收到多个链路的叠加信号
* 不再依赖标准一对一 peer 的唯一配对消除
* 转而采用“目标链路匹配补偿 + 非目标链路失配抑制”的方法进行目标设备恢复

对于目标设备 i，构造目标补偿项：

C_i(f_k) = exp(+j·2π·f_k·τ_i_ref)

其中 τ_i_ref 可以是真实目标时延、待搜索假设时延、或某种估计值。

补偿后的接收信号为：

H_tilde_i(f_k) = H(f_k) · C_i(f_k)

对于目标设备，其传播项被对齐，补偿后在多频点上表现出更强的一致性。对于非目标设备，由于其真实时延不等于目标时延，其补偿后仍保留残余频率相关相位项，因此表现为失配干扰项。在多频点、多次测量或统计处理中，目标设备可被增强，非目标设备被抑制。

本工程必须围绕这一思想设计 V4 及后续版本。

三、工程实现总要求

3.1 基本要求

1. 使用 Python 3 实现
2. 依赖尽量简单，优先使用：

   * numpy
   * scipy
   * pandas
   * matplotlib
   * pathlib
   * json
3. 所有实验可通过脚本或命令行单独运行
4. 所有参数必须可配置
5. 所有实验结果必须可复现（支持随机种子）
6. 所有图和表必须自动保存
7. 所有实验必须有统一输出目录结构
8. 所有模块需具备清晰输入输出，不允许把全部逻辑堆在一个脚本里

3.2 工程风格要求

1. 模块化设计
2. 函数和类命名清晰
3. 对关键公式给出注释
4. 每个实验脚本只负责组装场景和调用模块，不直接写底层数学细节
5. 所有实验结果都记录参数快照
6. 优先保证正确性，其次再考虑计算速度

3.3 输出要求

每个实验运行后至少输出：

1. 图像文件 png
2. 数据表 csv
3. 配置文件 json
4. 结果摘要 txt 或 json

四、项目目录结构要求

Codex 必须按如下目录组织工程：

project_root/
README.md
requirements.txt
main.py
configs/
default.json
v0.json
v1.json
v2.json
v3.json
v4.json
v5.json
v6.json
src/
**init**.py
constants.py
utils.py
frequency_plan.py
scenarios.py
signal_models.py
channel_models.py
impairments.py
target_matching.py
estimators.py
metrics.py
monte_carlo.py
plotting.py
io_utils.py
experiments/
exp_v0.py
exp_v1.py
exp_v2.py
exp_v3.py
exp_v4.py
exp_v5.py
exp_v6.py
results/
v0/
v1/
v2/
v3/
v4/
v5/
v6/
tests/
test_signal_models.py
test_estimators.py
test_metrics.py

五、模块职责定义

5.1 constants.py

定义工程中使用的常数，例如：

* 光速 c = 3e8
* 默认蓝牙频段起始频率
* 默认频点步进
* 默认随机种子
* 默认 Monte Carlo 次数

5.2 utils.py

提供通用函数，例如：

* db_to_linear
* linear_to_db
* set_random_seed
* complex_awgn
* ensure_dir
* normalize_complex_signal
* unwrap_phase_safe

5.3 frequency_plan.py

负责频点生成。

必须提供函数：

1. build_frequency_grid(f_start, f_step, n_freqs)
   输出频点数组

2. validate_frequency_spacing(freqs, tau_max)
   用于检查频点间隔是否可能导致相位展开困难

5.4 scenarios.py

负责生成仿真场景参数。

必须支持以下场景生成：

1. 单设备场景

   * 距离
   * 幅度
   * 固定相位偏置

2. 多设备场景

   * 设备数 K
   * 每个设备距离 d_i
   * 每个设备幅度 α_i
   * 每个设备随机相位策略
   * 设备空间分布方式（均匀、随机、指定）

3. 多径场景

   * 每条主链路的多径数
   * 各径时延
   * 各径增益

4. 室内信道场景

   * Rician K 因子
   * Rayleigh 开关
   * realization 数量

5.5 signal_models.py

这是核心模块之一。

必须实现以下函数：

1. single_link_frequency_response(freqs, distance, amplitude=1.0, phase_offset=0.0, round_trip=True)
   生成单设备理想复频响

2. multi_link_frequency_response(freqs, distances, amplitudes, phase_offsets=None, round_trip=True)
   生成多设备叠加复频响

3. repeated_measurements_single_link(freqs, distance, n_repeats, amplitude=1.0, random_phase=False, round_trip=True)
   生成单设备多次重复测量结果

4. repeated_measurements_multi_link(freqs, distances, amplitudes, n_repeats, random_phase=True, round_trip=True)
   生成多设备多次重复测量结果

注意：

* random_phase 必须表示链路相位偏置的变化，而不是人为“创造新物理”
* 若启用 random_phase，应对每个设备、每个频点、每次测量生成相位偏置项

5.6 channel_models.py

负责对理想信号施加信道影响。

必须支持：

1. add_awgn(signal, snr_db, rng)
2. apply_single_reflector_multipath(freqs, base_distance, extra_paths)
   extra_paths 包含：

   * delay_offset
   * amplitude
   * phase
3. apply_generic_multipath(freqs, path_delays, path_gains, path_phases)
4. apply_rician_fading(signal, k_factor, rng)
5. apply_rayleigh_fading(signal, rng)

要求：

* 可对单次测量和批量测量同时适用
* 输入输出保持 numpy 复数组格式

5.7 impairments.py

本工程中不做复杂硬件误差建模，但保留最基本的可扩展接口。

当前只允许放置：

* 随机链路相位偏置生成
* 固定相位偏置生成
* 频点依赖相位偏置辅助函数

不要实现过重的 CFO / SFO / IQ imbalance 复杂硬件链路，除非是轻量占位接口。

5.8 target_matching.py

这是本工程最关键模块之一，必须体现论文创新。

必须实现：

1. build_target_compensation(freqs, hypothesized_distance, round_trip=True)
   生成目标假设距离对应的相位补偿项

2. apply_target_compensation(response, freqs, hypothesized_distance, round_trip=True)
   对复频响进行目标补偿

3. coherent_score_after_compensation(response, freqs, hypothesized_distance, round_trip=True)
   定义一个补偿后的目标一致性评分指标

4. scan_distance_hypotheses(response, freqs, distance_grid, round_trip=True)
   对距离假设网格进行搜索，输出每个距离假设下的评分

5. estimate_target_distance_by_scan(response, freqs, distance_grid, round_trip=True)
   从扫描评分中找最大值对应的目标距离

必须保证：

* 能处理单次频响
* 能处理重复测量后的频响
* 能处理多设备叠加场景

这里不要只做简单直线拟合，要明确实现“目标匹配补偿”思路。

5.9 estimators.py

负责实现多种距离估计器，供不同版本调用。

必须至少实现两类方法：

A. 相位斜率法（适合 V0/V1/V2 单设备）

1. estimate_distance_by_phase_slope(response, freqs, round_trip=True)

步骤：

* 提取相位
* 相位展开
* 线性拟合
* 求时延
* 求距离

B. 目标扫描匹配法（适合 V4 及以后）
2. estimate_distance_by_target_scan(response, freqs, distance_grid, round_trip=True)

3. estimate_distance_batch(responses, freqs, method, **kwargs)

返回值必须包含：

* distance_est
* intermediate info（如 slope、phase、score curve）

5.10 metrics.py

负责统一计算性能指标。

必须实现：

1. absolute_error(true_d, est_d)
2. squared_error(true_d, est_d)
3. rmse(true_list, est_list)
4. mae(true_list, est_list)
5. median_ae(true_list, est_list)
6. percentile_ae(true_list, est_list, q)
7. success_rate(true_list, est_list, threshold)
8. summarize_errors(true_list, est_list, thresholds=None)

5.11 monte_carlo.py

负责批量仿真与统计。

必须实现：

1. run_monte_carlo(trial_fn, n_trials, seed)
2. collect_trial_results(results_list)
3. summarize_trial_dataframe(df)

要求：

* 每次 trial 结果统一为 dict
* 最终汇总为 pandas.DataFrame

5.12 plotting.py

负责统一绘图。

必须实现：

1. plot_phase_wrapped_unwrapped
2. plot_phase_fit
3. plot_true_vs_estimated
4. plot_snr_vs_rmse
5. plot_histogram_errors
6. plot_boxplot_errors
7. plot_repeats_vs_error
8. plot_num_devices_vs_error
9. plot_distance_gap_vs_success
10. plot_power_gap_vs_error
11. plot_multipath_vs_error
12. plot_kfactor_vs_error
13. plot_score_curve

所有绘图函数应：

* 接收明确数据输入
* 自动保存到指定路径
* 不在函数内部写死实验逻辑

5.13 io_utils.py

负责：

* 保存 json 配置
* 保存 csv 表格
* 保存实验摘要
* 创建结果目录

六、配置系统要求

每个版本必须能通过配置文件独立运行。配置至少包含：

* version_name
* random_seed
* round_trip
* f_start
* f_step
* n_freqs
* distance_grid
* monte_carlo_trials
* save_dir

不同版本需要增加各自专属参数。

示例：

V2:

* snr_db_list

V3:

* n_repeats_list
* random_phase_enable

V4:

* num_devices_list
* target_device_index
* device_distance_range
* power_gap_db_list

V5:

* n_paths_list
* path_gain_db_list

V6:

* rician_k_list
* channel_realizations

七、各版本实验说明与必须实现内容

V0 理想单设备相位测距模型

目标：
验证多频点相位斜率恢复距离机制是否正确。

环境假设：

* 单个主设备与单个从设备
* 固定真实距离
* 单径传播
* 无噪声
* 无随机相位偏置变化
* 无设备干扰

实现要求：

1. 构建多频点单链路复频响
2. 提取 wrapped phase 与 unwrapped phase
3. 进行相位-频率线性拟合
4. 估计时延与距离
5. 扫描距离、频点数、频点间隔

参数扫描：

* 距离范围：0.5 m – 10 m
* 频点数量：10 – 80
* 频点间隔：0.5 MHz – 5 MHz

必须输出：
图表：

* 相位（wrapped / unwrapped）随频率变化图
* 相位-频率线性拟合图
* 真实距离 vs 估计距离曲线

数据表：

* 不同距离下的测距误差
* 不同频点配置下的误差统计

V1 往返测距模型（PBR 近似建模）

目标：
在 V0 基础上构建更符合 PBR 思想的往返测距模型。

环境假设：

* 单个主设备与单个从设备
* 使用往返传播时延 τ_rt = 2d / c
* 允许固定链路相位偏置

实现要求：

1. 支持单程与往返两种模型切换
2. 在往返模式下估计距离
3. 验证固定相位偏置对相位斜率法的影响
4. 输出单程和往返对比

参数扫描：

* 距离范围：1 m – 20 m
* 相位偏置范围：0 – 2π

必须输出：
图表：

* 单程模型 vs 往返模型测距结果对比
* 不同相位偏置下测距误差图

数据表：

* 不同距离下的平均误差
* 偏置存在时的误差统计

V2 噪声环境测距实验

目标：
分析 AWGN 对相位测距精度的影响，验证多频点融合的抗噪性。

环境假设：

* 单设备
* 往返模型
* 加入 AWGN

实现要求：

1. 在频域复响应上加入复高斯白噪声
2. 支持多次 Monte Carlo
3. 统计误差分布与成功率

参数扫描：

* SNR：-5 dB – 30 dB
* Monte Carlo 次数：至少 200

必须输出：
图表：

* SNR vs RMSE 曲线
* 测距误差直方图
* 箱线图

数据表：

* 不同 SNR 下 MAE / RMSE
* 成功测距概率

V3 跳频随机相位偏置实验

目标：
分析链路相位偏置在多次测量中的变化对测距的影响，并比较不同积累策略。

注意：
这里的随机项不是“人为新引入物理”，而是对链路相位偏置在不同测量条件下变化的抽象建模。

环境假设：

* 单设备
* 往返模型
* 支持重复测量
* 每次测量中链路相位偏置可变化

实现要求：

1. 支持多次重复测量
2. 每次测量可生成不同相位偏置
3. 比较以下两种处理：

   * 直接单次估计
   * 重复测量后的简单统计/积累
4. 明确展示偏置变化对稳定性的影响

参数扫描：

* 重复测量次数：1 – 100
* 相位偏置变化范围：0 – 2π

必须输出：
图表：

* 积累次数 vs 测距误差
* 不同处理策略性能对比
* 随机相位分布图

数据表：

* 不同重复次数下 RMSE
* 稳定性指标统计

V4 多设备并发测距模型（核心版本）

目标：
实现论文核心：一主多从并发测距，并验证目标链路匹配补偿能否从叠加信号中恢复目标设备距离。

环境假设：

* 一个主设备
* 多个从设备
* 所有从设备同时返回
* 主设备接收所有链路复信号叠加
* 不依赖一对一 peer 唯一配对消除
* 使用目标匹配补偿与扫描检测恢复目标距离

实现要求：

1. 构造多设备叠加频域响应
2. 支持设置目标设备 index
3. 对目标设备执行距离假设扫描
4. 输出目标匹配评分曲线
5. 从评分最大值估计目标距离
6. 分析设备数、距离差、功率差对性能影响

参数扫描：

* 设备数量：1 – 16
* 设备距离差：0.5 m – 10 m
* 信号功率差：-10 dB – +10 dB

必须输出：
图表：

* 并发设备数量 vs 测距误差
* 距离差 vs 测距成功率
* 功率差 vs 测距误差
* 目标距离扫描评分曲线

数据表：

* 不同设备数量下 RMSE
* 成功率统计
* 目标估计偏差统计

这一版本必须清晰体现：

* 为什么一对一方法不能直接用于并发
* 为什么目标匹配补偿后可以增强目标链路
* 为什么非目标链路在失配条件下表现为扰动项

V5 多径信道实验

目标：
在并发模型上叠加多径传播，评估主径与反射径对目标恢复的影响。

环境假设：

* V4 基础上加入多径
* 每个设备可具有单反射径或多径

实现要求：

1. 为每个设备支持多径建模
2. 支持设置多径数、各径时延、各径增益
3. 分析多径对目标评分曲线和距离估计的扭曲

参数扫描：

* 多径数量：1 – 5
* 反射径功率：-3 dB – -20 dB

必须输出：
图表：

* 单径 vs 多径测距误差对比
* 多径功率 vs 测距误差
* 相位畸变示意图

数据表：

* 不同多径场景下误差统计

V6 室内信道模型实验

目标：
使用更接近真实室内环境的统计衰落模型评估系统稳定性。

环境假设：

* V4 或 V5 基础上
* 使用 Rician / Rayleigh 信道

实现要求：

1. 对链路施加 Rician 或 Rayleigh 衰落
2. 支持多 realizations 的 Monte Carlo
3. 分析 LOS / NLOS 场景下性能变化

参数扫描：

* Rician K 因子：0 – 10
* channel realizations：至少 200

必须输出：
图表：

* K 因子 vs 测距误差
* LOS / NLOS 场景对比
* 误差分布图

数据表：

* 不同信道模型下 RMSE
* 成功率统计

八、实验执行要求

8.1 每个实验脚本必须实现

每个 exp_vX.py 必须：

1. 读取对应配置
2. 生成场景
3. 运行仿真
4. 调用绘图与保存函数
5. 输出简要控制台日志

8.2 main.py 要求

main.py 必须支持：

* 指定版本运行
* 指定配置文件运行
* 指定是否覆盖输出目录
* 指定随机种子

示例命令行形式可为：

python main.py --version v4 --config configs/v4.json

8.3 DataFrame 统一列要求

所有实验的 trial 结果尽量统一包含：

* trial_id
* true_distance
* est_distance
* abs_error
* sq_error
* snr_db
* n_freqs
* f_step
* n_devices
* target_index
* power_gap_db
* n_paths
* k_factor
* seed

九、测试要求

tests 中至少包含：

1. 单链路理想模型测试

   * 已知距离下估计误差应足够小

2. 相位斜率估计器测试

   * 无噪声时可恢复正确距离

3. 目标扫描匹配测试

   * 单设备场景中评分峰值应靠近真实距离

4. 多设备简单场景测试

   * 目标设备比非目标强时应具有可检测峰值

十、README 要求

README.md 必须包含：

1. 项目简介
2. 理论背景
3. 安装方法
4. 运行方法
5. 各版本说明
6. 输出结果说明
7. 目录结构说明

十一、Codex 必须特别注意的事项

1. 不要把“随机相位”错误写成用户发明的新物理机制
   它应被实现为链路相位偏置变化的抽象模型

2. 不要把 V4 写成“直接对混合相位 unwrap 再直线拟合”
   V4 的核心应是“目标匹配补偿/扫描检测”

3. 不要依赖协议栈细节
   本工程是信号级和算法级仿真，不是完整 BLE 协议实现

4. 不要过早引入复杂硬件误差
   当前阶段重点是：

   * 多频点相位
   * 噪声
   * 多设备叠加
   * 多径
   * 室内衰落

5. 所有版本都必须可单独运行、单独出图、单独存档

十二、最终交付物

Codex 最终必须生成一个可以直接运行的 Python 工程，满足：

1. 目录结构完整
2. 所有版本实验脚本齐全
3. 所有核心模块可导入复用
4. 配置文件齐全
5. 可输出图表、数据表、配置快照
6. 至少能跑通一个最小完整流程：
   V0 → V4



请按照README.md ，构建一个“从标准单链路 PBR 相位测距逐步扩展到多设备并发测距”的 Python 仿真工程，其中 V4 是核心：在多设备同时返回造成复信号叠加的条件下，使用目标链路匹配补偿与距离扫描检测方法，从叠加频域响应中恢复目标设备距离，并对噪声、多径和室内衰落条件下的性能进行统计分析。
