# M5-4 显式 HIP Jacobi / 双 stream 产品合同

2026-10-10 用户确认 M5-4。新增本地算法后端；NVIDIA 上游尚无 BFGS。
算法来源是本仓库 M5-2/3A/3B 原创 Apache-2.0 探针，数学仍对齐 ASE 3.29.0。
不是来自 rocSOLVER DC，亦不导入外部 eigh-test 包。冻结探针继续保留为独立对照。

## 选择与数学

登记 `hip.bfgs-ase-jacobi-streams-v1`，family=hip、strategy=jacobi_streams，default_strategy=False，
auto_enabled=False。Framework 的固定晶胞和 UnitCellFilter 类通过现有 backend
参数精确选择，不另加分派表。`hip` 仍选原 rocSOLVER Jacobi；BFGS 默认仍为
torch_reference，None/auto 和 FIRE/FIRE2 语义不变。

三个 executor ABI 为 prepare_state/step/batch_step，与现有 BFGS 相同。
单体系数学不变；Batch 共用原数学表达式，先准备所有候选 H，再调用可选私有
grouped_eigh，检查完整谱/方程/正交/方向后统一提交。失败不提交任何体系的
坐标、Hessian、历史和计数。不加入阻尼、曲率跳过、谱 clamp 或 padding。
通用 State、Filter、ID 管理、restart 格式和模型整 Batch 求值不变。

## 原生边界

输入：连续 strided、无梯度 FP64 HIP `[n,n]` 或 `[B,n,n]`，非空，LOWER。
n 为正且小于 int32 上界，元素偏移必须可用 int64 表示；B 无经验性上限，
实际由内存和表示范围决定。无 147/285 或 B≤4 白名单。公共 Batch API 拒绝空
序列；私有 eigensolver 不接受空列表/空矩阵。UPPER 明确拒绝。
错误 device/dtype/layout/build 明确失败，无静默回退、转换或截断。

固定 Jacobi sweep 上限100、原始范数相对残差1e-15、完整循环配对、256线程
规约、中心化和设备稳定排序沿用已验收动态探针。诊断试验可指定1–100轮。
零轮收敛时，在设备上直接排序原始对角值，避免减/加均值抵消极小非零谱；
该数值边界在M5-4实测修正。迭代分支仍原样使用中心化谱与原偏移恢复。
因此非零轮矩阵五输出仍与原探针逐位对照；零轮对角谱以CPU精确对角排序
为基准，不要求复现原探针的抵消误差。保留负零、升序和重复值稳定列顺序。
五个输出为同设备独立连续 w/V/info/residual/sweeps，形状为
`[...,n]` / 输入shape / `[batch_count]`，诊断整数int32，其他FP64。
V 按列为特征向量，w 升序；选定下三角定义矩阵，未选上三角可含NaN。
输入不变；输出不 alias 输入。info 非零、非有限结果明确报错。
物理单位沿用 BFGS 广义坐标/力；PBC 与邻居不属于 eigensolver。

## 并发与生命周期

单个尺寸组在调用者当前 stream 直接分解，不创建线程池。多个尺寸组按元素量
分配到两个 worker；同一 Python 进程、同一 HCU。私有 scheduler 按设备缓存，
首次多组调用懒创建两条stream/线程池；相同 scheduler 的并发调用串行进入，
避免跨调用工作混淆。各 worker 明确设定 device/current stream。
输入生产 event、输出完成 event 和 record_stream 保护跨stream依赖及分配器寿命。
失败收集所有future并排空所有worker stream后抛错，保证无遗漏在途工作。
原生入口释放GIL；扩展在worker创建前加载。显式私有关闭入口和进程退出回收
线程池；关闭后旧对象不可调用，下一次正常入口可重新创建。

主机仍有调度和每sweep标量同步；不宣称全异步或零CPU。矩阵运算不回传CPU。
稳定排序和同表达式不代表真实MLIP长轨迹跨后端逐位相同。错误诊断允许读取
少量info整数，不回传H/V作为生产计算。两个worker不是输入容量限制。

## 支持与验收

当前准入仅已验证HCU Torch2.9.0+das.opt1.dtk2604/HIP6.3.26093、gfx936。
不升级DTK/Torch。通过torch.library私有custom_op接入，声明非mutation及
fake输出元数据；eager forward-only。autograd拒绝，compile/其他构建/硬件
尚未验收，不因fake通过承诺compile。CPU FP64 reference仍为正确性基线。

验收：原CPU reference门；动态尺寸/B及32真实/合成冻结矩阵；逐步ASE、变胞、
restart、ID重排/冻结/退出/补位；真未收敛/后组失败原子性；producer/consumer及
显存复用；同输入对照原probe；配对性能包含退化案例，不准入auto。
补旧46/92原子MPA-0 medium完整CSP收敛，Li/T2短步独立报告；生产全目录吞吐
不属本轮。命令、退出码、误差、来源hash和未覆盖范围写入阶段报告。
