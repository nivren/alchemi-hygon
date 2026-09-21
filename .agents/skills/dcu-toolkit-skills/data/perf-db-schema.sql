-- DCU Toolkit 性能数据库 Schema
-- 由 dcu-perfdb skill 管理
-- 数据库文件: data/perf.db (SQLite)

-- 设备表
CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_name TEXT NOT NULL,           -- 设备别名 (如 dcu-node-01)
    dcu_type TEXT,                       -- DCU型号
    dcu_count INTEGER,                   -- DCU卡数量
    cpu TEXT,                            -- CPU型号
    memory TEXT,                         -- 内存
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    UNIQUE(device_name)
);

-- 测试记录表
CREATE TABLE IF NOT EXISTS test_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id INTEGER NOT NULL,
    test_date TEXT DEFAULT (datetime('now', 'localtime')),
    test_type TEXT NOT NULL,             -- benchmark | stability | precision | interconnect
    model_name TEXT,                     -- 模型名称
    framework TEXT,                      -- vLLM | SGLang | PyTorch | 其他
    framework_version TEXT,
    dtk_version TEXT,
    batch_size INTEGER,
    precision TEXT,                      -- fp32 | fp16 | int8 | fp8
    tensor_parallel INTEGER,
    cpu_affinity TEXT,                   -- 绑核配置
    throughput REAL,                     -- tokens/s
    ttft REAL,                           -- Time To First Token (ms)
    latency_p50 REAL,                    -- 中位延迟 (ms)
    latency_p99 REAL,                    -- P99延迟 (ms)
    gpu_util REAL,                       -- DCU利用率 (%)
    mem_util REAL,                       -- 显存利用率 (%)
    power_avg REAL,                      -- 平均功耗 (W)
    temperature REAL,                    -- 温度 (C)
    notes TEXT,                          -- 备注
    report_path TEXT,                    -- 关联报告文件路径
    FOREIGN KEY (device_id) REFERENCES devices(id)
);

-- 测试环境详情表
CREATE TABLE IF NOT EXISTS test_env_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_record_id INTEGER NOT NULL,
    os_version TEXT,
    kernel_version TEXT,
    driver_version TEXT,
    topology_info TEXT,                  -- PCIe/NUMA拓扑信息
    hy_smi_snapshot TEXT,               -- hy-smi完整输出
    FOREIGN KEY (test_record_id) REFERENCES test_records(id)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_test_records_device ON test_records(device_id);
CREATE INDEX IF NOT EXISTS idx_test_records_type ON test_records(test_type);
CREATE INDEX IF NOT EXISTS idx_test_records_model ON test_records(model_name);
CREATE INDEX IF NOT EXISTS idx_test_records_date ON test_records(test_date);
