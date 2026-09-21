#!/bin/bash
# hy-smi 常用命令示例脚本
# 使用方法: bash examples.sh [command]
# 不带参数时显示所有示例

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_cmd() {
    echo -e "${YELLOW}[CMD]${NC} $1"
}

# 默认查看所有设备
default_view() {
    echo_info "查看所有设备概览"
    echo_cmd "hy-smi"
    hy-smi
}

# 查看设备详细信息
device_info() {
    echo_info "查看设备ID和序列号"
    echo_cmd "hy-smi -i --showserial --showuniqueid"
    hy-smi -i --showserial --showuniqueid
}

# 查看温度和功耗
temp_power() {
    echo_info "查看温度和功耗"
    echo_cmd "hy-smi -t -P"
    hy-smi -t -P
}

# 查看利用率
utilization() {
    echo_info "查看HCU和CU利用率"
    echo_cmd "hy-smi -u --showhcuutil --showcuutil"
    hy-smi -u --showhcuutil --showcuutil
}

# 查看内存使用
memory() {
    echo_info "查看显存使用情况"
    echo_cmd "hy-smi --showmemuse --showmemavailable"
    hy-smi --showmemuse --showmemavailable
}

# 查看时钟频率
clocks() {
    echo_info "查看时钟频率"
    echo_cmd "hy-smi -c -g --sclk --showclkfrq"
    hy-smi -c -g --sclk --showclkfrq
}

# 查看进程
processes() {
    echo_info "查看正在运行的进程"
    echo_cmd "hy-smi --showpids"
    hy-smi --showpids
}

# JSON格式输出
json_output() {
    echo_info "JSON格式输出所有设备信息"
    echo_cmd "hy-smi -a --json"
    hy-smi -a --json | head -50
}

# 批量查询
batch_query() {
    echo_info "批量查询设备0,1,2所有信息"
    echo_cmd "hy-smi -d 0 1 2 -a"
    hy-smi -d 0 1 2 -a
}

# 设置频率
set_frequency() {
    echo_info "设置设备0的HCU频率等级为7（需先切换到manual模式）"
    echo_cmd "hy-smi -d 0 --setsclk 7 --setperflevel manual"
    hy-smi -d 0 --setsclk 7 --setperflevel manual
}

# 保存配置
save_config() {
    echo_info "保存设备0配置到文件"
    echo_cmd "hy-smi -d 0 --save /tmp/dcu_config.txt"
    hy-smi -d 0 --save /tmp/dcu_config.txt
}

# 错误排查
debug() {
    echo_info "查看hycu相关内核日志"
    echo_cmd "dmesg -T | grep hycu"
    echo "（需要sudo权限）"
    # dmesg -T | grep hycu 2>/dev/null || echo "需要root权限"
}

# 主菜单
show_menu() {
    echo ""
    echo "========================================"
    echo "     hy-smi 常用命令示例脚本"
    echo "========================================"
    echo ""
    echo "使用方法: bash examples.sh [command]"
    echo ""
    echo "可用命令:"
    echo "  default      - 默认查看所有设备"
    echo "  info         - 查看设备ID和序列号"
    echo "  temp         - 查看温度和功耗"
    echo "  util         - 查看利用率"
    echo "  mem          - 查看内存使用"
    echo "  clock        - 查看时钟频率"
    echo "  proc         - 查看进程"
    echo "  json         - JSON格式输出"
    echo "  batch        - 批量查询多卡"
    echo "  freq         - 设置频率"
    echo "  save         - 保存配置"
    echo "  debug        - 错误排查"
    echo "  all          - 运行所有示例（危险！）"
    echo ""
    echo "不带参数时显示本菜单"
    echo ""
}

# 根据参数执行
case "${1:-menu}" in
    default)
        default_view
        ;;
    info)
        device_info
        ;;
    temp)
        temp_power
        ;;
    util)
        utilization
        ;;
    mem)
        memory
        ;;
    clock)
        clocks
        ;;
    proc)
        processes
        ;;
    json)
        json_output
        ;;
    batch)
        batch_query
        ;;
    freq)
        set_frequency
        ;;
    save)
        save_config
        ;;
    debug)
        debug
        ;;
    menu|*)
        show_menu
        ;;
esac