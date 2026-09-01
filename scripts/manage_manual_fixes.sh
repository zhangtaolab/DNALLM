#!/bin/bash
# 手动修复清单管理脚本

MANUAL_FIX_FILE="MANUAL_FIX_REQUIRED.md"

# 显示帮助信息
show_help() {
    echo "手动修复清单管理脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  list                   列出所有待修复问题"
    echo "  check <文件路径>       检查指定文件的问题是否已解决"
    echo "  remove <问题编号>      删除指定问题（如: #001）"
    echo "  status <问题编号> <状态> 更新问题状态（如: #001 已修复）"
    echo "  clean                 清理已解决的问题"
    echo "  help                  显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 list"
    echo "  $0 check tests/configuration/test_configs.py"
    echo "  $0 remove #001"
    echo "  $0 status #001 已修复"
    echo "  $0 clean"
}

# 列出所有待修复问题
list_issues() {
    if [ ! -f "$MANUAL_FIX_FILE" ]; then
        echo "❌ 手动修复清单文件不存在: $MANUAL_FIX_FILE"
        return 1
    fi
    
    echo "📋 手动修复清单中的问题:"
    echo ""
    
    # 提取问题编号和状态（只处理实际的问题，跳过模板）
    sed -n '/^### 待修复问题/,/^### 已修复问题/p' "$MANUAL_FIX_FILE" | grep -E "(问题 #|当前状态)" | while read -r line; do
        if [[ "$line" =~ ^####.*问题.*#([0-9]+) ]]; then
            issue_num="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^-.*当前状态.*:.*(.+) ]]; then
            status="${BASH_REMATCH[1]}"
            echo "问题 #$issue_num: $status"
        fi
    done
}

# 检查指定文件的问题是否已解决
check_file_issues() {
    local target_file="$1"
    
    if [ -z "$target_file" ]; then
        echo "❌ 请指定要检查的文件路径"
        return 1
    fi
    
    if [ ! -f "$MANUAL_FIX_FILE" ]; then
        echo "ℹ️  手动修复清单文件不存在，无需检查"
        return 0
    fi
    
    echo "🔍 检查文件 $target_file 的相关问题..."
    
    # 查找相关问题的行号
    local line_numbers=($(grep -n "文件.*:.*$target_file" "$MANUAL_FIX_FILE" | cut -d: -f1))
    
    if [ ${#line_numbers[@]} -eq 0 ]; then
        echo "✅ 没有发现相关待修复问题"
        return 0
    fi
    
    echo "⚠️  发现 ${#line_numbers[@]} 个相关问题:"
    for line_num in "${line_numbers[@]}"; do
        # 获取问题编号
        local issue_line=$(sed -n "${line_num}p" "$MANUAL_FIX_FILE")
        local issue_num=$(echo "$issue_line" | grep -o "#[0-9]\+")
        echo "  - 问题 $issue_num (行 $line_num)"
    done
    
    echo ""
    echo "💡 建议运行代码质量检查确认问题是否已解决:"
    echo "   ./scripts/check_code_quality.sh $target_file"
}

# 删除指定问题
remove_issue() {
    local issue_num="$1"
    
    if [ -z "$issue_num" ]; then
        echo "❌ 请指定要删除的问题编号（如: #001）"
        return 1
    fi
    
    if [ ! -f "$MANUAL_FIX_FILE" ]; then
        echo "❌ 手动修复清单文件不存在: $MANUAL_FIX_FILE"
        return 1
    fi
    
    # 查找问题所在行
    local issue_line=$(grep -n "问题 $issue_num" "$MANUAL_FIX_FILE" | head -1 | cut -d: -f1)
    
    if [ -z "$issue_line" ]; then
        echo "❌ 未找到问题 $issue_num"
        return 1
    fi
    
    # 查找问题结束行（下一个问题或文件结束）
    local next_issue_line=$(grep -n "问题 #" "$MANUAL_FIX_FILE" | awk -F: -v line="$issue_line" '$1 > line {print $1; exit}')
    
    if [ -z "$next_issue_line" ]; then
        # 删除到文件末尾
        sed -i "${issue_line},\$d" "$MANUAL_FIX_FILE"
    else
        # 删除到下一个问题
        local end_line=$((next_issue_line - 1))
        sed -i "${issue_line},${end_line}d" "$MANUAL_FIX_FILE"
    fi
    
    echo "✅ 已删除问题 $issue_num"
}

# 更新问题状态
update_status() {
    local issue_num="$1"
    local new_status="$2"
    
    if [ -z "$issue_num" ] || [ -z "$new_status" ]; then
        echo "❌ 请指定问题编号和新状态"
        echo "用法: $0 status <问题编号> <新状态>"
        return 1
    fi
    
    if [ ! -f "$MANUAL_FIX_FILE" ]; then
        echo "❌ 手动修复清单文件不存在: $MANUAL_FIX_FILE"
        return 1
    fi
    
    # 查找并更新状态
    if sed -i "s/问题 $issue_num.*当前状态.*:.*/问题 $issue_num (当前状态: $new_status)/" "$MANUAL_FIX_FILE"; then
        echo "✅ 已更新问题 $issue_num 状态为: $new_status"
    else
        echo "❌ 更新失败，请检查问题编号是否正确"
        return 1
    fi
}

# 清理已解决的问题
clean_resolved() {
    if [ ! -f "$MANUAL_FIX_FILE" ]; then
        echo "ℹ️  手动修复清单文件不存在，无需清理"
        return 0
    fi
    
    echo "🧹 清理已解决的问题..."
    
    # 创建临时文件
    local temp_file=$(mktemp)
    
    # 复制非已修复问题到临时文件
    awk '
    /^#### 问题 #/ { in_issue = 1; issue_lines = ""; status = "" }
    in_issue { issue_lines = issue_lines $0 "\n" }
    /当前状态.*:.*已修复/ { status = "resolved" }
    /^#### 问题 #/ && in_issue && status != "resolved" { 
        printf "%s", issue_lines
        in_issue = 0
        issue_lines = ""
    }
    !/^#### 问题 #/ && !in_issue { print }
    END { if (in_issue && status != "resolved") printf "%s", issue_lines }
    ' "$MANUAL_FIX_FILE" > "$temp_file"
    
    # 替换原文件
    mv "$temp_file" "$MANUAL_FIX_FILE"
    
    echo "✅ 清理完成"
}

# 主函数
main() {
    case "${1:-help}" in
        "list")
            list_issues
            ;;
        "check")
            check_file_issues "$2"
            ;;
        "remove")
            remove_issue "$2"
            ;;
        "status")
            update_status "$2" "$3"
            ;;
        "clean")
            clean_resolved
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# 运行主函数
main "$@"
