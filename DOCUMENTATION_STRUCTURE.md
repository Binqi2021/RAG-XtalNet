# 文档结构说明

## 📁 当前文档结构

```
RAG-XtalNet/
├── README.md                    # 项目主说明（面向所有用户）
├── CLAUDE.md                    # Claude Code 使用指导
├── COMPLETE_USER_MANUAL.md      # 📚 完整用户手册（重点推荐！）
└── xtalnet/
    └── retrieval/
        └── README.md            # 检索模块API文档
```

## 🎯 使用指南

### 👶 **新用户**
直接阅读：`COMPLETE_USER_MANUAL.md`
- 包含从环境配置到完整流程的所有内容
- 一份手册搞定一切

### 🤖 **Claude Code用户**
阅读：`CLAUDE.md`
- Claude Code的专用指导文档

### 🔧 **开发者**
1. 先读：`COMPLETE_USER_MANUAL.md`
2. 再看：`xtalnet/retrieval/README.md` （检索API文档）

## ✨ **为什么这样精简？**

- **删除了8个冗余文档**，避免信息重复
- **一本完整手册**替代多个零散指南
- **清晰的层次结构**，不同用户有不同入口
- **减少维护成本**，文档保持最新

## 🗑️ **已删除的冗余文档**

- ❌ `docs/xtalnet_baseline_summary.md`
- ❌ `docs/build_db_usage.md`
- ❌ `scripts/README_DB_BUILDER.md`
- ❌ `docs/retrieval_system_implementation.md`
- ❌ `RAG_IMPLEMENTATION_SUMMARY.md`
- ❌ `docs/rag_integration_guide.md`
- ❌ `docs/alignment_loss_guide.md`
- ❌ `MULTI_DATASET_MINIMAL_CHANGES.md`

**所有这些文档的内容都已经整合到 `COMPLETE_USER_MANUAL.md` 中！**